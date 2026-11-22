
import argparse
import bisect
import glob
import os
import re
import sqlite3
import sys
import unicodedata

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   
_CHUNK = 200_000                                           
_WS = re.compile(r"\s+")


def normalize(text):
    
    if text.isascii():
        return _WS.sub(" ", text.lower()).strip()
    t = unicodedata.normalize("NFD", text.casefold())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return _WS.sub(" ", t).strip()


#build: SQLite
def build_exact(tsv, db_path, batch=500_000):
    
    from tqdm import tqdm

    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA cache_size=-1000000")                 # ~1 GB cache
    con.execute("CREATE TABLE labels(qid INTEGER, label TEXT, norm TEXT)")

    n, rows = 0, []
    with open(tsv, encoding="utf-8-sig") as f:
        for line in tqdm(f, desc="sqlite insert", unit="lbl", mininterval=5):
            qid, _, label = line.rstrip("\n").partition("\t")
            try:
                q = int(qid[1:]) if qid[:1] == "Q" else 0
            except ValueError:
                q = 0
            rows.append((q, label, normalize(label)))
            n += 1
            if len(rows) >= batch:
                con.executemany("INSERT INTO labels VALUES (?,?,?)", rows)
                con.commit()
                rows = []
    if rows:
        con.executemany("INSERT INTO labels VALUES (?,?,?)", rows)
        con.commit()

    print(f"Δημιουργία ευρετηρίου ...",
          file=sys.stderr)
    con.execute("CREATE INDEX idx_norm ON labels(norm, qid)")
    con.commit()
    con.close()
    print(f"Έτοιμο: {n:,} γραμμές στο {db_path}", file=sys.stderr)


#build: FAISS
def _write_checkpoint(index, path):
    import faiss
    faiss.write_index(index, path + ".tmp")
    os.replace(path + ".tmp", path)


def build_faiss(emb_dir, index_path, nlist=16384, m=None, train_size=2_000_000,
                checkpoint_every=10, stop_after=0, seed=0):
    
    import faiss
    from tqdm import tqdm

    paths = sorted(glob.glob(os.path.join(emb_dir, "emb_*.npy")))
    if not paths:
        raise FileNotFoundError(f"Δεν βρέθηκαν emb_*.npy στο {emb_dir}")
    mm = [np.load(p, mmap_mode="r") for p in paths]
    sizes = [x.shape[0] for x in mm]
    dim = mm[0].shape[1]
    m = m or dim // 8
    if dim % m:
        raise ValueError(f"Οι διαστάσεις ({dim}) πρέπει να διαιρούνται με το m ({m})")

    partial = index_path + ".partial"
    done = 0
    quantizer = faiss.IndexFlatIP(dim)                     # να μένει ζωντανό όσο ζει το index
    if os.path.exists(partial):
        index = faiss.read_index(partial)
        cum = np.concatenate([[0], np.cumsum(sizes)])
        hits = np.where(cum == index.ntotal)[0]
        if len(hits) == 0:
            raise SystemExit(f"Το {partial} έχει {index.ntotal} διανύσματα, που δεν αντιστοιχούν σε όριο ")
        done = int(hits[0])
        print(f"Συνέχεια από το shard {done} ({index.ntotal:,} διανύσματα ήδη)", file=sys.stderr)
    else:
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        per = max(1, -(-train_size // len(mm)))
        parts = []
        for x in tqdm(mm, desc="sample", unit="shard"):
            k = min(per, x.shape[0])
            idx = np.sort(rng.choice(x.shape[0], size=k, replace=False))
            parts.append(np.asarray(x[idx], dtype=np.float32))
        sample = np.ascontiguousarray(np.vstack(parts))
        print(f"Εκπαίδευση σε {sample.shape[0]:,} διανύσματα (nlist={nlist}, m={m})...", file=sys.stderr)
        index.train(sample)
        del sample, parts
        _write_checkpoint(index, partial)                  

    pbar = tqdm(total=len(mm), initial=done, desc="add", unit="shard")
    for si in range(done, len(mm)):
        emb = mm[si]
        for start in range(0, emb.shape[0], 100_000):
            index.add(np.asarray(emb[start:start + 100_000], dtype=np.float32))
        pbar.update(1)
        last = si + 1 == len(mm)
        if not last and checkpoint_every and (si + 1) % checkpoint_every == 0:
            _write_checkpoint(index, partial)
        if not last and stop_after and (si + 1) >= stop_after:
            _write_checkpoint(index, partial)
            print(f"Διακοπή μετά από {si + 1} shards (--stop-after).", file=sys.stderr)
            return
    pbar.close()

    faiss.write_index(index, index_path)
    if os.path.exists(partial):
        os.remove(partial)
    print(f"Έτοιμο: {index.ntotal:,} διανύσματα στο {index_path}", file=sys.stderr)


#search
def _read_faiss_index(path):
   
    import faiss
    with open(path, "rb") as f:
        return faiss.read_index(faiss.PyCallbackIOReader(f.read))


class LocalEntityIndex:
    def __init__(self, emb_dir, faiss_path=None, sqlite_path=None, model_name=DEFAULT_MODEL,
                 embed_fn=None, max_len=32, nprobe=32, rerank_k=100):
        self.shards = []
        for p in sorted(glob.glob(os.path.join(emb_dir, "emb_*.npy"))):
            tag = os.path.basename(p)[len("emb_"):-len(".npy")]
            self.shards.append({
                "emb": np.load(p, mmap_mode="r"),
                "ids_path": os.path.join(emb_dir, f"ids_{tag}.txt"),
                "labels_path": os.path.join(emb_dir, f"labels_{tag}.txt"),
                "ids": None,
                "labels": None,
            })
        if not self.shards:
            raise FileNotFoundError(f"Δεν βρέθηκαν emb_*.npy στο {emb_dir}")
        self.offsets = [0]
        for sh in self.shards:
            self.offsets.append(self.offsets[-1] + sh["emb"].shape[0])

        self.rerank_k = rerank_k
        self.faiss = None
        if faiss_path:
            self.faiss = _read_faiss_index(faiss_path)
            if self.faiss.ntotal != self.offsets[-1]:
                raise ValueError(
                f"Το ευρετήριο FAISS έχει {self.faiss.ntotal:,} διανύσματα αλλά τα shards έχουν {self.offsets[-1]:,}")
            if hasattr(self.faiss, "nprobe"):
                self.faiss.nprobe = nprobe

        self.db = None
        if sqlite_path:
            self.db = sqlite3.connect(sqlite_path)
            self.db.execute("PRAGMA query_only=ON")

        self._embed_fn = embed_fn
        self._model_name = model_name
        self._max_len = max_len
        self._tok = self._model = None

 
    def _embed(self, texts):
        if self._embed_fn is not None:
            return np.asarray(self._embed_fn(texts), dtype=np.float32)
        import torch
        from transformers import AutoModel, AutoTokenizer
        if self._model is None:
            self._tok = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name).eval()
        with torch.no_grad():
            b = self._tok(texts, padding=True, truncation=True,
                          max_length=self._max_len, return_tensors="pt")
            h = self._model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1)                       
            v = torch.nn.functional.normalize(v, dim=-1)
        return v.numpy().astype(np.float32)

   
    def _load_text(self, shard, key):
        if shard[key] is None:
            with open(shard[f"{key}_path"], encoding="utf-8") as f:
                shard[key] = f.read().split("\n")
        return shard[key]

    def _lookup(self, g):
        
        if self.db is not None:
            row = self.db.execute("SELECT qid, label FROM labels WHERE rowid=?", (g + 1,)).fetchone()
            if row is None:
                raise KeyError(f"Δεν υπάρχει γραμμή {g + 1} στην SQLite")
            return f"Q{row[0]}", row[1]
        si = bisect.bisect_right(self.offsets, g) - 1
        r = g - self.offsets[si]
        sh = self.shards[si]
        return self._load_text(sh, "ids")[r], self._load_text(sh, "labels")[r]

   
    def _exact(self, mention, limit):
        if self.db is None:
            return []
        norm = normalize(mention)
        if not norm:
            return []
        
        rows = self.db.execute(
            "SELECT qid, label FROM labels WHERE norm=? ORDER BY qid LIMIT ?", (norm, limit)
        ).fetchall()
        return [(f"Q{q}", label) for q, label in rows]

    # ----------------------------------------------------- mention expansion
    def expand_mention(self, question, mention, max_extra=2, limit=10):
        
        if self.db is None:
            return None
        edge_strip = "?!,;:\"'()"
        toks = question.split()
        norm = [t.strip(edge_strip).lower() for t in toks]
        mt = [t.strip(edge_strip).lower() for t in mention.split()]
        if not mt:
            return None
        n = len(mt)
        start = next((i for i in range(len(norm) - n + 1) if norm[i:i + n] == mt), None)
        if start is None:
            return None                                         # το LLM άλλαξε το κείμενο, δεν εντοπίζεται
        stop = {"of", "the", "a", "an", "in", "on", "at", "to", "by", "for", "is", "was", "and", "or"}
        spans = []
        for left in range(0, max_extra + 1):
            for right in range(0, max_extra + 1):
                if left == 0 and right == 0:
                    continue
                i, j = start - left, start + n + right
                if i < 0 or j > len(toks):
                    continue
                words = [t.strip(edge_strip) for t in toks[i:j]]
                if words[0].lower() in stop or words[-1].lower() in stop:
                    continue
                spans.append((left + right, " ".join(words)))
        for _, text in sorted(spans, key=lambda x: -x[0]):      # πρώτα τα μακρύτερα
            found = self._exact(text, limit)
            if found:
                return text, found
        return None

    
    def _semantic(self, q, n):
        
        scored = []
        if self.faiss is not None:
            _, I = self.faiss.search(q[None, :], max(self.rerank_k, n))
            for g in I[0]:
                g = int(g)
                if g < 0:
                    continue
                si = bisect.bisect_right(self.offsets, g) - 1
                v = np.asarray(self.shards[si]["emb"][g - self.offsets[si]], dtype=np.float32)
                scored.append((float(v @ q), g))
        else:                                                   # brute force (μικρά σύνολα / δοκιμές)
            for si, sh in enumerate(self.shards):
                emb = sh["emb"]
                for start in range(0, emb.shape[0], _CHUNK):
                    block = np.asarray(emb[start:start + _CHUNK], dtype=np.float32)
                    sc = block @ q
                    k = min(n, sc.shape[0])
                    top = np.argpartition(sc, -k)[-k:]
                    scored.extend((float(sc[i]), self.offsets[si] + start + int(i)) for i in top)
        scored.sort(key=lambda x: -x[0])
        return scored[:n]

   
    def search(self, mention, limit=5, semantic=True):
        results, seen = [], set()
        for qid, label in self._exact(mention, limit):
            results.append({"id": qid, "label": label, "description": None,
                            "score": 1.0, "match": "exact"})
            seen.add(qid)
        if len(results) >= limit or not semantic:
            return results[:limit]

        q = self._embed([mention])[0]
        for score, g in self._semantic(q, limit + len(results)):
            qid, label = self._lookup(g)
            if qid in seen:
                continue
            seen.add(qid)
            results.append({"id": qid, "label": label, "description": None,
                            "score": score, "match": "semantic"})
            if len(results) >= limit:
                break
        return results



_INDEX = None


def _get_index():
    global _INDEX
    if _INDEX is None:
        from settings_run import config
        _INDEX = LocalEntityIndex(
            getattr(config, "EMBEDDINGS_DIR", "out_dir"),
            faiss_path=getattr(config, "FAISS_INDEX_PATH", None),
            sqlite_path=getattr(config, "SQLITE_PATH", None),
            model_name=getattr(config, "EMBEDDING_MODEL", DEFAULT_MODEL),
        )
    return _INDEX


def search_local(mention, limit=10):
   
    return _get_index().search(mention, limit)


def expand_mention_local(question, mention, limit=10):
    
    out = _get_index().expand_mention(question, mention, limit=limit)
    if out is None:
        return None
    span, found = out
    return span, [{"id": qid, "label": label, "description": None,
                   "score": 1.0, "match": "exact_expanded"} for qid, label in found]


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("build-exact", help="labels.tsv -> SQLite για ακριβές ταίριασμα")
    e.add_argument("tsv")
    e.add_argument("db")

    f = sub.add_parser("build-faiss", help="shards -> ευρετήριο FAISS IVF-PQ")
    f.add_argument("emb_dir")
    f.add_argument("index")
    f.add_argument("--nlist", type=int, default=16384)
    f.add_argument("--m", type=int, default=None, help="υπο-διανύσματα PQ (προεπιλογή: διαστάσεις/8)")
    f.add_argument("--train-size", type=int, default=2_000_000)
    f.add_argument("--checkpoint-every", type=int, default=10)
    f.add_argument("--stop-after", type=int, default=0, help="μόνο για δοκιμές")

    s = sub.add_parser("search", help="δοκιμαστική αναζήτηση")
    s.add_argument("mention")
    s.add_argument("--emb-dir", required=True)
    s.add_argument("--faiss", default=None)
    s.add_argument("--sqlite", default=None)
    s.add_argument("--model", default=DEFAULT_MODEL)
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--no-semantic", action="store_true")

    a = ap.parse_args()
    if a.cmd == "build-exact":
        build_exact(a.tsv, a.db)
    elif a.cmd == "build-faiss":
        build_faiss(a.emb_dir, a.index, nlist=a.nlist, m=a.m, train_size=a.train_size,
                    checkpoint_every=a.checkpoint_every, stop_after=a.stop_after)
    else:
        idx = LocalEntityIndex(a.emb_dir, faiss_path=a.faiss, sqlite_path=a.sqlite, model_name=a.model)
        for r in idx.search(a.mention, a.limit, semantic=not a.no_semantic):
            print(f"{r['id']:>12}  {r['score']:.3f}  {r['match']:<8}  {r['label']}")


if __name__ == "__main__":
    main()