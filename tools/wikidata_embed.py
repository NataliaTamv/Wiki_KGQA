
import argparse
import bz2
import gzip
import os
import subprocess
import sys

try:
    import orjson as _json
    loads = _json.loads
except ImportError:
    import json as _json
    loads = _json.loads

from tqdm import tqdm



def open_dump(path):
    
    if path.endswith(".bz2"):
        for tool in ("lbzip2", "pbzip2"):
            if subprocess.run(["which", tool], capture_output=True).returncode == 0:
                p = subprocess.Popen([tool, "-dc", path], stdout=subprocess.PIPE, bufsize=1 << 20)
                return p.stdout
        return bz2.open(path, "rb")
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def pick_label(labels, langs):
    for lang in langs:
        v = labels.get(lang)
        if v:
            return v["value"]
    return None


import re

_QID = re.compile(rb"wd:(Q\d+) a wikibase:Item")
_STR = re.compile(rb'"((?:[^"\\]|\\.)*)"@([A-Za-z0-9-]+)')


def _unescape(b):
    s = b.decode("utf-8", "replace")
    if "\\" in s:
        s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
        s = re.sub(r"\\[tnr]", " ", s)
        s = re.sub(r"\\(.)", r"\1", s)
    return s


def extract_ttl(args):
    
    wanted = set(args.langs)
    qid, found = None, {}
    pred = None
    n_seen = n_kept = 0

    with open_dump(args.dump) as f, open(args.out, "w", encoding="utf-8") as out:

        def flush():
            nonlocal n_kept
            if qid is None:
                return
            for lang in args.langs:                      # σειρά προτίμησης
                if lang in found:
                    label = _unescape(found[lang]).strip()
                    if label:
                        out.write(f"{qid}\t{label}\n")
                        n_kept += 1
                    return

        for raw in tqdm(f, desc="lines", unit="ln", mininterval=5):
            c = raw[:1]
            if c == b"\t":
                if qid is None:
                    continue
                if raw[1:2] == b"\t":                    
                    if pred != b"rdfs:label":
                        continue
                else:
                    parts = raw[1:].split(None, 1)
                    pred = parts[0] if parts else None
                    if pred != b"rdfs:label":
                        continue
                for text, lang in _STR.findall(raw):
                    l = lang.decode()
                    if l in wanted and l not in found:
                        found[l] = text
            elif c in (b"\n", b"\r", b""):
                continue
            else:                                      
                flush()
                qid, found, pred = None, {}, None
                m = _QID.match(raw)
                if m:
                    qid = m.group(1).decode()
                    n_seen += 1
                if args.limit and n_kept >= args.limit:
                    break
        flush()
    print(f"Βρέθηκαν {n_seen:,} items, κρατήθηκαν {n_kept:,} με label", file=sys.stderr)


_RDFS_LABEL = (b"rdfs:label", b"<http://www.w3.org/2000/01/rdf-schema#label>")
_QID_SUBJ = re.compile(rb"(?:wd:|entity/)(Q\d+)>?$")


def extract_ttl_generic(args):
 
    rank = {l: i for i, l in enumerate(args.langs)}
    best = {}                     # qid -> (rank, label_bytes)
    subj = pred = None

    with open_dump(args.dump) as f:
        for raw in tqdm(f, desc="lines", unit="ln", mininterval=5):
            c = raw[:1]
            if c in (b"\n", b"\r", b"", b"@", b"#"):
                continue
            if c in (b"\t", b" "):
                s = raw.lstrip()
                if s[:1] != b'"':                        
                    parts = s.split(None, 1)
                    pred = parts[0] if parts else None
            else:                                       
                parts = raw.split(None, 2)
                subj = parts[0]
                pred = parts[1] if len(parts) > 1 else None

            if pred in _RDFS_LABEL and subj is not None:
                m = _QID_SUBJ.search(subj)
                if not m:
                    continue
                qid = m.group(1).decode()
                for text, lang in _STR.findall(raw):
                    r = rank.get(lang.decode())
                    if r is None:
                        continue
                    cur = best.get(qid)
                    if cur is None or r < cur[0]:
                        best[qid] = (r, text)
                if args.limit and len(best) >= args.limit:
                    break

    n_kept = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for qid, (_, text) in best.items():
            label = _unescape(text).strip()
            if label:
                out.write(f"{qid}\t{label}\n")
                n_kept += 1
    print(f"Κρατήθηκαν {n_kept:,} items με label", file=sys.stderr)


def extract(args):
    if args.dump.endswith((".ttl", ".ttl.bz2", ".ttl.gz")):
        return extract_ttl_generic(args)
    return extract_json(args)


def extract_json(args):
    n_seen = n_kept = 0
    with open_dump(args.dump) as f, open(args.out, "w", encoding="utf-8") as out:
        for line in tqdm(f, desc="entities", unit="ent", mininterval=5):
            line = line.strip()
            if line in (b"[", b"]", b""):
                continue
            if line.endswith(b","):
                line = line[:-1]
            n_seen += 1
            try:
                ent = loads(line)
            except Exception:
                continue
            qid = ent.get("id", "")
            if not qid.startswith("Q"):           
                continue
            label = pick_label(ent.get("labels", {}), args.langs)
            if not label:
                continue
            label = label.replace("\t", " ").replace("\n", " ").strip()
            if label:
                out.write(f"{qid}\t{label}\n")
                n_kept += 1
            if args.limit and n_kept >= args.limit:
                break
    print(f"Διαβάστηκαν {n_seen:,} οντότητες, κρατήθηκαν {n_kept:,}", file=sys.stderr)



def embed(args):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).eval().to(device)
    if device == "cuda":
        model.half()

    @torch.no_grad()
    def encode(texts):
        b = tok(texts, padding=True, truncation=True, max_length=args.max_len,
                return_tensors="pt").to(device)
        h = model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)                     
        if args.normalize:
            v = torch.nn.functional.normalize(v, dim=-1)
        return v.float().cpu().numpy().astype(np.float16)

    pbar = tqdm(desc="embed", unit="lbl", mininterval=5)
    shard_idx, buf_ids, buf_txt = 0, [], []

    def flush():
        nonlocal shard_idx, buf_ids, buf_txt
        if not buf_ids:
            return
       
        order = sorted(range(len(buf_txt)), key=lambda i: len(buf_txt[i]))
        vecs = np.empty((len(buf_txt), model.config.hidden_size), dtype=np.float16)
        for s in range(0, len(order), args.batch_size):
            idx = order[s:s + args.batch_size]
            vecs[idx] = encode([buf_txt[i] for i in idx])
            pbar.update(len(idx))
        
        with open(os.path.join(args.out_dir, f"ids_{shard_idx:05d}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(buf_ids))
        np.save(os.path.join(args.out_dir, f"emb_{shard_idx:05d}.npy"), vecs)
        shard_idx += 1
        buf_ids, buf_txt = [], []

    done = len([x for x in os.listdir(args.out_dir) if x.startswith("emb_")])
    skip = done * args.shard_size
    if done:
        print(f"Συνέχεια από shard {done} (παράλειψη {skip:,} γραμμών)", file=sys.stderr)
        shard_idx = done

    with open(args.tsv, encoding="utf-8-sig") as f:      # utf-8-sig: αγνοεί ένα πιθανό BOM
        for i, line in enumerate(f):
            if i < skip:
                continue
            qid, _, label = line.rstrip("\n").partition("\t")
            buf_ids.append(qid)
            buf_txt.append(label)
            if len(buf_ids) >= args.shard_size:
                flush()
    flush()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract")
    e.add_argument("dump")
    e.add_argument("out")
    e.add_argument("--langs", nargs="+", default=["el", "en"])
    e.add_argument("--limit", type=int, default=0)
    e.set_defaults(fn=extract)

    m = sub.add_parser("embed")
    m.add_argument("tsv")
    m.add_argument("out_dir")
    m.add_argument("--model", default="bert-base-multilingual-cased")
    m.add_argument("--batch-size", type=int, default=512)
    m.add_argument("--shard-size", type=int, default=1_000_000)
    m.add_argument("--max-len", type=int, default=32)
    m.add_argument("--normalize", action="store_true")
    m.set_defaults(fn=embed)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()