
import argparse
import glob
import os
import sqlite3

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dir")
    ap.add_argument("faiss_index")
    ap.add_argument("sqlite")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nprobe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import faiss

    paths = sorted(glob.glob(os.path.join(a.emb_dir, "emb_*.npy")))
    mm = [np.load(p, mmap_mode="r") for p in paths]
    offsets = np.concatenate([[0], np.cumsum([x.shape[0] for x in mm])])
    total = int(offsets[-1])
    index = faiss.read_index(a.faiss_index)
    if hasattr(index, "nprobe"):
        index.nprobe = a.nprobe
    con = sqlite3.connect(a.sqlite)
    n_sql = con.execute("SELECT MAX(rowid) FROM labels").fetchone()[0]

    print(f"shards: {len(mm)} ({total:,} γραμμές) | FAISS: {index.ntotal:,} | SQLite: {n_sql:,}")
    counts_ok = total == index.ntotal == n_sql
    print("Πλήθη ίδια: " + ("ΝΑΙ" if counts_ok else "ΟΧΙ  <-- πρόβλημα"))

    rng = np.random.default_rng(a.seed)
    rows = np.sort(rng.choice(total, size=min(a.n, total), replace=False))
    # πάντα και η πρώτη/τελευταία γραμμή (όρια)
    rows = np.unique(np.concatenate([rows, [0, total - 1]]))

    ids_cache = {}

    def shard_ids(si):
        if si not in ids_cache:
            tag = os.path.basename(paths[si])[len("emb_"):-len(".npy")]
            with open(os.path.join(a.emb_dir, f"ids_{tag}.txt"), encoding="utf-8") as f:
                ids_cache[si] = f.read().split("\n")
        return ids_cache[si]

    mismatch, top1_exact, top10_exact, top10_label = [], 0, 0, 0
    for g in rows:
        g = int(g)
        si = int(np.searchsorted(offsets, g, side="right") - 1)
        r = g - int(offsets[si])
        shard_qid = shard_ids(si)[r]
        sql = con.execute("SELECT qid, label FROM labels WHERE rowid=?", (g + 1,)).fetchone()
        if sql is None or f"Q{sql[0]}" != shard_qid:
            mismatch.append((g, shard_qid, sql))
            continue
        v = np.asarray(mm[si][r], dtype=np.float32)[None, :]
        _, I = index.search(v, 10)
        found = [int(x) for x in I[0] if x >= 0]
        top1_exact += bool(found) and found[0] == g
        top10_exact += g in found
        same = False
        for h in found:
            lab = con.execute("SELECT label FROM labels WHERE rowid=?", (h + 1,)).fetchone()
            if lab and lab[0] == sql[1]:
                same = True
                break
        top10_label += same

    checked = len(rows) - len(mismatch)
    print(f"\nΈλεγχος 1: {len(rows)} γραμμές, ασυμφωνίες QID shard↔SQLite: {len(mismatch)}")
    for m in mismatch[:5]:
        print("   ασυμφωνία:", m)
    if checked:
        print(f"Έλεγχος 2 (στις {checked} γραμμές που ταίριαζαν):")
        print(f"   η ίδια γραμμή είναι #1 στο FAISS:            {100 * top1_exact / checked:5.1f}%")
        print(f"   η ίδια γραμμή είναι στο top-10:              {100 * top10_exact / checked:5.1f}%")
        print(f"   βρέθηκε οντότητα με το ίδιο label στο top-10: {100 * top10_label / checked:5.1f}%")
    ok = counts_ok and not mismatch
    print("\nΑΠΟΤΕΛΕΣΜΑ: " + ("ΟΚ -- ευθυγράμμιση σωστή." if ok else "ΠΡΟΒΛΗΜΑ -- δες τα παραπάνω."))


if __name__ == "__main__":
    main()