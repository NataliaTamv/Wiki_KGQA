
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   #
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)


def gold_entities(sparql):
    
    classes = set(re.findall(r"P(?:31|279)[*+?]*(?:/wdt:P279[*+?]*)?\s+wd:(Q\d+)", sparql))
    return set(re.findall(r"wd:(Q\d+)", sparql)) - classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/run_<tag>.jsonl")
    ap.add_argument("--dev", default="data/en_no_mentions_dev.json")
    ap.add_argument("--endpoint", default=None, help="endp1 / endp2 ")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--depth", type=int, default=50)
    a = ap.parse_args()

    from settings_run import config, pipeline
    if a.endpoint:
        config.set_endpoint(a.endpoint)
    print(f"[endpoint] {config.ENDPOINT_NAME} -> {config.WIKIKGQA_ENDPOINT}")
    el = pipeline.entity_linker

    raw = json.load(open(a.dev, encoding="utf-8"))
    gold = {}
    if isinstance(raw, dict):                      
        for q in raw["questions"]:
            gold[q["question"][0]["string"]] = q["sparql"]
    else:                                           
        for q in raw:
            if q.get("gold_sparql"):
                gold[q["question"]] = q["gold_sparql"]
    if not gold:
        sys.exit("Το αρχείο --dev δεν έχει gold SPARQL. Χρησιμοποίησε το data/en_no_mentions_dev.json.")
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]

    buckets = {"rank 1": 0, "rank 2-3": 0, "rank 4-10": 0, f"rank 11-{a.depth}": 0, "ΔΕΝ βρέθηκε": 0}
    lost = []
    total = 0
    for r in rows:
        if r["f1"] >= 0.999999 and not a.all:
            continue
        g = gold.get(r["question"])
        if not g:
            continue
        ents = gold_entities(g)
        if not ents:
            continue
        mentions = (r.get("diag", {}).get("mentions") or {}).get("entities", [])
        ranks = {}
        tops = {}
        for m in mentions:
            try:
                cands = el.search_wikidata(m, "item", a.depth)
            except Exception as e:
                print(f"  [σφάλμα στο {m!r}: {e}]")
                continue
            tops[m] = [(c["id"], c.get("label")) for c in cands[:3]]
            for pos, c in enumerate(cands, 1):
                if c["id"] in ents and (c["id"] not in ranks or pos < ranks[c["id"]]):
                    ranks[c["id"]] = pos
        for qid in sorted(ents):
            total += 1
            pos = ranks.get(qid)
            if pos is None:
                buckets["ΔΕΝ βρέθηκε"] += 1
                lost.append((r["question"], qid, mentions, tops))
            elif pos == 1:
                buckets["rank 1"] += 1
            elif pos <= 3:
                buckets["rank 2-3"] += 1
            elif pos <= 10:
                buckets["rank 4-10"] += 1
            else:
                buckets[f"rank 11-{a.depth}"] += 1

    print(f"\nGold entities που ελέγχθηκαν: {total}")
    for k, v in buckets.items():
        print(f"  {k:14} {v:4}")
    print("\nΧαμένα (ο gold QID δεν βρίσκεται σε κανένα mention του LLM):")
    for q, qid, ments, tops in lost:
        print(f"\n  {q}\n    gold: {qid}   mentions: {ments}")
        for m, t in tops.items():
            print(f"      {m!r} -> {t}")


if __name__ == "__main__":
    main()