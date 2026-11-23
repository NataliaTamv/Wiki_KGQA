
import argparse, json, os, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)

STOP = {"is", "are", "was", "were", "be", "been", "did", "do", "does", "has", "have", "had", "can", "could", "will", "would",
        "in", "on", "at", "of", "by", "with", "from", "to", "for", "that", "who", "whom", "whose", "which", "what", "when",
        "where", "why", "how", "and", "or", "the", "a", "an", "not", "also", "ever", "currently", "than", "as", "its", "it",
        "there", "this", "these", "those", "their", "his", "her", "many", "much", "more", "most", "any", "all", "one", "me", "us",
        "s", "'s", "if", "after", "before", "about", "into", "over", "under"}


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", "", (s or "").lower().replace("-", " ")).strip()


def tokens(q):
    return re.findall(r"[A-Za-z0-9][\w'.-]*", q)


def ngrams(q, maxn=3):
    toks = tokens(q)
    out = []
    for n in range(maxn, 0, -1):
        for i in range(0, len(toks) - n + 1):
            seg = toks[i:i + n]
            if seg[0].lower() in STOP or seg[-1].lower() in STOP:
                continue
            if all(w.lower() in STOP for w in seg):
                continue
            out.append(" ".join(seg))
    return out


def gold_qids(sparql):
    return set(re.findall(r"wd:(Q\d+)", sparql or ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dev", default="data/en_no_mentions_dev.json")
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--k", type=int, default=3, help="top-k υποψήφιοι ανά φράση")
    ap.add_argument("--no-aliases", action="store_true")
    ap.add_argument("--all", action="store_true", help="όλες οι ερωτήσεις (default: μόνο οι αποτυχημένες)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    from settings_run import config, pipeline
    if a.endpoint:
        config.set_endpoint(a.endpoint)
    print(f"[endpoint] {config.ENDPOINT_NAME} -> {config.WIKIKGQA_ENDPOINT}")
    el = pipeline.entity_linker
    variants = getattr(pipeline, "_term_variants", lambda s: [s.lower()])

    raw = json.load(open(a.dev, encoding="utf-8"))
    gold = {}
    if isinstance(raw, dict):
        for q in raw["questions"]:
            gold[q["question"][0]["string"]] = q["sparql"]
    else:
        for q in raw:
            if q.get("gold_sparql"):
                gold[q["question"]] = q["gold_sparql"]
    rows = [json.loads(l) for l in open(a.run, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if a.all or r["f1"] < 0.999999]
    if a.limit:
        rows = rows[:a.limit]

    cache = {}

    def lookup(term):
        if term in cache:
            return cache[term]
        found = []
        try:
            for c in el.search_wikidata(term, "item", 20):
                if norm(c.get("label")) == norm(term) and c["id"] not in found:
                    found.append(c["id"])
        except Exception as e:
            print(f"  [σφάλμα {term!r}: {e}]")
        if not a.no_aliases:
            try:
                for c in el._alias_candidates(term):
                    if c["id"] not in found:
                        found.append(c["id"])
            except Exception:
                pass
        cache[term] = found[:a.k]
        return cache[term]

    tot_gold = rec_mention = rec_ngram_extra = 0
    noise = []
    per_q = []
    for i, r in enumerate(rows, 1):
        g = gold.get(r["question"])
        if not g:
            continue
        gq = gold_qids(g)
        mentions = (r.get("diag", {}).get("mentions") or {}).get("entities", [])
        mtxt = [m.lower() for m in mentions]
        covered = set()
        for c in (r.get("diag", {}).get("entities") or {}).values():
            covered |= {x[0] for x in c}
        added, by_term = set(), {}
        for ng in ngrams(r["question"]):
            if any(ng.lower() in m for m in mtxt):        
                continue
            for term in variants(ng):
                ids = lookup(term)
                if ids:
                    by_term.setdefault(ng, set()).update(ids)
                    added |= set(ids)
        new = added - covered
        hit_m = gq & covered
        hit_n = (gq & new)
        tot_gold += len(gq); rec_mention += len(hit_m); rec_ngram_extra += len(hit_n)
        noise.append(len(new))
        per_q.append((r["question"], gq, hit_m, hit_n, len(new), by_term))
        print(f"[{i}/{len(rows)}] +{len(new):2} υποψήφια, gold από n-grams: {sorted(hit_n)}  | {r['question'][:55]}", flush=True)

    n = len(per_q) or 1
    print("\n" + "=" * 60)
    print(f"Ερωτήσεις: {len(per_q)}   gold QIDs συνολικά: {tot_gold}")
    print(f"  ήδη καλύπτονται από mentions (top-3 του linking): {rec_mention}")
    print(f"  ΕΠΙΠΛΕΟΝ ανάκτηση από n-grams (k={a.k}):           {rec_ngram_extra}")
    print(f"  ακόμα χαμένα:                                     {tot_gold - rec_mention - rec_ngram_extra}")
    print(f"Θόρυβος: μέσος όρος νέων υποψηφίων ανά ερώτηση = {sum(noise)/n:.1f}  (max {max(noise) if noise else 0})")
    print("\nΑκόμα χαμένα ανά ερώτηση (gold QIDs που δεν καλύπτονται ούτε από mentions ούτε από n-grams):")
    for q, gq, hm, hn, nn, bt in per_q:
        miss = gq - hm - hn
        if miss:
            print(f"  {q[:70]}  -> {sorted(miss)}")
    print("\nΑνακτήθηκαν από n-grams:")
    for q, gq, hm, hn, nn, bt in per_q:
        if hn:
            print(f"  {q[:70]}  -> {sorted(hn)}")
            for ng, ids in bt.items():
                if ids & hn:
                    print(f"       φράση {ng!r}: {sorted(ids)}")


if __name__ == "__main__":
    main()