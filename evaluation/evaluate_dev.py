import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)   

#   soft-hint 0     χωρίς soft hint (σχέσεις μέσω API, όπως πριν)
#   soft-hint 1     με soft hint
#   soft-hint both  τρέχει και τα δύο στις ΙΔΙΕΣ ερωτήσεις και τα συγκρίνει

import argparse
import json
import os
import random
import time
from collections import defaultdict

from src import config, few_shot_selector, pipeline, sparql_executor, sparql_generator

try:
    from src.evaluate import compute_prf1, evaluate_dataset, flatten_results
except ImportError:                                  # αν το evaluate.py είναι στον ριζικό φάκελο
    from evaluate import compute_prf1, evaluate_dataset, flatten_results

GOLD_CACHE = "data/cache/gold_cache.json"


_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "wikibase": "http://wikiba.se/ontology#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "schema": "http://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "bd": "http://www.bigdata.com/rdf#",
}


def with_prefixes(query):
    import re
    declared = {m.group(1).lower() for m in re.finditer(r"PREFIX\s+([A-Za-z][\w-]*)\s*:", query, re.IGNORECASE)}
    lines = [f"PREFIX {k}: <{v}>" for k, v in _PREFIXES.items()
             if k not in declared and re.search(rf"(?<![\w:?$]){re.escape(k)}:", query)]
    return "\n".join(lines) + "\n" + query if lines else query


def load_questions(path, language="en"):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out = []
    for item in data.get("questions", []):
        sparql = (item.get("sparql") or "").strip()
        texts = [q["string"] for q in item.get("question", []) if q.get("language") == language]
        if sparql and texts:
            out.append({"question": texts[0], "sparql": sparql})
    return out


def load_cache():
    try:
        with open(GOLD_CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_cache(cache):
    with open(GOLD_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)


def gold_values(item, cache):
    key = item["question"]
    if key not in cache:
        rows, error = sparql_executor.run_sparql_checked(with_prefixes(item["sparql"]))
        if error:                                    
            return None
        cache[key] = flatten_results(rows)
        save_cache(cache)
    return cache[key]


def install_leave_one_out():
    
    def patched(question, training_data, k=8):
        top = few_shot_selector.select_top_k(question, training_data, k + 1)
        q = question.strip().lower()
        top = [ex for ex in top if ex["question"].strip().lower() != q][:k]
        return "\n\n".join(few_shot_selector._format_training_example(ex) for ex in top)
    sparql_generator.select_few_shot_examples = patched


def run_mode(items, golds, soft_hint, out_path):
    pipeline.USE_SOFT_HINT = bool(soft_hint)
    preds, refs, rows = [], [], []
    consecutive_errors = 0
    for i, (item, gold) in enumerate(zip(items, golds), start=1):
        t0 = time.time()
        try:
            res = pipeline.answer_question(item["question"], verbose=False, resolve_labels=False)
            values = flatten_results(res["results"])
            method, sparql = res.get("method"), res.get("sparql")
        except Exception as e:
            import traceback
            values, method, sparql = [], "error", f"{type(e).__name__}: {e}"
            print("    ΣΦΑΛΜΑ: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)[-4:]))
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise SystemExit("3 συνεχόμενα σφάλματα, σταματώ. Στείλε μου το μήνυμα ΣΦΑΛΜΑ παραπάνω.")
        else:
            consecutive_errors = 0
        score = compute_prf1(values, gold)
        preds.append(values)
        refs.append(gold)
        rows.append({"question": item["question"], "method": method, "f1": score["f1"],
                     "precision": score["precision"], "recall": score["recall"],
                     "pred": values[:20], "gold": gold[:20], "sparql": sparql})
        print(f"[{i}/{len(items)}] f1={score['f1']:.2f} {method:<20} {time.time() - t0:5.1f}s  {item['question'][:70]}",
              flush=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return evaluate_dataset(preds, refs), rows


def summarize(name, metrics, rows):
    print(f"\n===== {name} =====")
    print(json.dumps(metrics, indent=2))
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r["f1"])
    for m, v in sorted(by.items(), key=lambda x: str(x[0])):
        print(f"  {str(m):<22} {len(v):>4} ερωτήσεις   μέσο F1 = {sum(v) / len(v):.3f}")
    print(f"  τέλεια (F1=1): {sum(r['f1'] == 1 for r in rows)}/{len(rows)}   "
          f"κενή πρόβλεψη: {sum(not r['pred'] for r in rows)}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=config.TRAINING_DATA_PATH)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--soft-hint", choices=["0", "1", "both"], default="both")
    ap.add_argument("--no-loo", action="store_true", help="μην αφαιρείς την ερώτηση από τα few-shot")
    a = ap.parse_args()

    items = load_questions(a.file)
    random.Random(a.seed).shuffle(items)
    if not a.no_loo:
        install_leave_one_out()

    cache = load_cache()
    chosen, golds, skipped = [], [], 0
    for item in items:
        if len(chosen) >= a.limit:
            break
        g = gold_values(item, cache)
        if g is None:
            skipped += 1
            print(f"[skip] άκυρο gold query: {item['question'][:60]}")
            if skipped >= 10 and not chosen:
                raise SystemExit("Τα πρώτα 10 gold queries απέτυχαν όλα. Κάτι είναι λάθος με το endpoint "
                                 "ή τα prefixes, σταματώ.")
            continue
        chosen.append(item)
        golds.append(g)
    print(f"{len(chosen)} ερωτήσεις από {a.file} (seed={a.seed})")

    modes = ["0", "1"] if a.soft_hint == "both" else [a.soft_hint]
    results = {}
    for m in modes:
        name = "με soft hint" if m == "1" else "χωρίς soft hint"
        metrics, rows = run_mode(chosen, golds, m == "1", f"results/eval_soft{m}.jsonl")
        results[m] = (metrics, rows)
        summarize(name, metrics, rows)

    if len(modes) == 2:
        r0, r1 = results["0"][1], results["1"][1]
        better = sum(b["f1"] > a_["f1"] for a_, b in zip(r0, r1))
        worse = sum(b["f1"] < a_["f1"] for a_, b in zip(r0, r1))
        print(f"\nΜε soft hint: καλύτερο σε {better}, χειρότερο σε {worse}, ίδιο σε {len(r0) - better - worse} ερωτήσεις.")
        print("Αναλυτικά ανά ερώτηση: results/eval_soft0.jsonl / results/eval_soft1.jsonl")


if __name__ == "__main__":
    main()