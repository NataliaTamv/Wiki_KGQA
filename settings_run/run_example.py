
import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)   

import argparse
import json
import time
import traceback
from collections import defaultdict

from settings_run import config, pipeline
from settings_run.pipeline import answer_question
from evaluation.evaluate import compute_prf1, flatten_results


def load_done(path):
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    return rows


def summarize(rows):
    n = len(rows) or 1
    print("\n" + "=" * 60)
    print(f"Συνολικά metrics (endpoint: {config.ENDPOINT_NAME}):")
    print(json.dumps({"precision": sum(r["precision"] for r in rows) / n,
                      "recall": sum(r["recall"] for r in rows) / n,
                      "f1": sum(r["f1"] for r in rows) / n,
                      "num_questions": len(rows)}, indent=2))
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r["f1"])
    for m, v in sorted(by.items(), key=lambda x: str(x[0])):
        print(f"  {str(m):<22} {len(v):>4} ερωτήσεις   μέσο F1 = {sum(v) / len(v):.3f}")
    print(f"  τέλεια (F1=1): {sum(r['f1'] == 1 for r in rows)}/{len(rows)}   "
          f"κενή πρόβλεψη: {sum(not r['pred'] for r in rows)}/{len(rows)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/dev_6.json",
                    help="αρχείο ερωτήσεων (λίστα με question και gold_answers)")
    ap.add_argument("--endpoint", default=None,
                    help=f"{' / '.join(config.ENDPOINTS)} ή πλήρες URL (default: {config.ENDPOINT_NAME})")
    ap.add_argument("--limit", type=int, default=0, help="μόνο οι πρώτες N ερωτήσεις (0 = όλες)")
    ap.add_argument("--tag", default="dev_6", help="όνομα αρχείου αποτελεσμάτων: results/run_<tag>.jsonl "
                                              "(default: το όνομα του αρχείου ερωτήσεων)")
    ap.add_argument("--resume", action="store_true", help="συνέχεια από το υπάρχον αρχείο αποτελεσμάτων")
    ap.add_argument("--quiet", action="store_true", help="μία γραμμή ανά ερώτηση αντί για αναλυτική έξοδο")
    ap.add_argument("--soft-hint", choices=["0", "1"], default=None,
                    help="0 = χωρίς soft hint, 1 = με (default: του pipeline)")
    a = ap.parse_args()

    if a.endpoint:
        config.set_endpoint(a.endpoint)
    if a.soft_hint is not None:
        pipeline.USE_SOFT_HINT = a.soft_hint == "1"
    print(f"[endpoint] {config.ENDPOINT_NAME} -> {config.WIKIKGQA_ENDPOINT}")

    with open(a.file, encoding="utf-8") as f:
        dataset = json.load(f)
    if a.limit:
        dataset = dataset[:a.limit]

    tag = a.tag or pathlib.Path(a.file).stem
    out_path = f"results/run_{tag}.jsonl"
    os.makedirs("results", exist_ok=True)

    rows = load_done(out_path) if a.resume else []
    done = {r["question"] for r in rows}
    if a.resume:
        print(f"[resume] {len(rows)} ερωτήσεις υπάρχουν ήδη στο {out_path}")
    print(f"{len(dataset)} ερωτήσεις από {a.file} -> {out_path}")

    t_start, n_new, errors = time.time(), 0, 0
    with open(out_path, "a" if a.resume else "w", encoding="utf-8") as out:
        for item in dataset:
            if item["question"] in done:
                continue
            gold_answers = item["gold_answers"]
            if isinstance(gold_answers, str):     
                gold_answers = [gold_answers]

            t0 = time.time()
            if not a.quiet:
                print("\n" + "=" * 60)
                print(f"Ερώτηση: {item['question']}")
            try:
                result = answer_question(item["question"], few_shot=True, verbose=not a.quiet)
                
                predicted = flatten_results(result["results"])
                method, sparql = result.get("method"), result.get("sparql")
                diag = pipeline.diagnostics(result)
                errors = 0
            except Exception as e:
                predicted, method, sparql = [], "error", f"{type(e).__name__}: {e}"
                diag = {}
                print("    ΣΦΑΛΜΑ: " + "".join(traceback.format_exception(type(e), e, e.__traceback__)[-4:]))
                errors += 1
                if errors >= 3:
                    summarize(rows)
                    raise SystemExit("3 συνεχόμενα σφάλματα, σταματώ (τα ως τώρα αποτελέσματα έχουν αποθηκευτεί).")

            score = compute_prf1(predicted, gold_answers)
            row = {"question": item["question"], "method": method, "f1": score["f1"],
                   "precision": score["precision"], "recall": score["recall"],
                   "pred": predicted[:20], "gold": gold_answers[:20], "sparql": sparql, "diag": diag}
            rows.append(row)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            n_new += 1

            if not a.quiet:
                print("PREDICTED:", predicted)
                print("GOLD:", gold_answers)
                if item.get("gold_sparql"):
                    print("GOLD SPARQL:\n" + item["gold_sparql"])
            eta = (time.time() - t_start) / n_new * (len(dataset) - len(rows)) / 60
            print(f"[{len(rows)}/{len(dataset)}] f1={score['f1']:.2f} {str(method):<20} "
                  f"{time.time() - t0:5.1f}s (~{eta:.0f} λ ακόμα)  {item['question'][:60]}", flush=True)

    summarize(rows)
   
    try:
        from tools.run_log import log_run
        _notes = ('resume ' if a.resume else '') + (f'limit={a.limit} ' if a.limit else '') \
                 + (f'soft_hint={a.soft_hint}' if a.soft_hint is not None else '')
        log_run(rows, a.file, tag=tag, t0=t_start, notes=_notes.strip())
    except Exception as _e:
        print('[run_log] απέτυχε:', _e)
    


if __name__ == "__main__":
    main()