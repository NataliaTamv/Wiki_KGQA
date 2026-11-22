
import argparse
import json
import os
import re
import sys
import time

import requests

DEFAULT_DATASET = ("C:\\Users\\pcsc\\Documents\\ReposΜεταπτυχιακο\\kgqa_baseline\\data\\train_data_no_mentions.json")
DEFAULT_ENDPOINT = "http://155.207.113.40:7001"

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX psv: <http://www.wikidata.org/prop/statement/value/>
PREFIX psn: <http://www.wikidata.org/prop/statement/value-normalized/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

ENTITY_RE = re.compile(r"http://www\.wikidata\.org/entity/(Q\d+)$")
NUM_RE = re.compile(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?$")


def canon_str(s):
    
    s = str(s).strip()
    m = ENTITY_RE.match(s)
    if m:
        return m.group(1)
    if NUM_RE.match(s):
        return "num:%.9g" % float(s)
    return s


def canon_gold(a):
    if isinstance(a, bool):
        return "bool:%s" % str(a).lower()
    if isinstance(a, (int, float)):
        return "num:%.9g" % float(a)
    return canon_str(a)


def canon_live(res):
    if "boolean" in res:
        return {"bool:%s" % str(res["boolean"]).lower()}
    vals = set()
    for row in res["results"]["bindings"]:
        for b in row.values():
            if b.get("type") == "bnode":
                continue
            vals.add(canon_str(b["value"]))
    return vals


def run_query(endpoint, query, timeout):
    r = requests.post(
        endpoint,
        data={"query": PREFIXES + query},
        headers={"Accept": "application/sparql-results+json",
                 "User-Agent": "wikikgqa-diff/0.1 (answer drift check)"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def load_dataset(src):
    if os.path.exists(src):
        with open(src, encoding="utf-8") as f:
            return json.load(f)["questions"]
    r = requests.get(src, timeout=120)
    r.raise_for_status()
    return r.json()["questions"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=DEFAULT_DATASET, help="URL ή τοπικό αρχείο")
    ap.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = όλες")
    ap.add_argument("--ids", default="", help="λίστα id χωρισμένη με κόμμα")
    ap.add_argument("--sleep", type=float, default=1.0, help="δευτερόλεπτα ανάμεσα στα ερωτήματα")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    questions = load_dataset(args.dataset)
    if args.ids:
        wanted = {int(x) for x in args.ids.split(",")}
        questions = [q for q in questions if q["id"] in wanted]
    if args.limit:
        questions = questions[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["id"])
                except Exception:
                    pass

    todo = [q for q in questions if q["id"] not in done]
    print("Ερωτήσεις: %d (υπολείπονται %d)  endpoint: %s" % (len(questions), len(todo), args.endpoint))

    with open(args.out, "a", encoding="utf-8") as out:
        for i, q in enumerate(todo, 1):
            gold = {canon_gold(a) for a in q["answers"]}
            rec = {
                "id": q["id"],
                "question": q["question"][0]["string"],
                "time_dependent": "NOW(" in q["sparql"].upper(),
            }
            try:
                live = canon_live(run_query(args.endpoint, q["sparql"], args.timeout))
                rec["status"] = "same" if live == gold else "changed"
                rec["removed"] = sorted(gold - live)[:30]   
                rec["added"] = sorted(live - gold)[:30]     
                rec["n_gold"], rec["n_live"] = len(gold), len(live)
            except Exception as e:
                rec["status"] = "error"
                rec["error"] = str(e)[:300]
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            print("[%d/%d] id=%s %s" % (i, len(todo), q["id"], rec["status"]))
            time.sleep(args.sleep)

    counts = {}
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            s = json.loads(line)["status"]
            counts[s] = counts.get(s, 0) + 1
    print("\nΠερίληψη:", counts)


if __name__ == "__main__":
    sys.exit(main())