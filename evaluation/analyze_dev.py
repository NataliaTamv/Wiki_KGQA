import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)   
import json, re, sys
from collections import Counter

path = sys.argv[1]
show = int(sys.argv[sys.argv.index("--show") + 1]) if "--show" in sys.argv else 40
rows = [json.loads(l) for l in open(path, encoding="utf-8")]

YESNO = re.compile(r"^(is|are|was|were|does|do|did|has|have|had|can|could|will|would)\b", re.I)
SUPER = re.compile(r"\b(most|least|largest|smallest|biggest|highest|lowest|tallest|longest|shortest|oldest|youngest|first|last|latest|earliest|fastest|heaviest|richest|deepest)\b", re.I)
COUNT = re.compile(r"^how many\b", re.I)

def tags(r):
    q, s = r["question"], r.get("sparql") or ""
    t = []
    if YESNO.match(q): t.append("yesno")
    if SUPER.search(q): t.append("superlative")
    if COUNT.match(q): t.append("count")
    if r["method"] == "error": t.append("exception")
    if s and not re.search(r"wd:Q\d+", s): t.append("no-entity-in-query")
    if re.search(r"\bnot\b|\bexcept\b|\bbut not\b|\bwithout\b", q, re.I): t.append("negation")
    if len(re.findall(r"wdt:P\d+|p:P\d+", s)) >= 3: t.append("multi-hop")
    return t

def reason(r):
    if r["f1"] >= 0.999: return "perfect"
    if not r["pred"]:
        return "empty"
    if r["precision"] == 0 and r["recall"] == 0: return "wrong-answer"
    return "partial"

cnt, bytag = Counter(), {}
for r in rows:
    k = reason(r); cnt[k] += 1
    for t in tags(r):
        bytag.setdefault(t, Counter())[k] += 1
print("-Κατανομή-", dict(cnt))
print("(perfect / empty / wrong-answer / partial)")
for t, c in sorted(bytag.items(), key=lambda x: -sum(x[1].values())):
    print(f"  {t:<20} n={sum(c.values()):<3} perfect={c['perfect']} empty={c['empty']} wrong={c['wrong-answer']} partial={c['partial']}")

print(f"\n Αποτυχίες {show}")
n = 0
for r in rows:
    k = reason(r)
    if k == "perfect": continue
    n += 1
    if n > show: break
    print(f"\n[{k}] f1={r['f1']:.2f} {r['method']} {tags(r)}\nQ: {r['question']}")
    print("pred:", r["pred"][:5], "| gold:", r["gold"][:5])
    print((r.get("sparql") or "").strip()[:600])
