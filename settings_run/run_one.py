import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)   

import sys
from settings_run import pipeline

if len(sys.argv) < 2:
    raise SystemExit('Χρήση: python run_one.py "<ερώτηση>"')

question = " ".join(sys.argv[1:])
res = pipeline.answer_question(question, verbose=True, resolve_labels=False)
print("\n=== Αποτέλεσμα ===")
print("method:", res.get("method"))
print("sparql:\n", res.get("sparql"))
print("results:", res.get("results"))
