
import pathlib, time
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "results" / "run_log.md"

HEADERS = ["id", "ημ/νία", "tag", "endpoint", "αρχείο", "LLM", "k", "N",
           "F1", "P", "R", "ok/μερ/λάθ/κεν", "λ"]


def _read_rows(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("|") and not line.startswith("|-") and not line.startswith("| id "):
                rows.append([c.strip() for c in line.strip().strip("|").split(" | ")])
    return rows


def _render(rows):
    rows = [(r + [""] * len(HEADERS))[:len(HEADERS)] for r in rows]
    w = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(HEADERS)]
    fmt = lambda r: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(r)) + " |"
    return "\n".join([fmt(HEADERS), "|" + "|".join("-" * (x + 2) for x in w) + "|"]
                     + [fmt(r) for r in rows]) + "\n"


def log_run(rows, input_file, tag=None, t0=None, notes=""):
  
    from settings_run import config
    n = len(rows) or 1
    f1s = [r["f1"] for r in rows]
    perfect = sum(1 for x in f1s if x >= 0.999999)
    empty = sum(1 for r in rows if not r.get("pred"))
    partial = sum(1 for r in rows if r.get("pred") and 0 < r["f1"] < 0.999999)
    wrong = sum(1 for r in rows if r.get("pred") and r["f1"] == 0)
    old = _read_rows(LOG_PATH)
    try:
        run_id = max(int(r[0]) for r in old) + 1 if old else 1
    except ValueError:
        run_id = len(old) + 1
    clean = lambda x: str(x).replace("|", "/").replace("\n", " ")
    new = [str(run_id), datetime.now().strftime("%d/%m %H:%M"), tag or "",
           str(getattr(config, "ENDPOINT_NAME", "")), pathlib.Path(str(input_file)).name,
           str(getattr(config, "LLM_MODEL", "")).replace("-instruct", ""), str(getattr(config, "FEW_SHOT_K", "")), str(len(rows)),
           f"{sum(f1s) / n:.2f}", f"{sum(r['precision'] for r in rows) / n:.2f}",
           f"{sum(r['recall'] for r in rows) / n:.2f}",
           f"{perfect}/{partial}/{wrong}/{empty}",
           f"{(time.time() - t0) / 60:.0f}" if t0 else ""]
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(_render(old + [[clean(c) for c in new]]), encoding="utf-8")
    print(f"[run_log] run #{run_id} -> {LOG_PATH}  (F1={new[8]}, P={new[9]}, R={new[10]})")
    return run_id
