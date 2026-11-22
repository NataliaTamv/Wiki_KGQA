

import argparse
import json


def pick_question(question_field, language):
   
    if isinstance(question_field, str):          
        return question_field.strip() or None
    for q in question_field or []:
        if q.get("language") == language and q.get("string"):
            return q["string"].strip()
    return None


def normalize_answers(answers):
    
    if answers is None:
        return []
    if not isinstance(answers, list):
        answers = [answers]
    return [str(a) for a in answers if a is not None and str(a).strip() != ""]


def convert(data, language="en", limit=None):
    items = data.get("questions", []) if isinstance(data, dict) else data
    out, skipped = [], 0

    for item in items:
        text = pick_question(item.get("question"), language)
        if not text:
            skipped += 1
            continue
        out.append({
            "question": text,
            "gold_answers": normalize_answers(item.get("answers")),
        })
        if limit and len(out) >= limit:
            break

    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/en_no_mentions_dev.json")
    ap.add_argument("--output", default="data/dev_questions_all.json")
    ap.add_argument("--language", default="en", help="γλώσσα ερώτησης (προεπιλογή: en)")
    ap.add_argument("--limit", type=int, default=None, help="μέγιστος αριθμός ερωτήσεων")
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        data = json.load(f)

    out, skipped = convert(data, args.language, args.limit)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"Γράφτηκαν {len(out)} ερωτήσεις στο {args.output} "
          f"(παραλείφθηκαν {skipped} χωρίς ερώτηση στα '{args.language}').")


if __name__ == "__main__":
    main()