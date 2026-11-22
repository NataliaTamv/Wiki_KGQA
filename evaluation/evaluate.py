"""
Απλή αξιολόγηση precision / recall / F1 σε επίπεδο "answer set".

Λογική: κάθε ερώτηση έχει ένα gold answer set (π.χ. λίστα από QIDs ή strings).
Συγκρίνουμε με το answer set που παρήγαγε το pipeline.

Η σύγκριση γίνεται πάνω στις ΑΚΑΤΕΡΓΑΣΤΕΣ τιμές που επιστρέφει το SPARQL query
(entity URIs, literals, ημερομηνίες, αριθμοί), ΟΧΙ στα labels. Η κανονικοποίηση
φέρνει και τις δύο πλευρές στην ίδια μορφή:

  http://www.wikidata.org/entity/Q90      -> q90
  Q90                                     -> q90
  "2026-01-19T00:00:00Z"^^xsd:dateTime    -> 2026-01-19
  2026-01-19T00:00:00Z                    -> 2026-01-19
  +42 / 42.0 / "42"^^xsd:integer          -> 42
  "Paris"@en                              -> paris

Στη διπλωματική σου θα χρειαστεί να προσαρμόσεις τη λογική σύγκρισης ανάλογα
με το format του WikiKGQA Challenge (π.χ. αν θέλει exact match σε QIDs,
ή σε labels, ή αν επιτρέπει partial credit).
"""

import re
from decimal import Decimal, InvalidOperation
from typing import Iterable

_ENTITY_URI = re.compile(r"^https?://www\.wikidata\.org/(?:entity|wiki)/([QPL]\d+)$", re.IGNORECASE)
_DATATYPE_SUFFIX = re.compile(r"\^\^(<[^>]*>|[\w-]+:[\w-]+)\s*$")
_LANG_SUFFIX = re.compile(r"@[A-Za-z]{2,3}(?:-[A-Za-z0-9]+)*\s*$")
_MIDNIGHT_DATE = re.compile(r"^([+-]?\d{4,}-\d{2}-\d{2})T00:00:00Z$")
_NUMBER = re.compile(r"^[+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


def normalize(value: str) -> str:
    """Φέρνει μια τιμή (URI, literal, ημερομηνία, αριθμός ή κείμενο) σε κανονική μορφή."""
    v = str(value).strip()

    # 1. Wikidata entity URI -> QID
    m = _ENTITY_URI.match(v)
    if m:
        return m.group(1).lower()

    # 2. Αφαίρεση datatype / language suffix και εισαγωγικών: "..."^^xsd:x , "..."@en
    v = _DATATYPE_SUFFIX.sub("", v).strip()
    v = _LANG_SUFFIX.sub("", v).strip()
    if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
        v = v[1:-1].strip()

    # 3. Ημερομηνία με μηδενική ώρα -> μόνο η ημερομηνία
    m = _MIDNIGHT_DATE.match(v)
    if m:
        return m.group(1).lstrip("+")

    # 4. Αριθμοί: +42, 42.0, 4.2e1 -> 42
    if _NUMBER.match(v):
        try:
            return format(Decimal(v).normalize(), "f")
        except InvalidOperation:
            pass

    return v.lower()


def flatten_results(rows: list[dict]) -> list[str]:
    """
    Από τα raw `results` του pipeline (λίστα από dicts {μεταβλητή: τιμή}) βγάζει
    μια επίπεδη λίστα τιμών, χωρίς label resolution.
    """
    return [value for row in rows for value in row.values() if value not in (None, "")]


def compute_prf1(predicted: Iterable[str], gold: Iterable[str]) -> dict:

    pred_set = set()
    gold_set = set()

    for p in predicted:
        pred_set.add(normalize(p))

    for g in gold:
        gold_set.add(normalize(g))

    if not pred_set and not gold_set:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_set or not gold_set:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(pred_set & gold_set)
    precision = true_positives / len(pred_set)
    recall = true_positives / len(gold_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def evaluate_dataset(predictions: list[list[str]], golds: list[list[str]]) -> dict:
    
    assert len(predictions) == len(golds), "predictions και golds πρέπει να έχουν το ίδιο μήκος"

    scores = [compute_prf1(p, g) for p, g in zip(predictions, golds)]
    n = len(scores) or 1
    return {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "f1": sum(s["f1"] for s in scores) / n,
        "num_questions": len(scores),
    }