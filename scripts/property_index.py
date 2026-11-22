
import json
import os
import re

import numpy as np

from . import sparql_executor
from .property_embeddings import _encode

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache")
META_PATH = os.path.join(DATA_DIR, "property_index.json")
VEC_PATH = os.path.join(DATA_DIR, "property_index.npy")

_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX schema: <http://schema.org/>
"""

_Q_LABELS = _PREFIXES + """
SELECT ?prop ?label ?type WHERE {
  ?prop wikibase:directClaim ?p .
  ?prop wikibase:propertyType ?type .
  FILTER(?type != wikibase:ExternalId)
  ?prop rdfs:label ?label .
  FILTER(LANG(?label) = "en")
}
"""

_Q_ALIASES = _PREFIXES + """
SELECT ?prop ?alias WHERE {
  ?prop wikibase:directClaim ?p .
  ?prop wikibase:propertyType ?type .
  FILTER(?type != wikibase:ExternalId)
  ?prop skos:altLabel ?alias .
  FILTER(LANG(?alias) = "en")
}
"""

_Q_DESCS = _PREFIXES + """
SELECT ?prop ?desc WHERE {
  ?prop wikibase:directClaim ?p .
  ?prop schema:description ?desc .
  FILTER(LANG(?desc) = "en")
}
"""

_meta = None      # {"props": {pid: {"label","description","type"}}, "texts": [[pid, text], ...]}
_vecs = None      # (N, 384) float32, L2-normalized


def _pid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def _build():
    props = {}
    for r in sparql_executor.run_sparql(_Q_LABELS):
        if r.get("prop") and r.get("label"):
            props[_pid(r["prop"])] = {"label": r["label"], "description": "",
                                      "type": _pid(r.get("type", ""))}
    if not props:
        raise RuntimeError("Το endpoint δεν επέστρεψε properties (έλεγξε το σύνδεσμο/endpoint).")
    try:
        for r in sparql_executor.run_sparql(_Q_DESCS):
            pid = _pid(r.get("prop", ""))
            if pid in props and r.get("desc"):
                props[pid]["description"] = r["desc"]
    except Exception as e:
        print(f"[property_index] περιγραφές δεν φορτώθηκαν: {e}")
    texts = [[pid, p["label"]] for pid, p in props.items()]
    try:
        for r in sparql_executor.run_sparql(_Q_ALIASES):
            pid = _pid(r.get("prop", ""))
            if pid in props and r.get("alias"):
                texts.append([pid, r["alias"]])
    except Exception as e:
        print(f"[property_index] aliases δεν φορτώθηκαν: {e}")
    print(f"[property_index] {len(props)} properties, {len(texts)} labels/aliases. Υπολογισμός embeddings...")
    chunks = []
    for i in range(0, len(texts), 256):
        chunks.append(_encode([t for _, t in texts[i:i + 256]]))
    vecs = np.concatenate(chunks).astype(np.float32)
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"props": props, "texts": texts}, f, ensure_ascii=False)
    np.save(VEC_PATH, vecs)
    return {"props": props, "texts": texts}, vecs


def load():
    global _meta, _vecs
    if _meta is not None:
        return
    if os.path.exists(META_PATH) and os.path.exists(VEC_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            _meta = json.load(f)
        _vecs = np.load(VEC_PATH)
        if len(_vecs) == len(_meta["texts"]):
            return
        print("[property_index] ασυμφωνία cache, ξαναχτίζω...")
    _meta, _vecs = _build()


def _norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def search(mention: str, k: int = 5, extra_variants: list[str] | None = None) -> list[dict]:
    
    load()
    variants = [mention] + list(extra_variants or [])
    qv = _encode(variants)                      # (V, d)
    sims = (_vecs @ qv.T).max(axis=1)           # (N,)
    wanted = {_norm(v) for v in variants}
    best: dict[str, float] = {}
    props0 = _meta["props"]
    for (pid, text), s in zip(_meta["texts"], sims):
        if _norm(text) in wanted:
            
            s = 1.0 if _norm(props0[pid]["label"]) == _norm(text) else 0.97
        else:
            s = float(s)
        if s > best.get(pid, -1.0):
            best[pid] = s
    top = sorted(best.items(), key=lambda x: x[1], reverse=True)[:k]
    props = _meta["props"]
    return [{"id": pid, "label": props[pid]["label"],
             "description": props[pid]["description"], "score": round(sc, 3)}
            for pid, sc in top]


def link_relations(mentions: list[str], k: int = 5, min_score: float = 0.35) -> dict:
    
    try:
        from .subgraph import _variants
    except Exception:
        _variants = lambda m: [m]
    out = {}
    for m in dict.fromkeys(mentions):
        if not m or not m.strip():
            continue
        vs = _variants(m)
        res = [r for r in search(vs[0], k=k, extra_variants=vs[1:]) if r["score"] >= min_score]
        if res:
            out[m] = res
    return out



_WDT_VALUE = re.compile(r"(?<![\w:/|^])wdt:(P\d+)(?=\s+[?\"'\d<-])")


def normalize_quantities(query: str) -> str:
    
    try:
        load()
    except Exception:
        return query
    props = _meta["props"]

    def rep(m):
        pid = m.group(1)
        if props.get(pid, {}).get("type", "").endswith("Quantity"):
            return f"p:{pid}/psn:{pid}/wikibase:quantityAmount"
        return m.group(0)

    return _WDT_VALUE.sub(rep, query)


_LABELS = None


def is_label(text: str) -> bool:
    
    global _LABELS
    try:
        load()
    except Exception:
        return False
    if _LABELS is None:
        _LABELS = {_norm(p["label"]) for p in _meta["props"].values()}
    return _norm(text) in _LABELS
