
from . import subgraph

_cache: dict[str, list[dict]] = {}


def _props(qid: str) -> list[dict]:
    if qid not in _cache:
        try:
            _cache[qid] = subgraph.fetch_subgraph(qid)
        except Exception as e:
            print(f"[property_candidates] αποτυχία για {qid}: {e}")
            _cache[qid] = []
    return _cache[qid]


def candidate_properties(entity_links: dict, relations: list[str], max_entities: int = 1, k: int = 5) -> list[dict]:
    "
    out, seen = [], set()
    for mention, cands in entity_links.items():
        for cand in cands[:max_entities]:
            props = _props(cand["id"])
            if not props:
                continue
            for rel in relations:
                for r in subgraph.rank_properties(rel, props, top_k=k):
                    key = (cand["id"], r["property_id"], r["direction"])
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append({"qid": cand["id"], "entity_label": cand.get("label"), "relation": rel,
                                "property_id": r["property_id"], "property_label": r["property_label"],
                                "direction": r["direction"], "score": r["score"]})
    return out
