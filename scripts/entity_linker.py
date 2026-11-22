
import re

import requests 
from settings_run import config


def _use_local_search() -> bool:
  
    return bool(getattr(config, "EMBEDDINGS_DIR", None))

#πσλιος τροπος
def search_wikidata(mention: str, entity_type: str = "item", limit: int = 10) -> list[dict]: 
    
    if entity_type == "item" and _use_local_search():
        from .local_entity_search import search_local
        candidates = search_local(mention, max(limit, _FETCH_LIMIT))
        candidates = _merge_alias_candidates(mention, candidates)
        return _rerank_by_label_match(mention, _rerank_by_popularity(candidates, limit))

    params = {
        "action": "wbsearchentities", 
        "search": mention,
        "language": "en",
        "type": entity_type,
        "limit": limit,
        "format": "json", 
    }
    headers = {"User-Agent": config.USER_AGENT} 
    resp = requests.get(config.WIKIDATA_ENTITY_SEARCH_API, params=params, headers=headers, timeout=10)
    data = resp.json()

    """
    η δομη που επιστρεφει το request ειναι:
            {
            "search": [
                {
                    "id": "Q937",
                    "label": "Albert Einstein",
                    "description": "...",
                    ...
                },
                ...
            ]
        }
    """
    results= []
    for item in data.get("search",[]):
        results.append({
            "id": item.get("id"),
            "label": item.get("label"),
            "description": item.get("description"),
        })
    return _rerank_by_label_match(mention,results)

_FETCH_LIMIT = 100
_SITELINKS: dict[str, int] = {}
_NOT_AN_ENTITY: dict[str, bool] = {}   # disambiguation / category / list / template
_POPULARITY_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
"""


def _fetch_popularity(qids: list[str]) -> None:
    
    from . import sparql_executor
    todo = [q for q in qids if q not in _SITELINKS]
    if not todo:
        return
    values = " ".join(f"wd:{q}" for q in todo)
    for q in todo:
        _SITELINKS[q] = 0
        _NOT_AN_ENTITY[q] = False
    rows = sparql_executor.run_sparql(
        _POPULARITY_PREFIXES
        + f"SELECT ?e (COUNT(*) AS ?s) WHERE {{ VALUES ?e {{ {values} }} ?e ?p ?o }} GROUP BY ?e")
    for row in rows:
        try:
            _SITELINKS[_qid_of(row.get("e"))] = int(float(row.get("s", 0)))
        except (TypeError, ValueError):
            pass
    rows = sparql_executor.run_sparql(
        _POPULARITY_PREFIXES
        + f"SELECT DISTINCT ?e WHERE {{ VALUES ?e {{ {values} }} ?e wdt:P31 ?t . "
          f"VALUES ?t {{ wd:Q4167410 wd:Q4167836 wd:Q13406463 wd:Q11266439 }} }}")
    for row in rows:
        _NOT_AN_ENTITY[_qid_of(row.get("e"))] = True


def _rerank_by_popularity(candidates: list[dict], limit: int) -> list[dict]:
    
    exact = [c for c in candidates if c.get("match") == "exact"]
    rest = [c for c in candidates if c.get("match") != "exact"]
    if len(exact) > 1:
        try:
            _fetch_popularity([c["id"] for c in exact])
            exact.sort(key=lambda c: (_NOT_AN_ENTITY.get(c["id"], False), -_SITELINKS.get(c["id"], 0)))
        except Exception as e:
            print(f"[entity_linker] αποτυχία popularity rerank: {e}")
            return candidates[:limit]
    return (exact + rest)[:limit]


def _rerank_by_label_match(mention: str, candidates: list[dict]) -> list[dict]:
  
    mention_normalized = mention.strip().lower()
 
    def sort_key(candidate: dict) -> int:
        label_normalized = candidate["label"].strip().lower()
        return 0 if (label_normalized == mention_normalized or candidate.get("alias")) else 1
 
    return sorted(candidates, key=sort_key)
##περιπτωση entity
def link_entities(mentions: list[str]) -> dict[str, list[dict]]:
   results = {}
   for mention in mentions:
          results[mention] = search_wikidata(mention, "item")
   return results

_DESCRIPTION_PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
"""


def _qid_of(uri: str) -> str:
    return (uri or "").rsplit("/", 1)[-1]


def add_descriptions(candidates: list[dict]) -> list[dict]:
    
    from . import sparql_executor

    missing = [c for c in candidates
               if not c.get("description") and re.fullmatch(r"Q\d+", c.get("id") or "")]
    if not missing:
        return candidates

    values = " ".join(f"wd:{c['id']}" for c in missing)
    descriptions: dict[str, str] = {}

    try:
        rows = sparql_executor.run_sparql(
            _DESCRIPTION_PREFIXES
            + f'SELECT ?e ?d WHERE {{ VALUES ?e {{ {values} }} '
              f'?e schema:description ?d . FILTER(LANG(?d) = "en") }}'
        )
        for row in rows:
            qid = _qid_of(row.get("e"))
            if row.get("d") and qid not in descriptions:
                descriptions[qid] = row["d"]
    except Exception as e:
        print(f"[entity_linker] αποτυχία αναζήτησης descriptions: {e}")

   
    still = [c for c in missing if c["id"] not in descriptions]
    if still:
        values = " ".join(f"wd:{c['id']}" for c in still)
        try:
            rows = sparql_executor.run_sparql(
                _DESCRIPTION_PREFIXES
                + f'SELECT ?e ?tl WHERE {{ VALUES ?e {{ {values} }} '
                  f'?e wdt:P31 ?t . ?t rdfs:label ?tl . FILTER(LANG(?tl) = "en") }}'
            )
            types: dict[str, list[str]] = {}
            for row in rows:
                qid = _qid_of(row.get("e"))
                label = row.get("tl")
                if label and label not in types.setdefault(qid, []):
                    types[qid].append(label)
            for qid, labels in types.items():
                descriptions[qid] = "instance of: " + ", ".join(labels[:3])
        except Exception as e:
            print(f"[entity_linker] αποτυχία αναζήτησης τύπων: {e}")

    for c in missing:
        if c["id"] in descriptions:
            c["description"] = descriptions[c["id"]]
    return candidates



def link_relations(mentions: list[str]) -> dict[str, list[dict]]:
    results = {}
    for mention in mentions:
             results[mention] = search_wikidata(mention, "property")
    return results

#παλιος τροπος 
def get_labels_for_uris(uris: list[str], language: str = "en") -> dict[str, str]:
    
    qid_by_uri = {}
    for uri in uris:
        if "/entity/Q" in uri:
            qid = uri.rsplit("/", 1)[-1]
            qid_by_uri[uri] = qid
 
    if not qid_by_uri:
        return {}
 
    labels = {}
    qids = list(set(qid_by_uri.values()))
 
   
    batch_size = 50
    for i in range(0, len(qids), batch_size):
        batch = qids[i:i + batch_size]
        params = {
            "action": "wbgetentities",
            "ids": "|".join(batch),
            "props": "labels",
            "languages": language,
            "format": "json",
        }
        headers = {"User-Agent": config.USER_AGENT}
        resp = requests.get(config.WIKIDATA_ENTITY_SEARCH_API, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
 
        for qid, entity_data in data.get("entities", {}).items():
            label_info = entity_data.get("labels", {}).get(language)
            if label_info:
                labels[qid] = label_info["value"]
 
   
    return {uri: labels.get(qid, qid) for uri, qid in qid_by_uri.items()}


_ALIAS_CACHE: dict[str, list[dict]] = {}
_ALIAS_LIMIT = 100


def _alias_candidates(mention: str) -> list[dict]:
   
    m = (mention or "").strip()
    if not m or len(m) > 80 or "\n" in m:
        return []
    if m in _ALIAS_CACHE:
        return _ALIAS_CACHE[m]
    from . import sparql_executor
    lit = m.replace("\\", "\\\\").replace('"', '\\"')
    out: list[dict] = []
    try:
        rows = sparql_executor.run_sparql(
            "PREFIX skos: <http://www.w3.org/2004/02/skos/core#>\n"
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n"
            f'SELECT DISTINCT ?e ?l WHERE {{ ?e skos:altLabel "{lit}"@en . '
            f'OPTIONAL {{ ?e rdfs:label ?l . FILTER(LANG(?l) = "en") }} }} LIMIT {_ALIAS_LIMIT}')
        seen = set()
        for row in rows:
            qid = _qid_of(row.get("e"))
            if not re.fullmatch(r"Q\d+", qid) or qid in seen:
                continue
            seen.add(qid)
            out.append({"id": qid, "label": row.get("l") or qid, "description": None,
                        "score": 1.0, "match": "exact", "alias": True})
    except Exception as e:
        print(f"[entity_linker] αποτυχία αναζήτησης aliases: {e}")
        return []
    _ALIAS_CACHE[m] = out
    return out


def _merge_alias_candidates(mention: str, candidates: list[dict]) -> list[dict]:
    
    if not getattr(config, "ALIAS_LOOKUP", True):
        return candidates
    has_label_match = any(c.get("match") == "exact" for c in candidates)
    if has_label_match and not _is_acronym(mention):
        return candidates
    have = {c["id"] for c in candidates}
    extra = [c for c in _alias_candidates(mention) if c["id"] not in have]
    if not extra:
        return candidates
    exact = [c for c in candidates if c.get("match") == "exact"]
    rest = [c for c in candidates if c.get("match") != "exact"]
    return exact + extra + rest


def _is_acronym(mention: str) -> bool:
    
    m = (mention or "").strip()
    return bool(re.fullmatch(r"(?:[A-Z0-9][.&-]?){2,8}", m)) and sum(ch.isupper() for ch in m) >= 2
