

import re
from difflib import SequenceMatcher

from . import sparql_executor

PREFIXES = """PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""


def fetch_subgraph(qids) -> list[dict]:
    
    qids = [qids] if isinstance(qids, str) else list(qids)
    values = " ".join(f"wd:{q}" for q in qids)
    query = PREFIXES + f"""
SELECT DISTINCT ?prop ?propLabel ?dir WHERE {{
  {{ VALUES ?n {{ {values} }} ?n ?p ?o . BIND("out" AS ?dir) }}
  UNION
  {{ VALUES ?n {{ {values} }} ?s ?p ?n . BIND("in" AS ?dir) }}
  ?prop wikibase:directClaim ?p .
  ?prop wikibase:propertyType ?type .
  FILTER(?type != wikibase:ExternalId)
  ?prop rdfs:label ?propLabel .
  FILTER(LANG(?propLabel) = "en")
}}
"""
    rows = sparql_executor.run_sparql(query)
    props = []
    for row in rows:
        if not row.get("prop") or not row.get("propLabel"):
            continue
        props.append({
            "property_id": row["prop"].rsplit("/", 1)[-1],
            "property_label": row["propLabel"],
            "direction": row.get("dir", "out"),
        })
    return props


def fetch_values(qids: list[str], property_id: str, direction: str, limit: int = 200) -> list[str]:
    values = " ".join(f"wd:{q}" for q in qids)
    if direction == "out":
        pattern = f"?n wdt:{property_id} ?x ."
    else:
        pattern = f"?x wdt:{property_id} ?n ."
    query = PREFIXES + (f"SELECT DISTINCT ?x WHERE {{ VALUES ?n {{ {values} }} "
                        f"{pattern} }} LIMIT {limit}")
    found = []
    for row in sparql_executor.run_sparql(query):
        m = re.search(r"/entity/(Q\d+)$", row.get("x", ""))
        if m:
            found.append(m.group(1))
    return found



def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


_STOPWORDS = {"of", "the", "a", "an", "in", "on", "at", "to", "by", "for",
              "is", "was", "has", "with"}


_RELATION_SYNONYMS = {
    "celebrated on": ["day in year for periodic occurrence"],
    "celebrated": ["day in year for periodic occurrence"],
    "observed on": ["day in year for periodic occurrence"],
    "held on": ["day in year for periodic occurrence"],
    "date": ["day in year for periodic occurrence"],
    "born in": ["place of birth"],
    "birthplace": ["place of birth"],
    "born on": ["date of birth"],
    "birthday": ["date of birth"],
    "died in": ["place of death"],
    "died on": ["date of death"],
    "wife": ["spouse"],
    "husband": ["spouse"],
    "married to": ["spouse"],
    "directed by": ["director"],
    "written by": ["author", "screenwriter"],
    "wrote": ["author"],
    "capital city": ["capital"],
    "works for": ["employer"],
    "worked for": ["employer"],
    "founded on": ["inception"],
    "date founded": ["inception"],
    "release date": ["publication date"],
    "released on": ["publication date"],
    "language spoken": ["official language", "languages spoken, written or signed"],
    "children": ["child"],
    "parents": ["father", "mother"],
    "studied at": ["educated at"],
    "attended": ["educated at"],
    "played by": ["cast member"],
    "played": ["cast member"],
    "starring": ["cast member"],
    "stars": ["cast member"],
    "acted in": ["cast member"],
    "follow": ["follows"],
    "follows": ["follows"],
    "preceded by": ["follows"],
    "followed by": ["followed by"],
    "precede": ["followed by"],
    "succeed": ["follows"],
    "invented": ["time of discovery or invention", "inception"],
    "invent": ["time of discovery or invention", "inception"],
    "discovered": ["time of discovery or invention"],
    "discover": ["time of discovery or invention"],
    "discovered by": ["discoverer or inventor"],
    "invented by": ["discoverer or inventor"],
    "released": ["publication date"],
    "release": ["publication date"],
    "released in": ["publication date"],
    "started": ["start time", "inception"],
    "start": ["start time", "inception"],
    "start date": ["start time", "inception"],
    "began": ["start time", "inception"],
    "opened": ["inception"],
    "open": ["inception"],
    "established": ["inception"],
    "founded": ["inception"],
    "created": ["inception"],
    "ended": ["end time", "dissolved, abolished or demolished date"],
    "end": ["end time", "dissolved, abolished or demolished date"],
    "made by": ["manufacturer"],
    "manufactured by": ["manufacturer"],
    "published by": ["publisher"],
    "composed by": ["composer"],
    "played with": ["member of"],
    "headquartered in": ["headquarters location"],
    "die": ["date of death"],
    "died": ["date of death"],
    "dead": ["date of death"],
    "death": ["date of death"],
}


def _variants(mention: str) -> list[str]:
    
    return [mention] + _RELATION_SYNONYMS.get(mention.strip().lower(), [])


def _content(text: str) -> str:
   
    words = [w for w in re.findall(r"\w+", text.lower()) if w not in _STOPWORDS]
    return " ".join(words) or text.strip().lower()


def label_similarity(mention: str, label: str) -> float:
   
    m, l = _content(mention), _content(label)
    char_sim = SequenceMatcher(None, m, l).ratio()
    tm, tl = _tokens(m), _tokens(l)
    jaccard = len(tm & tl) / len(tm | tl) if tm | tl else 0.0
    # Όλες οι λέξεις του mention υπάρχουν στο label (π.χ. "located in" ->
    # "located in the administrative territorial entity"): καλύτερο από
    # τυχαία ομοιότητα χαρακτήρων, αλλά χαμηλότερο από ακριβές ταίριασμα.
    containment = 0.85 if tm and tm <= tl else 0.0
    score = max(char_sim, jaccard, containment)
    if mention.strip().lower() == label.strip().lower():
        score += 0.01
    return score


def _best_similarity(mention: str, label: str) -> float:
   
    return max(label_similarity(v, label) for v in _variants(mention))


def rank_properties(mention: str, props: list[dict], top_k: int = 3) -> list[dict]:
    
    variants = _variants(mention)
    labels = [p["property_label"] for p in props]
    lex = [max(label_similarity(v, l) for v in variants) for l in labels]
    try:
        from .property_embeddings import cosine_scores
        cos = [max(vals) for vals in zip(*[cosine_scores(v, labels) for v in variants])] if labels else []
    except Exception as e:          # π.χ. δεν υπάρχει torch/μοντέλο: πέφτουμε στο λεκτικό
        print(f"[subgraph] embeddings μη διαθέσιμα ({e}), χρήση λεκτικού score")
        cos = [0.0] * len(props)
    scored = [{**p, "lex": l, "cos": float(c), "score": max(l, float(c))}
              for p, l, c in zip(props, lex, cos)]
    return sorted(scored, key=lambda x: x["score"], reverse=True)[:top_k]


def build_answer_query(qid: str, property_id: str, direction: str) -> str:
   
    if direction == "out":
        pattern = f"wd:{qid} wdt:{property_id} ?x ."
    else:
        pattern = f"?x wdt:{property_id} wd:{qid} ."
    return PREFIXES + f"SELECT ?x WHERE {{ {pattern} }} LIMIT 50"


def find_best_property(entity_links: dict, relation_mentions: list[str],
                       min_score: float = 0.6, max_candidates: int = 3) -> dict | None:
    
    best = None
    for mention, candidates in entity_links.items():
        for cand in candidates[:max_candidates]:
            qid = cand["id"]
            try:
                props = fetch_subgraph(qid)
            except Exception as e:
                print(f"[subgraph] αποτυχία για {qid}: {e}")
                continue
            if not props:
                print(f"[subgraph] κενό subgraph για {qid}")
                continue
            for rel in relation_mentions:
                ranked = rank_properties(rel, props, top_k=1)
                if ranked and (best is None or ranked[0]["score"] > best["score"]):
                    r = ranked[0]
                    best = {"qid": qid, "entity_label": cand.get("label"),
                            "property_id": r["property_id"],
                            "property_label": r["property_label"],
                            "direction": r["direction"], "score": r["score"]}
    if best and best["score"] >= min_score:
        return best
    return None

---------------------------------------------------------------

def _walk_chain(qid: str, chain: list[str], min_score: float, max_nodes: int) -> dict | None:
    nodes = [qid]
    hops = []
    for i, rel in enumerate(chain):
        props = fetch_subgraph(nodes[:max_nodes])
        ranked = rank_properties(rel, props, top_k=1)
        if not ranked or ranked[0]["score"] < min_score:
            return None
        hop = ranked[0]
        hops.append(hop)
        if i < len(chain) - 1:
            nodes = fetch_values(nodes[:max_nodes], hop["property_id"], hop["direction"])
            if not nodes:
                return None
    return {"qid": qid, "hops": hops, "score": min(h["score"] for h in hops)}


def find_best_path(entity_links: dict, relation_chain: list[str], min_score: float = 0.6,max_candidates: int = 3, max_nodes: int = 5) -> dict | None:
   
    best = None
    for mention, candidates in entity_links.items():
        for cand in candidates[:max_candidates]:
            try:
                path = _walk_chain(cand["id"], relation_chain, min_score, max_nodes)
            except Exception as e:
                print(f"[subgraph] αποτυχία path για {cand['id']}: {e}")
                continue
            if path and (best is None or path["score"] > best["score"]):
                path["entity_label"] = cand.get("label")
                best = path
    return best


def build_path_query(qid: str, hops: list[dict]) -> str:
       patterns = []
    prev = f"wd:{qid}"
    for i, hop in enumerate(hops):
        nxt = "?x" if i == len(hops) - 1 else f"?n{i + 1}"
        if hop["direction"] == "out":
            patterns.append(f"{prev} wdt:{hop['property_id']} {nxt} .")
        else:
            patterns.append(f"{nxt} wdt:{hop['property_id']} {prev} .")
        prev = nxt
    return PREFIXES + "SELECT DISTINCT ?x WHERE { " + " ".join(patterns) + " } LIMIT 50"