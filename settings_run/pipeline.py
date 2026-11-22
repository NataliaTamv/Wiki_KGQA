
import json
import os
import re
from settings_run import config
from scripts.property_candidates import candidate_properties
from scripts import property_index

from scripts import (
    choice_llm_entities,
    entity_linker,
    sparql_executor,
    sparql_generator,
    subgraph,
)
from scripts.few_shot_selector import load_training_file
from scripts.llm_client import call_llm
from scripts.mention_cleaning import clean_mentions

_TRAINING_EXAMPLES = load_training_file(config.TRAINING_DATA_PATH, language="en")


HINT_MIN_SCORE = 0.85

MAX_SPARQL_REPAIRS = 2

USE_PROPERTY_INDEX = os.getenv("USE_PROPERTY_INDEX", "1") == "1"
USE_SOFT_HINT = os.getenv("USE_SOFT_HINT", "1") == "1"



MENTION_EXTRACTION_SYSTEM_PROMPT = """You extract entity mentions and relation
mentions from a natural language question, for later linking to Wikidata.

CRITICAL RULE: only extract text that appears LITERALLY in the question itself.
NEVER extract the answer to the question, even if you happen to know it.
Entities are the "given" things the question is about; relations describe what
is being asked. If you are not 100% sure something is written in the question,
do not include it.
 
Respond ONLY with a JSON object like:
{"entities": ["..."], "relations": ["..."]}
No explanation, no markdown, no extra text.

- "relations" must be listed IN THE ORDER they are followed, starting from the
  entity and moving towards the answer. A question with one relation has one
  item; "the director of X's spouse" has two.
- Generic nouns describing the TYPE of the answer ("country", "city") are
  relations, not entities.
"""

def _parse_mentions_json(raw: str) -> dict:

    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    start = raw.find("{")
    if start != -1:
        try:
            obj, _ = json.JSONDecoder().raw_decode(raw, start)
            return obj
        except json.JSONDecodeError:
            pass

    print(f"[pipeline] ΠΡΟΣΟΧΗ: απέτυχε το parsing των mentions. Raw response του LLM:\n{raw!r}")
    return {"entities": [], "relations": []}


_FUNCTION_WORDS = {"of", "the", "a", "an", "in", "on", "at", "to", "for", "by", "with", "is", "are", "was",
                   "were", "from", "and", "or", "as"}


def _singular_phrase(text: str) -> str:
    words = text.lower().split()
    if not words:
        return text.lower()
    w = words[-1]
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        w = w[:-1]
    return " ".join(words[:-1] + [w])


def _span_in_question(question: str, mention: str):
    
    m = re.search(re.escape(mention.strip()), question, flags=re.IGNORECASE)
    return (question[m.start():m.end()], m.start()) if m else (None, -1)


def _refine_mentions(question: str, mentions: dict, is_label=None) -> dict:
   
    entities = list(dict.fromkeys(mentions.get("entities", [])))
    relations = list(dict.fromkeys(mentions.get("relations", [])))

    relations = [r for r in relations
                 if any(w not in _FUNCTION_WORDS for w in re.findall(r"\w+", r.lower()))]

    new_entities, moved_to_rel = [], []
    for e in entities:
        span, _ = _span_in_question(question, e)
        if (is_label and span and span == span.lower()
                and (is_label(span.lower()) or is_label(_singular_phrase(span)))):
            moved_to_rel.append(e)
        else:
            new_entities.append(e)

    keep_rel = []
    for r in relations:
        span, pos = _span_in_question(question, r)
        tokens = re.findall(r"[A-Za-z0-9][\w'.-]*", span) if span else []
        content = [t for t in tokens if t.lower() not in _FUNCTION_WORDS]
        if (span and pos > 0 and content and all(t[0].isupper() or t[0].isdigit() for t in content)
                and not (is_label and is_label(span.lower()))):
            new_entities.append(span)
        else:
            keep_rel.append(r)

    
    final_rel = keep_rel if keep_rel else moved_to_rel
    return {"entities": list(dict.fromkeys(new_entities)),
            "relations": list(dict.fromkeys(final_rel))}


def extract_mentions(question: str) -> dict:
    
    raw = call_llm(question,system=MENTION_EXTRACTION_SYSTEM_PROMPT,max_tokens=200,)

    data = clean_mentions(_parse_mentions_json(raw))
    original = {"entities": data.get("entities", []), "relations": data.get("relations", [])}
    refined = _refine_mentions(question, original, is_label=property_index.is_label)
    if refined != original:
        refined["refined_from"] = original
    return refined



def _answer_type_candidates(mentions: dict) -> list:
   
    original = (mentions.get("refined_from") or {}).get("entities", [])
    dropped = [e for e in original if e not in mentions.get("entities", []) and e not in mentions.get("relations", [])]
    out = []
    for noun in dropped[:2]:
        term = _singular_phrase(noun)
        try:
            links = entity_linker.link_entities([term])
        except Exception:
            continue
        for c in _with_aliases(term, links.get(term) or []):
            if c.get("id") in sparql_generator.COMMON_CLASSES:
                out.append({"noun": noun, "id": c["id"], "label": c.get("label")})
                break
    return out


_TYPE_STOP = {"is", "are", "was", "were", "did", "do", "does", "has", "have", "had", "can", "could", "will", "would",
              "in", "on", "at", "of", "by", "with", "from", "to", "for", "that", "who", "whom", "whose", "which",
              "what", "and", "or", "the", "a", "an", "not", "also", "ever", "currently", "been", "be"}
_WH_START = re.compile(r"^\s*(?:how\s+many|which|what)\s+(.+)$", re.IGNORECASE)


def _term_variants(span: str) -> list:
   
    words = span.lower().split()
    if not words:
        return [span.lower()]
    w, out = words[-1], [span.lower()]
    if w.endswith("ies") and len(w) > 4:
        out += [" ".join(words[:-1] + [w[:-1]]), " ".join(words[:-1] + [w[:-3] + "y"])]
    elif w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        out.append(" ".join(words[:-1] + [w[:-1]]))
    return list(dict.fromkeys(out))


def _with_aliases(term: str, cands: list) -> list:
    try:
        extra = entity_linker._alias_candidates(term)
    except Exception:
        extra = []
    have = {c.get("id") for c in cands}
    return list(cands) + [c for c in extra if c.get("id") not in have]


def _class_usage(qid: str):
   
    best = None
    for prop in ("P31", "P106"):
        try:
            rows = sparql_executor.run_sparql(
                "PREFIX wd: <http://www.wikidata.org/entity/>\n"
                "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
                f"SELECT (COUNT(*) AS ?n) WHERE {{ SELECT ?s WHERE {{ ?s wdt:{prop} wd:{qid} }} LIMIT 1000 }}")
            n = int(float(rows[0]["n"])) if rows else 0
        except Exception:
            n = 0
        if n >= 3 and (best is None or n > best[1]):
            best = (prop, n)
    return best


def _question_type_candidates(question: str, mentions: dict) -> list:
    
    m = _WH_START.match(question or "")
    if not m:
        return []
    words = []
    for w in re.findall(r"[A-Za-z0-9][\w'.-]*", m.group(1)):
        if w.lower() in _TYPE_STOP:
            break
        words.append(w)
        if len(words) >= 3:
            break
    if not words:
        return []
    entity_texts = [e.lower() for e in mentions.get("entities", [])]
    n = len(words)
    spans = [(i, i + k) for k in range(n, 0, -1) for i in range(0, n - k + 1)]
    best = None
    for i, j in spans[:6]:
        span = " ".join(words[i:j])
        if any(span.lower() in e for e in entity_texts):
            continue
        for term in _term_variants(span):
            try:
                links = entity_linker.link_entities([term])
            except Exception:
                continue
            cands = (links.get(term) or [])[:5]
            for c in _with_aliases(term, links.get(term) or []):
                if c.get("id") in sparql_generator.COMMON_CLASSES:
                    return [{"noun": span, "id": c["id"], "label": c.get("label"), "prop": "P31"}]
            for c in cands:
                if (c.get("label") or "").lower() == term and not c.get("alias"):
                    use = _class_usage(c["id"])
                    if use and (best is None or use[1] > best[0]):
                        best = (use[1], {"noun": span, "id": c["id"], "label": c.get("label"), "prop": use[0]})
    return [best[1]] if best else []


def answer_question(question: str,few_shot: bool = True,verbose: bool = True,use_subgraph: bool = True,resolve_labels: bool = False,) -> dict:
    import time as _t
    from scripts import llm_client as _lc
    _t0, _c0, _s0 = _t.perf_counter(), _lc.STATS["calls"], _lc.STATS["seconds"]
   
    mentions = extract_mentions(question)
    if verbose:
        print(f"[1/4] Mentions: {mentions}")

  
    entity_links = entity_linker.link_entities(mentions["entities"])
    entity_links = choice_llm_entities.disambiguate_entities(question, entity_links, relations=mentions["relations"], verbose=verbose)
   
    if verbose:
        print(f"[2/4] Entity links: {entity_links}")

  
    relation_chain = list(dict.fromkeys(mentions["relations"]))
    
    path_match = None
    if use_subgraph and relation_chain:
        path_match = subgraph.find_best_path(entity_links, relation_chain)
        if verbose:
            if path_match:
                print(f"[3/4] Subgraph match found: {path_match}")
            else:
                print(f"[3/4] Subgraph: ΔΕΝ βρέθηκε μονοπάτι (relations={relation_chain})")
    elif verbose:
        print(f"[3/4] Subgraph παραλείφθηκε (use_subgraph={use_subgraph}, relations={relation_chain})")

    hint = path_match
    if path_match and path_match["score"] < HINT_MIN_SCORE:
        if verbose:
            print(f"[3/4] Το hint απορρίφθηκε (score {path_match['score']:.2f} < {HINT_MIN_SCORE})")
        hint = None

   
    property_cands = None
    if USE_SOFT_HINT and use_subgraph and relation_chain and not hint:
        property_cands = candidate_properties(entity_links, relation_chain)
        if verbose:
            print("[3/4] Soft hint (properties του entity):")
            for c in property_cands:
                print(f"       {c['relation']!r} -> {c['property_id']} {c['property_label']} "
                      f"({c['direction']}, score {c['score']:.2f})")

    relation_links = {}
    if not hint and USE_PROPERTY_INDEX:
        
        try:
            relation_links = property_index.link_relations(mentions["relations"])
            if verbose:
                print("[3/4] Property index:")
                for m, cs in relation_links.items():
                    print(f"       {m!r} -> " + ", ".join(f"{c['id']} {c['label']} ({c['score']})" for c in cs))
        except Exception as e:
            print(f"[property_index] αποτυχία ({e}), πέφτω στο API")
            relation_links = {}
    if not hint and not relation_links and not property_cands:
        relation_links = entity_linker.link_relations(mentions["relations"])

    
    answer_type = _answer_type_candidates(mentions)
    if not answer_type:
        try:
            answer_type = _question_type_candidates(question, mentions)
        except Exception as e:
            print(f"[answer_type] αποτυχία: {e}")
    if verbose and answer_type:
        print(f"[3/4] Answer type: {answer_type}")
    sparql_query = sparql_generator.generate_sparql(
        question=question,
        entity_links=entity_links,
        relation_links=relation_links,
        subgraph_hint=hint,
        few_shot=few_shot,
        property_candidates=property_cands,
        answer_type=answer_type,
    )
    method = "llm_subgraph_hybrid" if hint else ("llm_soft_hint" if property_cands else "llm")
    if "# age_template" in (sparql_query or ""):
        method = "age_template"
    elif "# alive_template" in (sparql_query or ""):
        method = "alive_template"
    allowed_qids = sparql_generator.allowed_qids(entity_links, hint, answer_type)
    sparql_query, qid_info = sparql_generator.enforce_known_qids(
        question, sparql_query, allowed_qids, entity_links, verbose=verbose)

    if verbose:
        print(f"[3/4] SPARQL ({method}):\n{sparql_query}")

    
    results, syntax_error = sparql_executor.run_sparql_checked(sparql_query)
    repair_attempts = 0
    while syntax_error and repair_attempts < MAX_SPARQL_REPAIRS:
        repair_attempts += 1
        if verbose:
            print(f"[3.2/4] Διόρθωση query ({repair_attempts}/{MAX_SPARQL_REPAIRS}), σφάλμα: {syntax_error}")
        sparql_query = sparql_generator.repair_sparql(question, sparql_query, syntax_error)
        sparql_query, _ = sparql_generator.enforce_known_qids(
            question, sparql_query, allowed_qids, entity_links, verbose=verbose)
        if verbose:
            print(f"[3.2/4] Διορθωμένο SPARQL:\n{sparql_query}")
        results, syntax_error = sparql_executor.run_sparql_checked(sparql_query)

   
    if not results:
        plain_query = sparql_generator.denormalize_quantities(sparql_query)
        if plain_query != sparql_query:
            plain_results = sparql_executor.run_sparql(plain_query)
            if plain_results:
                if verbose:
                    print(f"[3.4/4] Επανάληψη χωρίς normalization:\n{plain_query}")
                sparql_query, results = plain_query, plain_results

   
    if not results and method != "subgraph":
        reversed_query = sparql_generator.try_reverse_direction(sparql_query)
        if reversed_query:
            if verbose:
                print(f"[3.5/4] Δοκιμή αντίστροφης φοράς:\n{reversed_query}")
            reversed_results = sparql_executor.run_sparql(reversed_query)
            if reversed_results:
                sparql_query = reversed_query
                results = reversed_results

    if (not results and hint and f"wd:{hint['qid']}" in (sparql_query or "")
            and "pq:" not in (sparql_query or "")):
        try:
            hint_query = subgraph.build_path_query(hint["qid"], hint["hops"][:1])
            hint_results = sparql_executor.run_sparql(hint_query)
        except Exception as e:
            hint_results = []
            if verbose:
                print(f"[3.6/4] Αποτυχία fallback hint: {e}")
        if hint_results:
            if verbose:
                print(f"[3.6/4] Fallback στο query του hint:\n{hint_query}")
            sparql_query, results, method = hint_query, hint_results, "subgraph_fallback"

    
    fallback_used = None
    if not results and sparql_query:
        for kind, alt_query in sparql_generator.alternative_queries(
                sparql_query, entity_links, None, relation_links):
            try:
                alt_results = sparql_executor.run_sparql(alt_query)
            except Exception:
                alt_results = []
            if alt_results:
                if verbose:
                    print(f"[3.7/4] Fallback ({kind}):\n{alt_query}")
                sparql_query, results, fallback_used = alt_query, alt_results, kind
                method = f"{method}+fb_{kind}"
                break

    results = sparql_generator.keep_answer_column(results)

    
    labels_by_uri = {}
    if resolve_labels:
        all_values = [v for row in results for v in row.values() if v]
        try:
            labels_by_uri = entity_linker.get_labels_for_uris(all_values)
        except Exception as e:
            if verbose:
                print(f"[pipeline] Η αναζήτηση labels απέτυχε ({e}). Συνεχίζω με τις raw τιμές.")

    labeled_results = []
    for row in results:
        labeled_row = {}
        for var, value in row.items():
            labeled_row[var] = labels_by_uri.get(value, value)
        labeled_results.append(labeled_row)
    
    if verbose:
        print("\n" + "="*50)
        print(" ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ:")
        print(f" Raw Results: {results}")
        print(f" Labeled Results: {labeled_results}")
        print("="*50 + "\n")


    return {
        "question": question,
        "mentions": mentions,
        "entity_links": entity_links,
        "relation_links": relation_links,
        "method": method,
        "subgraph_match": path_match,
        "property_candidates": property_cands,
        "repair_attempts": repair_attempts,
        "timing": {"total": round(_t.perf_counter() - _t0, 1), "llm_calls": _lc.STATS["calls"] - _c0,
                   "llm_s": round(_lc.STATS["seconds"] - _s0, 1)},
        "qid_check": qid_info,
        "answer_type": answer_type,
        "fallback": fallback_used,
        "sparql": sparql_query,
        "results": results,
        "labeled_results": labeled_results,
    }


def diagnostics(result: dict) -> dict:
   
    def top(cands, n=3):
        return [[c.get("id"), c.get("label")] for c in (cands or [])[:n]]

    hint = result.get("subgraph_match") or {}
    soft = result.get("property_candidates") or []
    return {
        "mentions": result.get("mentions"),
        "entities": {m: top(c) for m, c in (result.get("entity_links") or {}).items()},
        "relations": {m: top(c, 4) for m, c in (result.get("relation_links") or {}).items()},
        "hint": ({"qid": hint.get("qid"), "path": [h.get("property_id") for h in hint.get("hops", [])],
                  "score": round(hint.get("score", 0), 2)} if hint else None),
        "soft_hint": [[c["property_id"], c["direction"], round(c["score"], 2)] for c in soft[:6]],
        "repair_attempts": result.get("repair_attempts"),
        "timing": result.get("timing"),
        "qid_check": result.get("qid_check"),
        "answer_type": result.get("answer_type"),
        "fallback": result.get("fallback"),
    }
