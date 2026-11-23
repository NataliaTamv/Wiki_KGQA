

import re
from settings_run import config
from .llm_client import call_llm
from .few_shot_selector import select_few_shot_examples, load_training_file

try:
    _TRAINING_EXAMPLES = load_training_file(config.TRAINING_DATA_PATH, language="en")
    print(f"[sparql_generator] Φορτώθηκαν {len(_TRAINING_EXAMPLES)} training παραδείγματα από {config.TRAINING_DATA_PATH}")
except FileNotFoundError:
    print("Πρόβλημα στο filepath του training_set")
    _TRAINING_EXAMPLES = []

SYSTEM_PROMPT = """You are an expert at converting natural language questions into
SPARQL queries for the Wikidata knowledge graph.
Use standard Wikidata prefixes (wd:, wdt:, p:, ps:, pq:, wikibase:, rdfs:).

CRITICAL RULE FOR LABELS: 
NEVER use the `SERVICE wikibase:label` block. It is strictly forbidden.
If you need labels, use `?var rdfs:label ?varLabel . FILTER(LANG(?varLabel) = "en")`.
If the question asks for a count ("How many"), do NOT request labels at all, just use COUNT(?var).

CRITICAL RULE FOR FILTERS:
Do NOT add language filters on date, number or entity values; only use LANG()
on rdfs:label variables. Always put a "." between a triple pattern and a
FILTER, and wrap the whole condition in parentheses: FILTER(LANG(?x) = "en").

CRITICAL RULE FOR SYNTAX:
A FILTER takes only a boolean expression such as FILTER(?x > 5). NEVER put a triple
pattern inside FILTER; write it as a normal triple before the FILTER. GROUP BY,
ORDER BY and LIMIT go AFTER the closing brace of the WHERE block, never inside it.

CRITICAL RULE FOR SUPERLATIVES:
For "largest", "smallest", "highest", "first", "last", "oldest" ... do NOT use
GROUP BY, SAMPLE or FILTER NOT EXISTS. Select the entity and the value property, then
ORDER BY DESC(?value) (or ASC) LIMIT 1 after the WHERE block.

CRITICAL RULE FOR YES/NO QUESTIONS:
If the question can be answered with yes or no (it starts with Is, Are, Was, Were,
Does, Do, Did, Has, Have, Can ...), write an ASK query, NOT a SELECT or COUNT.
Example: "Is Berlin the capital of Germany?" ->
ASK { wd:Q64 wdt:P1376 wd:Q183 }

IMPORTANT -- pick ONE property: choose the SINGLE property that best matches the
question. Do NOT try to use several candidate properties together.

IMPORTANT -- Graph Path Hint: If a "CRITICAL GRAPH HINT" is provided, you MUST
use EXACTLY the triple pattern provided in the hint. Do not invent other properties.
Just take the provided triple pattern and apply SPARQL logic (like COUNT) around it.
 
Only output the SPARQL query, wrapped in a ```sparql code block, nothing else."""

_VALID_STARTS = ("SELECT", "ASK", "CONSTRUCT", "DESCRIBE", "PREFIX", "WITH")

_KNOWN_PREFIXES = {
    "wd": "http://www.wikidata.org/entity/",
    "wdt": "http://www.wikidata.org/prop/direct/",
    "p": "http://www.wikidata.org/prop/",
    "ps": "http://www.wikidata.org/prop/statement/",
    "pq": "http://www.wikidata.org/prop/qualifier/",
    "psv": "http://www.wikidata.org/prop/statement/value/",
    "psn": "http://www.wikidata.org/prop/statement/value-normalized/",
    "pqv": "http://www.wikidata.org/prop/qualifier/value/",
    "pqn": "http://www.wikidata.org/prop/qualifier/value-normalized/",
    "wdtn": "http://www.wikidata.org/prop/direct-normalized/",
    "schema": "http://schema.org/",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "wikibase": "http://wikiba.se/ontology#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "bd": "http://www.bigdata.com/rdf#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
}


def _format_candidates(cands: list[dict], max_candidates: int = 3) -> str:
    if not cands:
        return "(δεν βρέθηκε)"
    lines = []
    for c in cands[:max_candidates]:
        lines.append(f"  - {c['id']} ({c['label']}): {c.get('description') or 'no description'}")
    return "\n" + "\n".join(lines)

def _format_property_candidates(cands: list[dict]) -> str:
    lines = ["PROPERTIES THAT THESE ENTITIES ACTUALLY HAVE (ranked by similarity to the question's relation; "
             "a suggestion, not certain). If one of them expresses what the question asks, use it; "
             "otherwise rely on your own knowledge. All rules above still apply, e.g. yes/no questions "
             "must be ASK queries and superlatives use ORDER BY ... LIMIT 1 without GROUP BY:"]
    for c in cands:
        if c.get("direction") == "in":
            pattern = f"?x wdt:{c['property_id']} wd:{c['qid']}"
        else:
            pattern = f"wd:{c['qid']} wdt:{c['property_id']} ?x"
        lines.append(f"  - {c['property_label']} ({c.get('entity_label')}): {pattern}")
    return "\n".join(lines)


def _build_prompt(
    question: str, 
    entity_links: dict, 
    relation_links: dict, 
    few_shot: bool, 
    subgraph_hint: dict = None,
    property_candidates: list = None,
    answer_type: list = None,
    other_entities: list = None
) -> str:

    entities_str = []
    for mention, cands in entity_links.items():
        line = f"{mention} -> {_format_candidates(cands)}"
        entities_str.append(line)
    entities_str = "\n".join(entities_str)

    relations_str = []
    for mention, cands in relation_links.items():
        line = f"{mention} -> {_format_candidates(cands)}"
        relations_str.append(line)
    relations_str = "\n".join(relations_str)

    prompt_parts = []

   
    if few_shot and _TRAINING_EXAMPLES:
        examples = select_few_shot_examples(question, _TRAINING_EXAMPLES, k=config.FEW_SHOT_K)
        if examples:
            prompt_parts.append(examples)

  
    context = f"""Question: {question}
Linked entities:
{entities_str or "(καμία)"}"""
    
    if relation_links:
        context += f"\nLinked relations:\n{relations_str}"

    if answer_type:
        context += ("\nExpected answer type (the answers are instances of this class; constrain with "
                    "?x wdt:P31/wdt:P279* wd:ID when it helps):")
        for t in answer_type:
            context += f"\n  - {t['noun']} -> wd:{t['id']} ({t.get('label')})"
            if t.get("prop"):
                _p = "P31/wdt:P279*" if t["prop"] == "P31" else t["prop"]
                context += f" [filter: ?x wdt:{_p} wd:{t['id']}]"
    if other_entities:   # (8a)
        context += ("\nOther entities that may be relevant (matched on phrases of the question). Use one ONLY if the "
                    "question needs it, e.g. as a class or as a constraint; otherwise ignore it:")
        for o in other_entities:
            context += f"\n  - {o['phrase']} -> wd:{o['id']} ({o.get('label')}: {o.get('description') or 'no description'})"
    prompt_parts.append(context)

    
    if subgraph_hint:
        qid = subgraph_hint.get('qid')
        hops = subgraph_hint.get('hops', [])
        
        hint_text = "CRITICAL GRAPH HINT: An external search confirmed the exact path.\n"
        hint_text += "You MUST build your query around this exact triple pattern:\n\n"
        
        for hop in hops:
            prop = hop.get('property_id')
            direction = hop.get('direction', 'out')
            if direction == 'in':
                hint_text += f"?obj wdt:{prop} wd:{qid} .\n"
            else:
                hint_text += f"wd:{qid} wdt:{prop} ?obj .\n"
                
       
        others = []
        for mention, cands in entity_links.items():
            if cands and cands[0].get("id") != qid:
                c = cands[0]
                others.append((mention, c.get("id"), c.get("label"), c.get("description")))
        if others:
            hint_text += "\nThe question also mentions other entities. If the question needs them, use them as extra constraints (an additional triple or a qualifier value) instead of ignoring them:\n"
            for mention, oid, olabel, odesc in others:
                hint_text += f"  - {mention} -> wd:{oid} ({olabel}: {odesc or 'no description'})\n"
            if any(h.get("property_id") == "P161" for h in hops):
                hint_text += ("For 'who played <character> in <work>' use the cast-member statement with the "
                              "character-role qualifier, e.g.:\n"
                              f"wd:{qid} p:P161 ?st . ?st ps:P161 ?obj . ?st pq:P453 wd:<character id> .\n")
        hint_text += ("\nDo NOT invent other properties (like P22). Just use the pattern above and apply "
                      "DISTINCT, ORDER BY, etc. as required by the question. Only if the question "
                      "asks 'how many' of something countable use COUNT; a question asking for a value (e.g. 'how many "
                      "floors', 'what religion') just selects ?obj. For a yes/no question write ASK { ... }.")
        prompt_parts.append(hint_text)

    if property_candidates and not subgraph_hint:
        prompt_parts.append(_format_property_candidates(property_candidates))

 
    prompt_parts.append("SPARQL:")

    return "\n\n".join(prompt_parts)


def _strip_stray_language_tag(content: str) -> str:
    content = content.strip()
    lines = content.split("\n", 1)
    if lines[0].strip().lower() == "sparql" and len(lines) > 1:
        content = lines[1].strip()
    return content


def _strip_until_valid_start(content: str) -> str:
    content = content.strip()
    while content:
        first_word_match = re.match(r"[A-Za-z]+", content)
        if first_word_match:
            first_word = first_word_match.group(0).upper()
        else:
            first_word = ""

        if first_word in _VALID_STARTS:
            return content

        lines = content.split("\n", 1)
        if len(lines) < 2:
            return content
        content = lines[1].strip()
    return content


def _extract_sparql(raw_response: str) -> str:
   
    match = re.search(r"```sparql\s*(.*?)```", raw_response, re.DOTALL | re.IGNORECASE)
    if match:
        content = _strip_stray_language_tag(match.group(1))
        return _strip_until_valid_start(content)

    
    match = re.search(r"```(.*?)```", raw_response, re.DOTALL)
    if match:
        content = _strip_stray_language_tag(match.group(1))
        return _strip_until_valid_start(content)

    
    content = raw_response.strip().strip("`").strip()
    content = _strip_stray_language_tag(content)
    return _strip_until_valid_start(content)


_LANG_FILTER = re.compile(
    r'FILTER\s*\(?\s*lang\(\s*(\?\w+)\s*\)\s*=\s*("[^"]*")\s*\)?',
    re.IGNORECASE,
)

_MISSING_DOT_BEFORE_FILTER = re.compile(
    r'(\?\w+|<[^>\s]*>|\w+:\w+|"[^"]*"(?:@[\w-]+)?)\s+(FILTER\b)',
    re.IGNORECASE,
)



_TRIPLE_IN_FILTER = re.compile(
    r'FILTER\s*\(\s*(\?\w+\s+wdt:P\d+\s+(?:wd:Q\d+|\?\w+))\s*\.?\s*\)',
    re.IGNORECASE,
)

_MODIFIERS_INSIDE_WHERE = re.compile(
    r'\s*((?:GROUP\s+BY|HAVING|ORDER\s+BY|LIMIT|OFFSET)\b[^{}]*?)\s*\}\s*$',
    re.IGNORECASE,
)


def _fix_filter_syntax(query: str) -> str:
    
    if not query.strip():
        return query

    query = _TRIPLE_IN_FILTER.sub(r"\1 .", query)
    
    query = re.sub(r"(\s)\.(?:\s+\.)+(?=\s|\}|$)", r"\1.", query)

    def repl(m: re.Match) -> str:
        var, literal = m.group(1), m.group(2)
        if re.search(rf"rdfs:label\s+{re.escape(var)}(?![\w])", query):
            return f"FILTER(LANG({var}) = {literal})"
        return ""

    query = _LANG_FILTER.sub(repl, query)
    query = _MISSING_DOT_BEFORE_FILTER.sub(r"\1 . \2", query)
    
    query = re.sub(r"^[ \t]*\.[ \t]*$\n?", "", query, flags=re.MULTILINE)
    query = re.sub(r"\{\s*\.", "{", query)

    m = _MODIFIERS_INSIDE_WHERE.search(query)
    if m:
        query = query[:m.start()] + "\n}\n" + m.group(1).strip()
    return query



def _ensure_prefixes(query: str) -> str:
    if not query.strip():
        return query

  
    lines = query.split('\n')
    clean_lines = []
    
    
    for line in lines:
        if not re.match(r"^\s*PREFIX\s+", line, re.IGNORECASE):
            clean_lines.append(line)
            
    query_body = "\n".join(clean_lines).strip()

   
    needed_lines = []
    for prefix, uri in _KNOWN_PREFIXES.items():
        
        if re.search(rf"(?<![\w:]){re.escape(prefix)}:", query_body):
            needed_lines.append(f"PREFIX {prefix}: <{uri}>")

    if not needed_lines:
        return query_body

    return "\n".join(needed_lines) + "\n\n" + query_body

_ORDER_BY_BARE = re.compile(r"ORDER\s+BY\s+(\?\w+)\s+(ASC|DESC)\b(?!\s*\()", re.IGNORECASE)
_SELECT_COUNT_NO_ALIAS = re.compile(r"SELECT\s+(COUNT\s*\([^()]*\))\s*(?=WHERE|\{)", re.IGNORECASE)
_COUNT_OF_QUANTITY = re.compile(
    r"SELECT\s*\(\s*COUNT\s*\(\s*(?:DISTINCT\s+)?(\?\w+)\s*\)\s+AS\s+\?\w+\s*\)\s*(?:WHERE\s*)?\{\s*"
    r"(wd:Q\d+\s+p:P\d+/psn:P\d+/wikibase:quantityAmount\s+)\1\s*\.?\s*\}", re.IGNORECASE)


_STATEMENT_VAR = re.compile(r"(\?\w+)\s+(?:ps|psv|psn):(P\d+)\b")


def _fix_statement_nodes(query: str) -> str:
    
    for var, pid in set(_STATEMENT_VAR.findall(query)):
        query = re.sub(rf"\bwdt:{pid}(\s+){re.escape(var)}(?![\w])", rf"p:{pid}\1{var}", query)
    return query


def _fix_qlever_syntax(query: str) -> str:
    
    query = _fix_statement_nodes(query)
    query = _unwrap_sample_superlative(query)
    query = _ORDER_BY_BARE.sub(lambda m: f"ORDER BY {m.group(2).upper()}({m.group(1)})", query)
    query = _SELECT_COUNT_NO_ALIAS.sub(lambda m: f"SELECT ({m.group(1)} AS ?count) ", query)
    return query


_NORMALIZED_PATH = re.compile(r"p:(P\d+)/psn:\1/wikibase:quantityAmount")


def denormalize_quantities(query: str) -> str:
   
    return _ensure_prefixes(_NORMALIZED_PATH.sub(lambda m: f"wdt:{m.group(1)}", query))


def _normalize_quantities(query: str) -> str:
   
    query = _fix_qlever_syntax(query)
    try:
        from .property_index import normalize_quantities
        query = normalize_quantities(query)
    except Exception:
        return query
    return _COUNT_OF_QUANTITY.sub(lambda m: f"SELECT {m.group(1)} WHERE {{ {m.group(2)}{m.group(1)} }}", query)


_YES_NO_QUESTION = re.compile(
    r"^\s*(is|are|was|were|does|do|did|has|have|had|can|could|will|would|should)\b", re.IGNORECASE)
_ALIVE_QUESTION = re.compile(r"\b(alive|living)\b", re.IGNORECASE)


def _ask_for_yes_no(question: str, query: str) -> str:
   
    if not _YES_NO_QUESTION.match(question or ""):
        return query
    sel = re.search(r"\bSELECT\b", query, re.IGNORECASE)
    if not sel or re.search(r"\bASK\b", query, re.IGNORECASE):
        return query
    start = query.find("{", sel.end())
    if start < 0:
        return query
    depth, end = 0, -1
    for i in range(start, len(query)):
        if query[i] == "{":
            depth += 1
        elif query[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return query
    body = query[start + 1:end].strip()
    if not body:
        return query
    if _ALIVE_QUESTION.search(question):
        body = f"FILTER NOT EXISTS {{ {body} }}"
    return f"{query[:sel.start()]}ASK {{ {body} }}"


def generate_sparql(
    question: str, 
    entity_links: dict, 
    relation_links: dict, 
    subgraph_hint: dict = None, 
    few_shot: bool = True,
    property_candidates: list = None,
    answer_type: list = None,
    other_entities: list = None
) -> str:
    
    age_query = build_age_query(question, entity_links)
    if age_query:
        return age_query
    alive_query = build_alive_query(question, entity_links)
    if alive_query:
        return alive_query

    prompt = _build_prompt(
        question=question, 
        entity_links=entity_links, 
        relation_links=relation_links, 
        few_shot=few_shot, 
        subgraph_hint=subgraph_hint,
        property_candidates=property_candidates,
        answer_type=answer_type,
        other_entities=other_entities
    )

    raw_response = call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=400)
    query = _extract_sparql(raw_response)
    
    return _widen_types(_ensure_prefixes(_ask_for_yes_no(question, _normalize_quantities(_fix_filter_syntax(query)))),
                        answer_type)


REPAIR_PROMPT = """The following SPARQL query for a Wikidata question was rejected by the
SPARQL server as invalid.

Question: {question}

Query:
```sparql
{query}
```

Server error:
{error}

Fix the syntax error and output the corrected query. Keep the same entities and
properties; change only what is needed to make the query valid. Remember: FILTER
takes a parenthesised boolean expression (never a triple pattern), GROUP BY / ORDER BY /
LIMIT go AFTER the closing brace of WHERE, and yes/no questions use ASK.

SPARQL:"""


def repair_sparql(question: str, query: str, error: str) -> str:
    """Ζητά από το LLM να διορθώσει ένα query που απορρίφθηκε από τον server (HTTP 400)."""
    prompt = REPAIR_PROMPT.format(question=question, query=query, error=error)
    raw_response = call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=400)
    repaired = _extract_sparql(raw_response)
    return _ensure_prefixes(_ask_for_yes_no(question, _normalize_quantities(_fix_filter_syntax(repaired))))


_SINGLE_PROPERTY_TRIPLE = re.compile(
    r"(\S+)\s+(wdt:P\d+)\s+([^\s.}]+)"
)


def try_reverse_direction(query: str) -> str | None:
    matches = list(_SINGLE_PROPERTY_TRIPLE.finditer(query))
    if len(matches) != 1:
        return None

    subj, pred, obj = matches[0].groups()
    reversed_triple = f"{obj} {pred} {subj} "
    start, end = matches[0].span()
    return query[:start] + reversed_triple + query[end:]


_SAMPLE_ALIAS_RE = re.compile(r"\(\s*SAMPLE\s*\(\s*(\?\w+)\s*\)\s+AS\s+(\?\w+)\s*\)", re.IGNORECASE)
_OTHER_AGGREGATE_RE = re.compile(r"\b(?:COUNT|SUM|AVG|MIN|MAX|GROUP_CONCAT)\s*\(", re.IGNORECASE)


def _unwrap_sample_superlative(query: str) -> str:
    """SELECT ?e (SAMPLE(?v) AS ?a) ... GROUP BY ?e ORDER BY DESC(?a) LIMIT 1  ->  SELECT ?e ?v ... ORDER BY DESC(?v) LIMIT 1"""
    if not (re.search(r"\bORDER\s+BY\b", query, re.I) and re.search(r"\bLIMIT\s+\d+\b", query, re.I)):
        return query
    if not _SAMPLE_ALIAS_RE.search(query) or _OTHER_AGGREGATE_RE.search(query):
        return query
    for var, alias in _SAMPLE_ALIAS_RE.findall(query):
        query = _SAMPLE_ALIAS_RE.sub(lambda m: m.group(1), query, count=1)
        if alias != var:
            query = re.sub(re.escape(alias) + r"(?!\w)", var, query)
    return re.sub(r"\bGROUP\s+BY\s+(?:\?\w+\s*)+", "", query, flags=re.I)



from collections import Counter as _Counter

_QID_RE = re.compile(r"\bwd:(Q\d+)(?!\d)")
_CLASS_QID_RE = re.compile(r"wdt:P(?:31|279)(?:/wdt:P279\*)?\s+wd:(Q\d+)")
_PROP_RE = re.compile(r"\b(?:wdt|p|ps|psn|psv):(P\d+)\b")
_STRUCTURAL_PROPS = {"P31", "P279"}


def _common_class_qids(min_count: int = 2) -> set:
    
    counts = _Counter()
    for ex in _TRAINING_EXAMPLES:
        counts.update(set(_CLASS_QID_RE.findall(ex.get("sparql", ""))))
    return {q for q, n in counts.items() if n >= min_count}


COMMON_CLASSES = _common_class_qids()


def allowed_qids(entity_links: dict, hint: dict = None, answer_type: list = None, top: int = 3, extra: list = None) -> set:
    ids = set(COMMON_CLASSES)
    for cands in (entity_links or {}).values():
        ids |= {c.get("id") for c in (cands or [])[:top] if c.get("id")}
    if hint and hint.get("qid"):
        ids.add(hint["qid"])
    for t in answer_type or []:
        ids.add(t["id"])
    for o in extra or []:   # (8a)
        ids.add(o["id"])
    return ids


def foreign_qids(query: str, allowed: set) -> list:
    return [q for q in dict.fromkeys(_QID_RE.findall(query or "")) if q not in allowed]


FIX_QID_PROMPT = """The following SPARQL query for a Wikidata question uses entity IDs that were NOT provided
and may be invented: {foreign}.

Question: {question}

Query:
```sparql
{query}
```

Allowed entity IDs (use ONLY these, plus well-known classes you are certain of):
{allowed}

Rewrite the query so that every wd:Q... is one of the allowed IDs. Do not invent IDs. If a constraint needs
an entity that was not provided, drop that constraint or express it with another property. Keep the rest
of the query and obey all the rules above.

SPARQL:"""


def enforce_known_qids(question: str, query: str, allowed: set, entity_links: dict = None, verbose: bool = False):
    """Αν το query έχει QIDs εκτός του allowed, ξαναρωτά το LLM μία φορά. Επιστρέφει (query, info)."""
    foreign = foreign_qids(query, allowed)
    info = {"foreign": foreign, "fixed": False}
    if not foreign:
        return query, info
    # Μηχανική αντικατάσταση: ένα ξένο QID + ένα επιλεγμένο entity που δεν χρησιμοποιήθηκε -> αντικατάσταση.
    in_query = set(_QID_RE.findall(query))
    unused = [c[0]["id"] for c in (entity_links or {}).values() if c and c[0].get("id") and c[0]["id"] not in in_query]
    if len(foreign) == 1 and len(unused) == 1:
        new = re.sub(rf"\bwd:{foreign[0]}(?!\d)", f"wd:{unused[0]}", query)
        info.update({"fixed": True, "after": [], "substituted": [foreign[0], unused[0]]})
        if verbose:
            print(f"[qid_check] αντικατάσταση wd:{foreign[0]} -> wd:{unused[0]} (χωρίς LLM)")
        return new, info
    lines = []
    for mention, cands in (entity_links or {}).items():
        for c in (cands or [])[:3]:
            lines.append(f"  - wd:{c.get('id')} ({c.get('label')}): {c.get('description') or 'no description'}")
    shown = "\n".join(lines) or "  (none)"
    prompt = FIX_QID_PROMPT.format(foreign=", ".join(foreign), question=question, query=query, allowed=shown)
    try:
        raw = call_llm(prompt, system=SYSTEM_PROMPT, max_tokens=400)
        new = _extract_sparql(raw)
        new = _ensure_prefixes(_ask_for_yes_no(question, _normalize_quantities(_fix_filter_syntax(new))))
    except Exception as e:
        if verbose:
            print(f"[qid_check] αποτυχία ({e})")
        return query, info
    new_foreign = foreign_qids(new, allowed)
    if verbose:
        print(f"[qid_check] QIDs εκτός λίστας: {foreign} -> μετά τη διόρθωση: {new_foreign}")
    if new.strip() and len(new_foreign) < len(foreign):
        info["fixed"] = True
        info["after"] = new_foreign
        return new, info
    return query, info


def alternative_queries(query: str, entity_links: dict, property_cands: list = None,
                        relation_links: dict = None, limit: int = 8) -> list:
   
    out = []
    if not query or re.search(r"\bASK\b", query, re.I):
        return out
    used = set(_QID_RE.findall(query))
    for mention, cands in list((entity_links or {}).items())[:2]:
        if not cands or cands[0].get("id") not in used:
            continue
        first = cands[0]["id"]
        for alt in cands[1:3]:
            aid = alt.get("id")
            if aid and aid not in used:
                out.append(("entity", re.sub(rf"\bwd:{first}(?!\d)", f"wd:{aid}", query)))
    props = set(_PROP_RE.findall(query)) - _STRUCTURAL_PROPS
    if len(props) == 1:
        cur = next(iter(props))
        alts = []
        for c in property_cands or []:
            alts.append(c.get("property_id"))
        for cands in (relation_links or {}).values():
            alts += [c.get("id") for c in (cands or [])[:5] if (c.get("score") or 0) >= 0.75]
        seen = {cur}
        for pid in alts:
            if not pid or pid in seen or not re.fullmatch(r"P\d+", pid):
                continue
            seen.add(pid)
            out.append(("property", re.sub(rf"\b((?:wdt|p|ps|psn|psv):){cur}\b", lambda m: m.group(1) + pid, query)))
            if len(seen) >= 5:
                break
    return out[:limit]



import datetime as _dt

_AGE_QUESTION = re.compile(r"^\s*(?:what\s+age|how\s+old|what\s+is\s+the\s+age\s+of|what[\'\u2019]s\s+the\s+age\s+of)\b", re.IGNORECASE)
_AT_DEATH = re.compile(r"\b(?:when|at\s+the\s+time)\b.*\b(?:died|dies|death|passed\s+away)\b", re.IGNORECASE)


def build_age_query(question: str, entity_links: dict, today: "_dt.date | None" = None):
    
    if not _AGE_QUESTION.match(question or "") or len(entity_links or {}) != 1:
        return None
    cands = next(iter(entity_links.values()))
    if not cands or not cands[0].get("id"):
        return None
    qid = cands[0]["id"]
    today = today or _dt.date.today()
    if _AT_DEATH.search(question):
        end = "?d"
        end_triple = f"wd:{qid} wdt:P570 ?d ."
    else:
        end = f'"{today.isoformat()}"^^xsd:date'
        end_triple = ""
    return (
        "# age_template\n"
        "PREFIX wd: <http://www.wikidata.org/entity/>\n"
        "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n"
        "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>\n\n"
        "SELECT DISTINCT ?age WHERE {\n"
        f"  wd:{qid} wdt:P569 ?b .\n"
        + (f"  {end_triple}\n" if end_triple else "") +
        f"  BIND((YEAR({end}) - YEAR(?b) - IF(MONTH({end}) < MONTH(?b) || (MONTH({end}) = MONTH(?b) && DAY({end}) < DAY(?b)), 1, 0)) AS ?age)\n"
        "}"
    )


_ALIVE_DEAD = re.compile(r"^\s*(?:is|are|was|were)\b.*\b(alive|living|dead|deceased)\b[\s?.!]*$", re.IGNORECASE)


def build_alive_query(question: str, entity_links: dict):
    
    m = _ALIVE_DEAD.match(question or "")
    if not m or len(entity_links or {}) != 1:
        return None
    cands = next(iter(entity_links.values()))
    if not cands or not cands[0].get("id"):
        return None
    qid = cands[0]["id"]
    alive = m.group(1).lower() in ("alive", "living")
    body = f"wd:{qid} wdt:P570 ?d ."
    if alive:
        body = f"FILTER NOT EXISTS {{ {body} }}"
    return ("# alive_template\n"
            "PREFIX wd: <http://www.wikidata.org/entity/>\n"
            "PREFIX wdt: <http://www.wikidata.org/prop/direct/>\n\n"
            f"ASK {{ {body} }}")



_ENTITY_PREFIX = "http://www.wikidata.org/entity/"


def keep_answer_column(results: list) -> list:
    
    if not results or len(results[0]) <= 1:
        return results
    keys = list(results[0].keys())
    entity_keys = [k for k in keys
                   if any(isinstance(r.get(k), str) and r[k].startswith(_ENTITY_PREFIX) for r in results)]
    key = entity_keys[0] if entity_keys else keys[0]
    return [{key: r.get(key)} for r in results]


def _widen_types(query: str, answer_type: list = None) -> str:
    
    ids = set(COMMON_CLASSES) | {t.get("id") for t in (answer_type or [])}
    return re.sub(r"wdt:P31(\s+)wd:(Q\d+)",
                  lambda m: f"wdt:P31/wdt:P279*{m.group(1)}wd:{m.group(2)}" if m.group(2) in ids else m.group(0),
                  query or "")
