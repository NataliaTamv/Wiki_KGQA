

import re

from . import entity_linker
from .llm_client import call_llm

DISAMBIGUATION_SYSTEM_PROMPT = """You are given a question, an entity mention
taken from that question, the relation the question asks about, and a numbered
list of candidate Wikidata entities (label and description). Choose the
candidate that the question is actually about. Think about what kind of entity
could have the asked relation (for example a country has a capital, a given
name does not).

First write ONE short sentence of reasoning. Then, on a new line, write
"Answer: N" where N is the candidate number, or "Answer: 0" if none fits.

Example:
Question: Who directed Titanic?
Mention: "Titanic"
Relation: directed by
Candidates:
1. Titanic - British passenger liner that sank in 1912
2. Titanic - 1997 film directed by James Cameron
Reasoning: The question asks who directed it, so it must be a film.
Answer: 2
"""


def _normalize(text: str | None) -> str:
    return (text or "").strip().lower()


def _build_prompt(question: str, mention: str, relations: list[str],
                  candidates: list[dict]) -> str:
    lines = [f"Question: {question}", f'Mention: "{mention}"']
    if relations:
        lines.append(f"Relation: {', '.join(relations)}")
    lines.append("Candidates:")
    for i, c in enumerate(candidates, start=1):
        description = c.get("description") or "no description"
        lines.append(f"{i}. {c.get('label')} - {description}")
    lines.append("Reasoning:")
    return "\n".join(lines)


def _parse_choice(raw: str, n_candidates: int) -> int | None:
    matches = re.findall(r"Answer:\s*(\d+)", raw, flags=re.IGNORECASE)
    if not matches:
        return None
    choice = int(matches[-1])
    if 1 <= choice <= n_candidates:
        return choice - 1
    return None


def _says_none(raw: str) -> bool:
  
    matches = re.findall(r"Answer:\s*(\d+)", raw, flags=re.IGNORECASE)
    return bool(matches) and int(matches[-1]) == 0


def _recover_by_expansion(question: str, mention: str):
    
    try:
        from . import config
        if not getattr(config, "EMBEDDINGS_DIR", None):
            return None
        from .local_entity_search import expand_mention_local
    except ImportError:
        return None
    return expand_mention_local(question, mention)


def disambiguate_mention(question: str, mention: str, candidates: list[dict],
                         relations: list[str] | None = None,
                         always_ask: bool = False, verbose: bool = False,
                         _recovered: bool = False) -> list[dict]:
    if len(candidates) <= 1:
        return candidates

    mention_norm = _normalize(mention)
    exact = [c for c in candidates
             if _normalize(c.get("label")) == mention_norm or c.get("alias")]

    
    if len(exact) == 1 and not always_ask:
        chosen = exact[0]
        return [chosen] + [c for c in candidates if c is not chosen]

  
    shortlist = exact if exact else candidates

    
    entity_linker.add_descriptions(shortlist)

    prompt = _build_prompt(question, mention, relations or [], shortlist)
    raw = call_llm(prompt, system=DISAMBIGUATION_SYSTEM_PROMPT, max_tokens=150)
    idx = _parse_choice(raw, len(shortlist))

    if verbose:
        print(f"[disambiguation] '{mention}': LLM -> {raw!r} (index {idx})")

    if idx is None:
        
        if _says_none(raw) and not _recovered:
            recovered = _recover_by_expansion(question, mention)
            if recovered:
                new_mention, new_candidates = recovered
                if verbose:
                    print(f"[disambiguation] '{mention}' -> επεκτάθηκε σε '{new_mention}' "
                          f"({len(new_candidates)} candidates)")
                ranked = disambiguate_mention(question, new_mention, new_candidates, relations,
                                              always_ask, verbose, _recovered=True)
                seen, tail = {c["id"] for c in ranked}, []
                for c in shortlist + candidates:           
                    if c["id"] not in seen:
                        seen.add(c["id"])
                        tail.append(c)
                return ranked + tail
        
        return shortlist + [c for c in candidates if c not in shortlist]

    chosen = shortlist[idx]
    return [chosen] + [c for c in candidates if c is not chosen]


def disambiguate_entities(question: str, entity_links: dict[str, list[dict]],
                          relations: list[str] | None = None,
                          always_ask: bool = False, verbose: bool = False) -> dict[str, list[dict]]:
    return {
        mention: disambiguate_mention(question, mention, candidates, relations,
                                      always_ask, verbose)
        for mention, candidates in entity_links.items()
    }