

import re

_QUESTION_WORDS = {
    "who", "whom", "whose", "what", "which", "where", "when", "why", "how",
    "who's", "what's", "where's", "when's", "how's", "whos", "whats",
}
_AUX_WORDS = {"is", "are", "was", "were", "do", "does", "did",
              "has", "have", "had", "can", "could", "will", "would"}
_MEASURE_WORDS = {"many", "much"}   # "how many", "how much"
_ARTICLES = {"the", "a", "an"}


def _strip_leading(text: str, drop: set[str]) -> str:

    words = text.split()
    while words and re.sub(r"[^\w']", "", words[0].lower()) in drop:
        words.pop(0)
    return " ".join(words).strip()


def clean_entity(mention: str) -> str:
   
    return _strip_leading(mention, _QUESTION_WORDS)


def clean_relation(mention: str) -> str:
    return _strip_leading(mention, _QUESTION_WORDS | _AUX_WORDS | _MEASURE_WORDS | _ARTICLES)


def clean_mentions(data: dict) -> dict:
   
    out = {}
    for key, cleaner in (("entities", clean_entity), ("relations", clean_relation)): #"entities": [...], "relations": [...]
        values = data.get(key, [])
        if isinstance(values, str):          # μπορει "director" αντι για ["director"] 
            values = [values]
        seen, cleaned = set(), []
        for v in values:
            if not isinstance(v, str):
                continue
            c = cleaner(v)
            if c and c.lower() not in seen:
                seen.add(c.lower())
                cleaned.append(c)
        out[key] = cleaned
    return out


if __name__ == "__main__":
        # Test cases
        test_data = {
            "entities": [
                "Who is the president of the United States?",
                "The Beatles",
                "A quick brown fox",
                "An apple a day keeps the doctor away",
            ],
            "relations": [
                "What is the capital of France?",
                "Is it raining?",
                "How many states are in the USA?",
            ],
        }

        test_data1={
            "entities": "director", "relations": "the director of X's spouse"}
        cleaned_data = clean_mentions(test_data)
        cleaned_data1 = clean_mentions(test_data1)
        print(cleaned_data)
        print(cleaned_data1)