import json
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "what",
    "which",
    "who",
    "whom",
    "does",
    "do",
    "did",
    "how",
    "many",
    "much",
}


def _tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())

    words_set=set()
    for word in words:
        if word not in _STOPWORDS:
            words_set.add(word)
    return words_set


def load_training_file(path: str, language: str = "en") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    training_data = []

    for item in data.get("questions", []):
        sparql_text = item.get("sparql", "").strip()
        
        question_entries = []
        for q in item.get("question", []):
            if q.get("language") == language:
                question_entries.append(q)  
       
        if not question_entries:
            continue

        question_text = question_entries[0]["string"]

        training_data.append(
            {
                "sparql": sparql_text,
                "question": question_text,
            }
        )

    return training_data


def _format_training_example(example: dict) -> str:
    return f"""Question: {example['question']}
SPARQL:
```sparql
{example['sparql']}
```"""




_TRAINING_EMBEDDINGS_CACHE = {}
_embedder = SentenceTransformer('all-MiniLM-L6-v2')

def select_top_k(question: str, training_data: list[dict], k: int) -> list[dict]:
   
    if not training_data:
        return []



    
    train_key = id(training_data)

    if train_key not in _TRAINING_EMBEDDINGS_CACHE:
        print("[few_shot_selector] Υπολογισμός διανυσμάτων (embeddings) για το training set... ")
        train_questions = [ex["question"] for ex in training_data]
        _TRAINING_EMBEDDINGS_CACHE[train_key] = _embedder.encode(train_questions, show_progress_bar=False)
    
    train_embeddings = _TRAINING_EMBEDDINGS_CACHE[train_key]

    #  Υπολογισμός embedding για τη νέα ερώτηση
    query_embedding = _embedder.encode([question], show_progress_bar=False)

    #  Υπολογισμός Ομοιότητας (Cosine Similarity)
 
    similarities = cosine_similarity(query_embedding, train_embeddings)[0]
    picked = _category_select(question, training_data, k)
    if picked is not None:
        general = [i for i in np.argsort(similarities)[::-1] if i not in picked]
        return [training_data[i] for i in (list(picked) + general)[:k]]

    #  Ταξινόμηση και επιλογή των K καλύτερων (μεγαλύτερο cosine similarity)
    top_k_indices = np.argsort(similarities)[::-1][:k]

    return [training_data[i] for i in top_k_indices]

def category(s):
    u = s.upper()
    if re.match(r"\s*ASK", u):
        return "ask"
    if re.search(r"GROUP BY.*ORDER BY\s+(DESC|ASC)\s*\(\s*COUNT|HAVING", u, re.S):
        return "count-superlative"
    if "ORDER BY" in u and "LIMIT" in u:
        return "numeric-superlative"
    if re.search(r"\bCOUNT\b", u):
        return "count"
    if re.search(r"\bMINUS\b|NOT EXISTS", u):
        return "negation"
    if re.search(r"\bPQ:P\d+", u):
        return "qualifier"
    if re.search(r"WDT:P31\*?/WDT:P279|WDT:P31\s+WD:Q", u):
        return "type-star"
    triples = len(re.findall(r"\s\.\s|\s\.\s*}", s)) + 1
    if triples >= 2 or "/" in re.sub(r"<[^>]*>", "", s.split("{", 1)[-1]):
        return "multi-hop"
    return "simple"


AUX=r"(?:is|are|was|were|do|does|did|has|have|had|can|could|will|would|should)"
def rule_category(q):
    s=q.strip().lower()
    if re.match(rf"^{AUX}\b",s): return "ask"
    if re.search(r"\b(the )?(most|fewest|least)\b|\bmore than one\b",s): return "count-superlative"
    if re.search(r"\b(largest|biggest|highest|lowest|smallest|tallest|oldest|youngest|longest|shortest|nearest|closest|first|latest|last|best|deepest|heaviest|fastest)\b",s): return "numeric-superlative"
    if re.match(r"^how many\b|\bnumber of\b",s): return "count"
    if re.search(r"\bnot\b|n't\b|\bwithout\b|\bexcept\b|\bnon-?\w+|\bneither\b|\bno longer\b",s): return "negation"
    if re.search(r"\b(before|after|preceded|precedes|succeeded|succeeds|next|previous|predecessor|successor|follows)\b",s): return "qualifier"
    if re.match(r"^(which|what|who)\s+(?!is\b|was\b|are\b|were\b|did\b|does\b|do\b|has\b|have\b)\w+",s): return "type-star"
    if re.search(r"\b(of|by|in|from)\b.*\b(of|by|in|from|that|who|which)\b",s): return "multi-hop"
    return "simple"


_STRUCTURAL = {"ask", "count-superlative", "numeric-superlative", "count", "negation", "qualifier"}
_CAT_CACHE = {}


def _mask(q: str) -> str:
    
    toks = re.findall(r"[\w'.&-]+|[?]", q)
    out = ["X" if (i > 0 and (t[0].isupper() or t[0].isdigit())) else t.lower() for i, t in enumerate(toks)]
    return re.sub(r"(X )+X\b", "X", " ".join(out))


def _category_select(question: str, training_data: list[dict], k: int):
    
    cat = rule_category(question)
    if cat not in _STRUCTURAL:
        return None
    key = id(training_data)
    if key not in _CAT_CACHE:
        cats = [category(ex.get("sparql", "")) for ex in training_data]
        masked = _embedder.encode([_mask(ex["question"]) for ex in training_data], show_progress_bar=False)
        _CAT_CACHE[key] = (cats, masked)
    cats, masked = _CAT_CACHE[key]
    in_cat = [i for i, c in enumerate(cats) if c == cat]
    if len(in_cat) < 3:
        return None
    sims = cosine_similarity(_embedder.encode([_mask(question)], show_progress_bar=False), masked)[0]
    return sorted(in_cat, key=lambda i: -sims[i])[:k]



def select_few_shot_examples( question: str, training_data: list[dict], k: int = 8) -> str:
    top_k = select_top_k(question, training_data, k)
    return "\n\n".join([_format_training_example(ex) for ex in top_k])


if __name__ == "__main__":
    file_path = "C:\\Users\\pcsc\\Documents\\ReposΜεταπτυχιακο\\kgqa_baseline\\data\\en_no_mentions_train.json"

    training_data = load_training_file(file_path, language="en")
    question = "Which countries have won a FIFA World Cup??"

    few_shot_prompt = select_few_shot_examples(question, training_data, k=8)

    print("--- Few-Shot Prompt ---\n")
    print(few_shot_prompt)