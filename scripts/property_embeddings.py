
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_tok = _model = None
_cache: dict[str, np.ndarray] = {}


def _encode(texts: list[str], max_len: int = 32) -> np.ndarray:
    global _tok, _model
    import torch
    from transformers import AutoModel, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME).eval()
    with torch.no_grad():
        b = _tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        h = _model(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
        v = (h * m).sum(1) / m.sum(1)
        v = torch.nn.functional.normalize(v, dim=-1)
    return v.numpy().astype(np.float32)


def embed(texts: list[str]) -> np.ndarray:
    
    missing = [t for t in dict.fromkeys(texts) if t not in _cache]
    if missing:
        for t, v in zip(missing, _encode(missing)):
            _cache[t] = v
    return np.stack([_cache[t] for t in texts])


def cosine_scores(mention: str, labels: list[str]) -> np.ndarray:
    
    if not labels:
        return np.zeros(0, dtype=np.float32)
    return embed(labels) @ embed([mention])[0]
