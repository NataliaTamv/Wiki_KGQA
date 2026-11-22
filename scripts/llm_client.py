from openai import OpenAI
from settings_run import config

_client = OpenAI(base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY)


def call_llm(prompt: str, system: str = "", max_tokens: int = 512,
             stop: list[str] | None = None, temperature: float = 0.0) -> str:


    messages = []
          
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    try:
        response = _client.chat.completions.create(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            messages=messages,
            stop=stop,
            temperature=temperature,
        )
    except Exception as e:
        raise RuntimeError(
            f"Αποτυχία κλήσης στο LLM ({config.LLM_BASE_URL}, model={config.LLM_MODEL}). "
            f"Έλεγξε ότι τρέχει `ollama serve` και ότι έχεις κάνει `ollama pull {config.LLM_MODEL}`. "
            f"Αρχικό σφάλμα: {e}"
        )

    return response.choices[0].message.content or ""


# 
import time as _time

STATS = {"calls": 0, "seconds": 0.0}
_raw_call_llm = call_llm


def call_llm(*args, **kwargs):
    t = _time.perf_counter()
    try:
        return _raw_call_llm(*args, **kwargs)
    finally:
        STATS["calls"] += 1
        STATS["seconds"] += _time.perf_counter() - t
