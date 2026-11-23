

import os
import sys
from dotenv import load_dotenv

load_dotenv()

TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH", "data/en_no_mentions_train.json")
FEW_SHOT_K = int(os.getenv("FEW_SHOT_K", "3"))
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1") 
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-coder:14b")#llama3.1:8b-instruct-q8_0

ENDPOINTS = {"endp1": "https://backend.wikikgqa.skynet.coypu.org",  # διαγωνισμός
    "endp2": "http://155.207.113.40:7001",
}
DEFAULT_ENDPOINT_NAME = "endp1"



def resolve_endpoint(name_or_url: str) -> tuple[str, str]:
    """('endp2' | 'http://...') -> (όνομα, url)."""
    value = (name_or_url or "").strip()
    if value in ENDPOINTS:
        return value, ENDPOINTS[value]
    if value.startswith(("http://", "https://")):
        return "custom", value
    raise ValueError(f"Άγνωστο endpoint '{value}'. Διαθέσιμα: {', '.join(ENDPOINTS)} ή πλήρες URL.")


def set_endpoint(name_or_url: str) -> str:
    global ENDPOINT_NAME, WIKIKGQA_ENDPOINT, WIKIDATA_SPARQL_ENDPOINT
    ENDPOINT_NAME, WIKIKGQA_ENDPOINT = resolve_endpoint(name_or_url)
    WIKIDATA_SPARQL_ENDPOINT = WIKIKGQA_ENDPOINT
    return WIKIKGQA_ENDPOINT


def apply_cli_endpoint(argv=None) -> None:
  
    argv = sys.argv if argv is None else argv
    for i, arg in enumerate(list(argv)):
        if arg == "--endpoint" and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i:i + 2]
            break
        if arg.startswith("--endpoint="):
            value = arg.split("=", 1)[1]
            del argv[i]
            break
    else:
        return
    set_endpoint(value)
    print(f"[config] endpoint: {ENDPOINT_NAME} -> {WIKIKGQA_ENDPOINT}")



ENDPOINT_NAME, WIKIKGQA_ENDPOINT = resolve_endpoint(
    os.getenv("KGQA_ENDPOINT") or os.getenv("WIKIKGQA_ENDPOINT") or DEFAULT_ENDPOINT_NAME
)

WIKIDATA_SPARQL_ENDPOINT = WIKIKGQA_ENDPOINT


USER_AGENT = "KGQA-Thesis-Baseline/0.1 (student research project)"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR", os.path.join(_ROOT, "indexes", "out_dir"))
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", os.path.join(_ROOT, "indexes", "index.faiss"))
SQLITE_PATH = os.getenv("SQLITE_PATH", os.path.join(_ROOT, "indexes", "entities.sqlite"))