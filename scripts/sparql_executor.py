
import os
import time

import requests

try:
    from settings_run import config
    _USER_AGENT = getattr(config, "USER_AGENT", "kgqa-thesis/0.1")
except Exception:  
    _USER_AGENT = "kgqa-thesis/0.1"

DEFAULT_ENDPOINT = "https://backend.wikikgqa.skynet.coypu.org"


def _cfg(name: str, default=None):
    
    try:
        value = getattr(config, name)
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def _endpoints() -> list[str]:
    endpoints = [_cfg("WIKIKGQA_ENDPOINT", DEFAULT_ENDPOINT)]
    fallback = _cfg("SPARQL_FALLBACK_ENDPOINT")
    if fallback:
        endpoints.append(fallback)
    return endpoints


def _auth_headers() -> dict:
   
    token = _cfg("WIKIKGQA_TOKEN")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _strip_qlever_term(term):

    if term is None:
        return None
    if not isinstance(term, str):
        return str(term)
    if term.startswith("<") and term.endswith(">"):
        return term[1:-1]
    if term.startswith('"'):
        end = term.rfind('"')
        if end > 0:
            return term[1:end]
    return term


def _parse_response(data: dict) -> list[dict]:
    # Standard SPARQL results JSON 
    if "results" in data and "bindings" in data["results"]:
        return [
            {var: val.get("value") for var, val in binding.items()}
            for binding in data["results"]["bindings"]
        ]

    if "selected" in data and "res" in data:
        variables = [v.lstrip("?") for v in data["selected"]]
        rows = []
        for row in data["res"]:
            rows.append(
                {
                    var: _strip_qlever_term(cell)
                    for var, cell in zip(variables, row)
                    if cell is not None
                }
            )
        return rows
    # ASK queries
    if "boolean" in data:
        return [{"boolean": str(data["boolean"]).lower()}]
    return []


def _run_once(endpoint: str, query: str, timeout: int) -> list[dict]:
   
    headers = {
        "Accept": _cfg("WIKIKGQA_ACCEPT", "application/qlever-results+json"),
        "Content-Type": "application/sparql-query",
        "User-Agent": _USER_AGENT,
        **_auth_headers(),
    }
    response = requests.post(
        endpoint,
        data=query.encode("utf-8"),  
        headers=headers,
        timeout=timeout,
    )
    response.raise_for_status()
    return _parse_response(response.json())


def _syntax_error_message(response) -> str:
    
    try:
        data = response.json()
        message = data.get("exception")
        if message:
            return str(message)[:400]
    except Exception:
        pass
    return (response.text or "").strip()[:400]


def run_sparql_checked(
    query: str,
    timeout: int = 60,
    max_retries: int = 2,
) -> tuple[list[dict], str | None]:
    
    last_error = None
    syntax_error = None

    for endpoint in _endpoints():
        for attempt in range(max_retries):
            try:
                return _run_once(endpoint, query, timeout), None
            except requests.HTTPError as e:
                last_error = e
                status = e.response.status_code if e.response is not None else None
                body = ""
                if e.response is not None:
                    body = (e.response.text or "").strip()[:800]
                print(f"[sparql_executor] HTTP {status} στο {endpoint}: {e}")
                if body:
                    print(f"[sparql_executor] Απάντηση server: {body}")
                if status == 400 and e.response is not None:
                    syntax_error = _syntax_error_message(e.response)
                if status in (400, 401, 403):
                    break  # λάθος query ή credentials, το retry δεν βοηθάει
                time.sleep(2 * (attempt + 1))
            except (requests.Timeout, requests.ConnectionError) as e:
                last_error = e
                print(
                    f"[sparql_executor] Timeout/σύνδεση στο {endpoint} "
                    f"(προσπάθεια {attempt + 1}/{max_retries}): {e}"
                )
                time.sleep(2 * (attempt + 1))
            except Exception as e:
                last_error = e
                print(f"[sparql_executor] Σφάλμα στο {endpoint}: {e}")
                break

    print(f"[sparql_executor] Τελική αποτυχία. Τελευταίο σφάλμα: {last_error}")
    return [], syntax_error


def run_sparql(
    query: str,
    timeout: int = 60,
    max_retries: int = 2,
    prefer_fallback_first: bool = False,  # κρατιέται για συμβατότητα, δεν χρησιμοποιείται
) -> list[dict]:
    
    return run_sparql_checked(query, timeout=timeout, max_retries=max_retries)[0]


if __name__ == "__main__":
   
    print(run_sparql("SELECT * WHERE { ?s ?p ?o } LIMIT 1"))