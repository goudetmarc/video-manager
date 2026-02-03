#!/usr/bin/env python3
"""
Test rapide de l'API Video Manager.
Lance d'abord le backend (Lancer Video Manager.command ou uvicorn), puis :
  python scripts/test_api.py
"""
import json
import sys
import urllib.error
import urllib.request

API_BASE = "http://127.0.0.1:8000"


def get(path: str) -> tuple[int, dict | None]:
    try:
        req = urllib.request.Request(f"{API_BASE}{path}")
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode()) if r.length else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            return e.code, json.loads(body) if body else {}
        except json.JSONDecodeError:
            return e.code, {"detail": body or str(e)}
    except urllib.error.URLError as e:
        return 0, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def main() -> int:
    print("Test API Video Manager (127.0.0.1:8000)")
    print("-" * 40)

    # 1. Health
    code, data = get("/api/health")
    if code != 200:
        print(f"FAIL /api/health → {code} {data}")
        print("→ Lancez le backend : Lancer Video Manager.command ou uvicorn main:app --host 127.0.0.1 --port 8000")
        return 1
    if data.get("status") != "ok":
        print(f"FAIL /api/health → réponse inattendue: {data}")
        return 1
    print("OK  /api/health")

    # 2. TMDB test (réponse 200 avec ok true/false)
    code, data = get("/api/tmdb/test")
    if code != 200:
        print(f"FAIL /api/tmdb/test → {code} {data}")
        return 1
    ok = data.get("ok", False)
    err = data.get("error", "")
    if ok:
        print("OK  /api/tmdb/test (clé valide)")
    else:
        print(f"OK  /api/tmdb/test (réponse API ok, clé: {err or 'non configurée/invalide'})")

    print("-" * 40)
    print("API OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
