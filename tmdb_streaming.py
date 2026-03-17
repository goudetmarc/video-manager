"""
TMDB Streaming Catalogs Service.
Uses TMDB discover API with watch_providers filter.
Pour les séries : tri par date de la dernière saison (last_air_date).
"""
import asyncio
import json
from pathlib import Path

import httpx
from typing import Any

TMDB_BASE = "https://api.themoviedb.org/3"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

# TMDB watch provider IDs for France
PROVIDERS: dict[str, int] = {
    "netflix": 8,
    "prime": 119,   # Amazon Prime Video
    "disney": 337,  # Disney+
    "apple": 350,   # Apple TV+
}


def _get_api_key() -> str:
    if not SETTINGS_FILE.exists():
        raise ValueError("Clé API TMDB requise. Paramètres → Clé API TMDB (v3).")
    data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    key = (data.get("tmdb_api_key") or "").strip()
    if not key:
        raise ValueError("Clé API TMDB requise. Paramètres → Clé API TMDB (v3).")
    return key


async def _fetch_tv_last_air_date(client: httpx.AsyncClient, tv_id: int, api_key: str) -> tuple[str | None, str | None]:
    """Récupère last_air_date et first_air_date pour une série via l'API TV details."""
    try:
        r = await client.get(
            f"{TMDB_BASE}/tv/{tv_id}",
            params={"api_key": api_key, "language": "fr-FR"},
        )
        if r.status_code == 200:
            data = r.json()
            last = data.get("last_air_date") or data.get("first_air_date")
            first = data.get("first_air_date")
            return (last, first)
    except Exception:
        pass
    return (None, None)


async def enrich_tv_with_last_air_date(
    results: list[dict[str, Any]], api_key: str, client: httpx.AsyncClient
) -> None:
    """Enrichit les résultats TV avec last_air_date (appels parallèles par lots de 20)."""
    batch_size = 20
    for i in range(0, len(results), batch_size):
        batch = results[i : i + batch_size]
        tasks = [_fetch_tv_last_air_date(client, r["id"], api_key) for r in batch]
        dates_list = await asyncio.gather(*tasks)
        for r, (last, _first) in zip(batch, dates_list):
            sort_date = last or r.get("first_air_date") or ""
            r["_sort_date"] = sort_date
            if last:
                r["last_air_date"] = last  # pour affichage frontend


async def get_platform_catalog(
    platform: str,
    content_type: str = "movie",
    region: str = "FR",
    page: int = 1,
    per_page: int = 100,
) -> dict[str, Any]:
    """
    Récupère catalogue via TMDB discover + watch_providers.
    TMDB retourne 20/page; on agrège pour atteindre per_page (défaut 100).
    page=1 → TMDB pages 1-5, page=2 → 6-10, etc.
    """
    provider_id = PROVIDERS.get(platform.lower())
    if not provider_id:
        raise ValueError(f"Platform {platform} not supported")

    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/discover/{content_type}"
    # Films : tri par date de sortie. Séries : on récupère puis tri par last_air_date
    sort_by = "release_date.desc" if content_type == "movie" else "popularity.desc"
    pages_to_fetch = max(1, (per_page + 19) // 20)
    tmdb_page_start = (page - 1) * pages_to_fetch + 1

    all_results: list[dict[str, Any]] = []
    total_pages = 1
    total_results = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for i in range(pages_to_fetch):
            tmdb_page = tmdb_page_start + i
            params = {
                "api_key": api_key,
                "watch_region": region,
                "with_watch_providers": provider_id,
                "sort_by": sort_by,
                "page": tmdb_page,
                "language": "fr-FR",
            }
            r = await client.get(endpoint, params=params)
            r.raise_for_status()
            data = r.json()
            results = data.get("results") or []
            all_results.extend(results)
            total_pages = data.get("total_pages", 1)
            total_results = data.get("total_results", 0)
            if len(results) < 20:
                break

        # Séries : enrichir avec last_air_date et trier par date dernière saison
        if content_type == "tv" and all_results:
            await enrich_tv_with_last_air_date(all_results, api_key, client)
            all_results.sort(key=lambda x: x.get("_sort_date", "") or "", reverse=True)
            for r in all_results:
                r.pop("_sort_date", None)

    our_total_pages = max(1, (total_results + per_page - 1) // per_page) if total_results else 1
    return {
        "results": all_results[:per_page],
        "page": page,
        "total_pages": our_total_pages,
        "total_results": total_results,
    }


async def get_trending(
    content_type: str = "all",
    time_window: str = "week",
) -> dict[str, Any]:
    """Contenu tendance toutes plateformes."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/trending/{content_type}/{time_window}"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def search_multi(query: str, page: int = 1) -> dict[str, Any]:
    """Recherche multi-plateformes."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/search/multi"
    params = {
        "api_key": api_key,
        "query": query,
        "page": page,
        "language": "fr-FR",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()
