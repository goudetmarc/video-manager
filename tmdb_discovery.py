"""
TMDB Discovery Service - Exploration par connexions.
Nouveautés semaine, réalisateurs, acteurs, genres/époques, collections.
Pour les séries : tri par date de la dernière saison (last_air_date).
"""
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

from tmdb_streaming import enrich_tv_with_last_air_date

TMDB_BASE = "https://api.themoviedb.org/3"
SETTINGS_FILE = Path(__file__).resolve().parent / "settings.json"

PROVIDERS: dict[str, int] = {
    "netflix": 8,
    "prime": 119,
    "disney": 337,
    "apple": 350,
}

POPULAR_COLLECTIONS = [
    {"id": 86311, "name": "The Avengers"},
    {"id": 535313, "name": "Godzilla"},
    {"id": 2150, "name": "Astérix"},
    {"id": 263, "name": "The Dark Knight"},
    {"id": 556, "name": "Spider-Man"},
    {"id": 9485, "name": "Fast & Furious"},
    {"id": 8091, "name": "Alien"},
    {"id": 1241, "name": "Harry Potter"},
    {"id": 748, "name": "X-Men"},
    {"id": 529892, "name": "James Bond"},
    {"id": "studio_ghibli", "name": "Studio Ghibli", "company_id": 10342},
]


def _get_api_key() -> str:
    if not SETTINGS_FILE.exists():
        raise ValueError("Clé API TMDB requise. Paramètres → Clé API TMDB (v3).")
    data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    key = (data.get("tmdb_api_key") or "").strip()
    if not key:
        raise ValueError("Clé API TMDB requise. Paramètres → Clé API TMDB (v3).")
    return key


async def get_weekly_new(
    platform: str,
    content_type: str = "movie",
    days_back: int = 7,
    per_page: int = 100,
) -> dict[str, Any]:
    """Films/séries sortis dans les X derniers jours ET disponibles sur la plateforme.
    Films : primary_release_date. Séries : air_date (épisodes récents) puis tri par last_air_date.
    TMDB retourne 20/page, on agrège plusieurs pages pour atteindre per_page (défaut 100)."""
    provider_id = PROVIDERS.get(platform.lower())
    if not provider_id:
        raise ValueError(f"Platform {platform} not supported")

    date_from = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    date_to = datetime.now().strftime("%Y-%m-%d")
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/discover/{content_type}"
    sort_by = "primary_release_date.desc" if content_type == "movie" else "popularity.desc"

    base_params: dict[str, Any] = {
        "api_key": api_key,
        "watch_region": "FR",
        "with_watch_providers": provider_id,
        "sort_by": sort_by,
        "language": "fr-FR",
    }
    if content_type == "movie":
        base_params["primary_release_date.gte"] = date_from
    else:
        # Séries : épisodes diffusés dans la fenêtre (air_date = date des épisodes)
        base_params["air_date.gte"] = date_from
        base_params["air_date.lte"] = date_to

    pages_to_fetch = max(1, (per_page + 19) // 20)
    all_results: list[dict[str, Any]] = []
    total_pages = 1
    total_results = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for page in range(1, pages_to_fetch + 1):
            params = {**base_params, "page": page}
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

    return {
        "results": all_results[:per_page],
        "page": 1,
        "total_pages": total_pages,
        "total_results": total_results,
    }


async def search_person(query: str) -> dict[str, Any]:
    """Chercher un réalisateur ou acteur."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/search/person"
    params = {"api_key": api_key, "query": query, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def get_person_credits(person_id: int, role: str = "all") -> dict[str, Any]:
    """Filmographie d'une personne (filtrable par rôle)."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/person/{person_id}/combined_credits"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        data = r.json()

    if role == "directing":
        data["cast"] = []
        data["crew"] = [c for c in data.get("crew", []) if c.get("job") == "Director"]
    elif role == "acting":
        data["crew"] = []
    elif role == "writing":
        data["cast"] = []
        data["crew"] = [c for c in data.get("crew", []) if c.get("department") == "Writing"]

    return data


async def discover_by_genre_era(
    genre_ids: list[int],
    year_from: int | None = None,
    year_to: int | None = None,
    content_type: str = "movie",
    per_page: int = 100,
) -> dict[str, Any]:
    """Découvrir par genre + époque (ex: Thrillers années 80). Retourne per_page items (défaut 100).
    Séries : tri par date de la dernière saison (last_air_date)."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/discover/{content_type}"
    base_params: dict[str, Any] = {
        "api_key": api_key,
        "with_genres": ",".join(map(str, genre_ids)),
        "sort_by": "vote_average.desc" if content_type == "movie" else "popularity.desc",
        "vote_count.gte": 100,
        "language": "fr-FR",
    }
    if content_type == "movie":
        if year_from:
            base_params["primary_release_date.gte"] = f"{year_from}-01-01"
        if year_to:
            base_params["primary_release_date.lte"] = f"{year_to}-12-31"
    else:
        if year_from:
            base_params["first_air_date.gte"] = f"{year_from}-01-01"
        if year_to:
            base_params["first_air_date.lte"] = f"{year_to}-12-31"

    pages_to_fetch = max(1, (per_page + 19) // 20)
    all_results: list[dict[str, Any]] = []
    total_pages = 1
    total_results = 0

    async with httpx.AsyncClient(timeout=15.0) as client:
        for page in range(1, pages_to_fetch + 1):
            params = {**base_params, "page": page}
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

    return {
        "results": all_results[:per_page],
        "page": 1,
        "total_pages": total_pages,
        "total_results": total_results,
    }


async def get_genres(content_type: str = "movie") -> dict[str, Any]:
    """Liste des genres disponibles."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/genre/{content_type}/list"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def search_collection(query: str) -> dict[str, Any]:
    """Chercher une collection."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/search/collection"
    params = {"api_key": api_key, "query": query, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def get_collection_details(collection_id: int) -> dict[str, Any]:
    """Détails d'une collection (tous les films)."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/collection/{collection_id}"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


def get_popular_collections() -> list[dict[str, Any]]:
    """Collections populaires pré-définies."""
    return POPULAR_COLLECTIONS


async def get_studio_films(company_id: int) -> dict[str, Any]:
    """Films d'un studio (ex: Ghibli = 10342)."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/discover/movie"
    params = {
        "api_key": api_key,
        "with_companies": company_id,
        "sort_by": "primary_release_date.desc",
        "language": "fr-FR",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def get_similar(item_id: int, content_type: str = "movie") -> dict[str, Any]:
    """Films similaires."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/{content_type}/{item_id}/similar"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()


async def get_recommendations(item_id: int, content_type: str = "movie") -> dict[str, Any]:
    """Recommandations basées sur un film."""
    api_key = _get_api_key()
    endpoint = f"{TMDB_BASE}/{content_type}/{item_id}/recommendations"
    params = {"api_key": api_key, "language": "fr-FR"}

    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(endpoint, params=params)
        r.raise_for_status()
        return r.json()
