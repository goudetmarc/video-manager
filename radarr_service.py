"""
Service Radarr - Intégration pour auto-téléchargement depuis la wishlist.
Envoie les films à Radarr qui lance la recherche et le téléchargement via NZBGet.
"""
import os
from typing import Any

import httpx


def _get_radarr_config() -> tuple[str, str]:
    url = os.getenv("RADARR_URL", "http://localhost:7878").rstrip("/")
    api_key = (os.getenv("RADARR_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("RADARR_API_KEY doit être défini dans .env")
    return url, api_key


async def test_radarr_connection() -> bool:
    """Tester la connexion à Radarr."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/system/status",
                headers={"X-Api-Key": api_key},
            )
            return r.status_code == 200
    except Exception:
        return False


async def search_movie_in_radarr(tmdb_id: int) -> dict[str, Any] | None:
    """Chercher un film dans Radarr via TMDB ID. Retourne le film si existant, None sinon."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{url}/api/v3/movie",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code != 200:
                return None
            movies = r.json()
            for m in movies:
                if m.get("tmdbId") == tmdb_id:
                    return m
            return None
    except Exception:
        return None


async def lookup_movie_radarr(tmdb_id: int) -> dict[str, Any] | None:
    """Lookup d'un film via TMDB pour obtenir les métadonnées Radarr."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{url}/api/v3/movie/lookup/tmdb",
                params={"tmdbId": tmdb_id},
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                return r.json()
            return None
    except Exception:
        return None


async def _get_default_root_folder() -> str:
    """Récupérer le premier root folder configuré dans Radarr."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/rootfolder",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                folders = r.json()
                if folders:
                    return folders[0]["path"]
    except Exception:
        pass
    return "/movies"


async def add_movie_to_radarr(
    tmdb_id: int,
    title: str,
    quality_profile_id: int = 1,
    root_folder: str | None = None,
    monitored: bool = True,
    search_for_movie: bool = True,
) -> dict[str, Any]:
    """
    Ajouter un film à Radarr et lancer la recherche.

    Returns:
        dict avec success, already_exists, movie, message, error
    """
    # 1. Vérifier si déjà dans Radarr
    existing = await search_movie_in_radarr(tmdb_id)
    if existing:
        return {
            "success": False,
            "already_exists": True,
            "movie": existing,
            "message": f"Le film '{title}' est déjà dans Radarr",
        }

    # 2. Lookup pour métadonnées
    movie_data = await lookup_movie_radarr(tmdb_id)
    if not movie_data:
        return {
            "success": False,
            "error": "Film introuvable dans TMDB via Radarr",
            "message": f"Impossible de trouver '{title}' dans la base Radarr",
        }

    # 3. Root folder
    if not root_folder:
        root_folder = await _get_default_root_folder()

    # 4. Payload
    payload = {
        "title": movie_data.get("title", title),
        "tmdbId": tmdb_id,
        "qualityProfileId": quality_profile_id,
        "rootFolderPath": root_folder,
        "monitored": monitored,
        "addOptions": {"searchForMovie": search_for_movie},
        "year": movie_data.get("year"),
        "images": movie_data.get("images", []),
        "titleSlug": movie_data.get("titleSlug"),
    }

    # 5. Ajouter à Radarr
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{url}/api/v3/movie",
                json=payload,
                headers={"X-Api-Key": api_key},
            )
            if r.status_code in (200, 201):
                movie = r.json()
                return {
                    "success": True,
                    "movie": movie,
                    "message": f"Film '{title}' ajouté à Radarr. Recherche en cours...",
                }
            return {
                "success": False,
                "error": r.text,
                "message": f"Erreur Radarr: {r.text}",
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Erreur lors de l'ajout: {str(e)}",
        }


async def get_radarr_quality_profiles() -> list[dict[str, Any]]:
    """Récupérer les profils qualité disponibles."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/qualityprofile",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return []


async def get_radarr_movie_status(radarr_movie_id: int) -> dict[str, Any] | None:
    """Obtenir le statut d'un film dans Radarr."""
    try:
        url, api_key = _get_radarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/movie/{radarr_movie_id}",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None
