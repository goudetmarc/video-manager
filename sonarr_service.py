"""
Service Sonarr - Intégration pour auto-téléchargement des séries depuis la wishlist.

Comportement quand on envoie une série à Sonarr depuis la wishlist :
- La série est ajoutée à Sonarr avec TOUTES les saisons monitorées
- searchForMissingEpisodes=True : Sonarr lance immédiatement la recherche de
  TOUS les épisodes manquants (toutes saisons confondues)
- La qualité dépend du profil Sonarr (quality_profile_id, par défaut 1).
  Configurez dans Sonarr le profil "meilleure qualité" souhaité (HD 1080p, 4K, etc.)
- Sonarr télécharge via les indexeurs configurés (NZBGet, etc.)
"""
import os
from typing import Any

import httpx


def _get_sonarr_config() -> tuple[str, str]:
    url = os.getenv("SONARR_URL", "http://localhost:8989").rstrip("/")
    api_key = (os.getenv("SONARR_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("SONARR_API_KEY doit être défini dans .env")
    return url, api_key


async def test_sonarr_connection() -> bool:
    """Tester la connexion à Sonarr."""
    try:
        url, api_key = _get_sonarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/system/status",
                headers={"X-Api-Key": api_key},
            )
            return r.status_code == 200
    except Exception:
        return False


async def search_series_in_sonarr(tmdb_id: int) -> dict[str, Any] | None:
    """Chercher une série dans Sonarr via TMDB ID. Retourne la série si existante, None sinon."""
    try:
        url, api_key = _get_sonarr_config()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{url}/api/v3/series",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code != 200:
                return None
            series_list = r.json()
            for s in series_list:
                if s.get("tmdbId") == tmdb_id:
                    return s
            return None
    except Exception:
        return None


async def lookup_series_sonarr(tmdb_id: int) -> dict[str, Any] | None:
    """Lookup d'une série via TMDB pour obtenir les métadonnées Sonarr.
    Utilise term=tmdb:ID supporté par Sonarr."""
    try:
        url, api_key = _get_sonarr_config()
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(
                f"{url}/api/v3/series/lookup",
                params={"term": f"tmdb:{tmdb_id}"},
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                results = r.json()
                # Retourne le premier résultat (match exact par tmdb id)
                for s in results:
                    if s.get("tmdbId") == tmdb_id:
                        return s
                return results[0] if results else None
            return None
    except Exception:
        return None


async def _get_default_root_folder() -> str:
    """Récupérer le premier root folder configuré dans Sonarr."""
    try:
        url, api_key = _get_sonarr_config()
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
    return "/tv"


async def add_series_to_sonarr(
    tmdb_id: int,
    title: str,
    quality_profile_id: int = 1,
    root_folder: str | None = None,
    monitored: bool = True,
    search_for_missing_episodes: bool = True,
) -> dict[str, Any]:
    """
    Ajouter une série à Sonarr et lancer la recherche de TOUS les épisodes manquants.

    search_for_missing_episodes=True : Sonarr cherche et télécharge tous les épisodes
    de toutes les saisons dans la meilleure qualité selon le profil.

    Returns:
        dict avec success, already_exists, series, message, error
    """
    # 1. Vérifier si déjà dans Sonarr
    existing = await search_series_in_sonarr(tmdb_id)
    if existing:
        return {
            "success": False,
            "already_exists": True,
            "series": existing,
            "message": f"La série '{title}' est déjà dans Sonarr",
        }

    # 2. Lookup pour métadonnées
    series_data = await lookup_series_sonarr(tmdb_id)
    if not series_data:
        return {
            "success": False,
            "error": "Série introuvable dans TMDB via Sonarr",
            "message": f"Impossible de trouver '{title}' dans la base Sonarr",
        }

    # 3. Root folder
    if not root_folder:
        root_folder = await _get_default_root_folder()

    # 4. Préparer payload - Sonarr attend un objet SeriesResource
    seasons = series_data.get("seasons", [])
    # S'assurer que toutes les saisons sont monitorées pour le téléchargement
    if search_for_missing_episodes and seasons:
        seasons = [
            {**s, "monitored": True} if isinstance(s, dict) else s
            for s in seasons
        ]

    payload = {
        "title": series_data.get("title", title),
        "tmdbId": tmdb_id,
        "tvdbId": series_data.get("tvdbId"),
        "qualityProfileId": quality_profile_id,
        "rootFolderPath": root_folder,
        "monitored": monitored,
        "addOptions": {
            "searchForMissingEpisodes": search_for_missing_episodes,
            "searchForCutoffUnmetEpisodes": False,
        },
        "year": series_data.get("year"),
        "images": series_data.get("images", []),
        "titleSlug": series_data.get("titleSlug"),
        "seasons": seasons,
    }

    # 5. Ajouter à Sonarr
    try:
        url, api_key = _get_sonarr_config()
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(
                f"{url}/api/v3/series",
                json=payload,
                headers={"X-Api-Key": api_key},
            )
            if r.status_code in (200, 201):
                series = r.json()
                return {
                    "success": True,
                    "series": series,
                    "message": f"Série '{title}' ajoutée à Sonarr. Recherche de tous les épisodes en cours...",
                }
            return {
                "success": False,
                "error": r.text,
                "message": f"Erreur Sonarr: {r.text}",
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Erreur lors de l'ajout: {str(e)}",
        }


async def get_sonarr_quality_profiles() -> list[dict[str, Any]]:
    """Récupérer les profils qualité disponibles."""
    try:
        url, api_key = _get_sonarr_config()
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


async def get_sonarr_series_status(sonarr_series_id: int) -> dict[str, Any] | None:
    """Obtenir le statut d'une série dans Sonarr (épisodes téléchargés, etc.)."""
    try:
        url, api_key = _get_sonarr_config()
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(
                f"{url}/api/v3/series/{sonarr_series_id}",
                headers={"X-Api-Key": api_key},
            )
            if r.status_code == 200:
                return r.json()
    except Exception:
        pass
    return None
