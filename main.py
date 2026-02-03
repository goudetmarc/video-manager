"""
Video Manager: inventaire persistant des vidéos (scan une fois, tout enregistré).
Supports Supabase for cloud storage with fallback to local JSON files.
"""
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Body, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from scanner import (
    scan_and_build_inventory,
    scan_and_build_inventory_stream,
    scan_raw_stream,
    build_inventory_from_raw_stream,
    get_folder_diagnostic,
    get_video_files_count,
    detect_media_type,
)
from normalizer import get_normalizer
from cache_service import cache, inventory_cache, series_cache
from offline_normalizer import get_offline_normalizer

# =============================================================================
# Supabase Client Configuration
# =============================================================================
# Set SUPABASE_URL and SUPABASE_KEY environment variables to enable cloud storage.
# Without these, the app falls back to local JSON files.

_supabase_client = None

def get_supabase_client():
    """Get or create the Supabase client (lazy initialization)."""
    global _supabase_client
    if _supabase_client is not None:
        return _supabase_client

    # Use settings from config.py (loads from .env file)
    supabase_url = settings.supabase_url.strip() if settings.supabase_url else ""
    supabase_key = settings.supabase_key.strip() if settings.supabase_key else ""

    if not supabase_url or not supabase_key:
        return None

    try:
        from supabase import create_client, Client
        _supabase_client = create_client(supabase_url, supabase_key)
        print(f"[SUPABASE] Connected to {supabase_url}")
        return _supabase_client
    except ImportError:
        print("[SUPABASE] supabase-py not installed. Using local storage.")
        return None
    except Exception as e:
        print(f"[SUPABASE] Connection error: {e}. Using local storage.")
        return None


def is_supabase_enabled() -> bool:
    """Check if Supabase is configured and available."""
    return get_supabase_client() is not None

app = FastAPI(title="Video Manager", description="Inventaire vidéo persistant")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
POSTERS_DIR = STATIC_DIR / "posters"
SETTINGS_FILE = BASE_DIR / "settings.json"
INVENTORY_FILE = BASE_DIR / "inventory.json"
RAW_INVENTORY_FILE = BASE_DIR / "raw_inventory.json"
POSTER_CACHE_FILE = BASE_DIR / "poster_cache.json"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
STATIC_DIR.mkdir(exist_ok=True)
POSTERS_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Toute exception non gérée → 500 avec JSON (pas de page HTML)."""
    return JSONResponse(
        status_code=500,
        content={"ok": False, "error": str(exc), "detail": str(exc)},
    )


def load_settings() -> dict:
    if not SETTINGS_FILE.exists():
        return {
            "video_path": getattr(settings, "video_root", "") or "",
            "series_path": "",
            "tmdb_api_key": "",
            "tvdb_api_key": "",
            "scan_mode": "direct",
            "nzb_api_key": "",
            "nzb_api_user": "",
        }
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        return {
            "video_path": data.get("video_path", ""),
            "series_path": data.get("series_path", ""),
            "tmdb_api_key": data.get("tmdb_api_key", ""),
            "tvdb_api_key": data.get("tvdb_api_key", ""),
            "scan_mode": data.get("scan_mode", "direct"),
            "nzb_api_key": data.get("nzb_api_key", ""),
            "nzb_api_user": data.get("nzb_api_user", ""),
        }
    except Exception:
        return {"video_path": "", "series_path": "", "tmdb_api_key": "", "tvdb_api_key": "", "scan_mode": "direct", "nzb_api_key": "", "nzb_api_user": ""}


def save_settings(
    video_path: str | None = None,
    series_path: str | None = None,
    tmdb_api_key: str | None = None,
    tvdb_api_key: str | None = None,
    scan_mode: str | None = None,
    nzb_api_key: str | None = None,
    nzb_api_user: str | None = None,
) -> None:
    current = load_settings()
    if video_path is not None:
        current["video_path"] = video_path
    if series_path is not None:
        current["series_path"] = series_path
    if tmdb_api_key is not None:
        current["tmdb_api_key"] = tmdb_api_key
    if tvdb_api_key is not None:
        current["tvdb_api_key"] = tvdb_api_key
    if scan_mode is not None:
        current["scan_mode"] = scan_mode if scan_mode in ("direct", "async") else "direct"
    if nzb_api_key is not None:
        current["nzb_api_key"] = nzb_api_key
    if nzb_api_user is not None:
        current["nzb_api_user"] = nzb_api_user
    SETTINGS_FILE.write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")


def _load_inventory_local() -> dict:
    """Charge l'inventaire depuis le fichier JSON local."""
    if not INVENTORY_FILE.exists():
        return {"scanned_path": None, "scanned_at": None, "files": []}
    try:
        data = json.loads(INVENTORY_FILE.read_text(encoding="utf-8"))
        files = data.get("files", [])
        # Calculate media_type using TMDB data + file patterns
        for f in files:
            path_str = f.get("path", "")
            if path_str:
                # Get series info from filename patterns
                file_info = detect_media_type(Path(path_str))
                f["season"] = file_info["season"]
                f["episode"] = file_info["episode"]
                f["series_name"] = file_info["series_name"]
                # Calculate media_type: Documentary first, then series, then movie
                tmdb_media_type = f.get("tmdb_media_type")
                genre_ids = f.get("genre_ids") or []
                if 99 in genre_ids:  # Documentary genre takes priority (even for docuseries)
                    f["media_type"] = "documentary"
                elif tmdb_media_type == "tv" or file_info["media_type"] == "series":
                    f["media_type"] = "series"
                else:
                    f["media_type"] = "movie"
            else:
                f["media_type"] = "movie"
        return {
            "scanned_path": data.get("scanned_path"),
            "scanned_at": data.get("scanned_at"),
            "files": files,
        }
    except Exception:
        return {"scanned_path": None, "scanned_at": None, "files": []}


def _load_inventory_supabase() -> dict:
    """Charge l'inventaire depuis Supabase."""
    client = get_supabase_client()
    if not client:
        return _load_inventory_local()

    try:
        # Get scan metadata
        meta_response = client.table("scan_metadata").select("*").order("scanned_at", desc=True).limit(1).execute()
        meta = meta_response.data[0] if meta_response.data else {}

        # Get all movies with pagination (Supabase server limit is 1000)
        all_movies = []
        page_size = 1000
        offset = 0
        while True:
            movies_response = client.table("movies").select("*").range(offset, offset + page_size - 1).execute()
            batch = movies_response.data or []
            if not batch:
                break
            all_movies.extend(batch)
            if len(batch) < page_size:
                break
            offset += page_size

        files = []
        for row in all_movies:
            # Convert Supabase row to frontend FileRow format
            # Map database column names to frontend expected names
            file_data = {
                "path": row.get("path"),
                "name": row.get("name"),
                "size_bytes": row.get("size_bytes"),
                "mtime": row.get("mtime"),
                # Frontend expects duration_sec, database has duration
                "duration_sec": row.get("duration"),
                "width": row.get("width"),
                "height": row.get("height"),
                # Frontend expects video_codec, database has codec
                "video_codec": row.get("codec"),
                "video_profile": row.get("video_profile") or row.get("codec_profile"),
                "audio_codec": row.get("audio_codec"),
                "audio_channels": row.get("audio_channels"),
                # Frontend expects bit_rate, database has bitrate
                "bit_rate": row.get("bitrate"),
                "video_bit_rate": row.get("video_bitrate"),
                "audio_bit_rate": row.get("audio_bitrate"),
                "fps": row.get("fps"),
                "hdr": row.get("hdr"),
                # Frontend expects format_name, database has container
                "format_name": row.get("container"),
                "audio_languages": row.get("audio_languages"),
                "metadata_title": row.get("metadata_title"),
                "custom_group_key": row.get("custom_group_key"),
                # ===== TOUS les champs TMDB =====
                "tmdb_id": row.get("tmdb_id"),
                "tmdb_title": row.get("tmdb_title"),
                "tmdb_year": row.get("tmdb_year"),
                "tmdb_media_type": row.get("tmdb_media_type"),
                "genre_ids": row.get("genre_ids"),
                "tmdb_original_title": row.get("tmdb_original_title"),
                "tmdb_overview": row.get("tmdb_overview"),
                "tmdb_backdrop_path": row.get("tmdb_backdrop_path"),
                "tmdb_popularity": row.get("tmdb_popularity"),
                "tmdb_vote_average": row.get("tmdb_vote_average"),
                "tmdb_vote_count": row.get("tmdb_vote_count"),
                "tmdb_original_language": row.get("tmdb_original_language"),
                "tmdb_adult": row.get("tmdb_adult"),
                "tmdb_release_date": row.get("tmdb_release_date"),
                "tmdb_origin_country": row.get("tmdb_origin_country"),
                "poster_url": row.get("poster_url"),
                # ===== Champs TVDB =====
                "tvdb_id": row.get("tvdb_id"),
                "tvdb_name": row.get("tvdb_name"),
                "tvdb_year": row.get("tvdb_year"),
            }
            # Remove None values to match local format
            file_data = {k: v for k, v in file_data.items() if v is not None}
            # Calculate media_type using TMDB data + file patterns
            path_str = file_data.get("path", "")
            if path_str:
                # Get series info from filename patterns
                file_info = detect_media_type(Path(path_str))
                file_data["season"] = file_info["season"]
                file_data["episode"] = file_info["episode"]
                file_data["series_name"] = file_info["series_name"]
                # Calculate media_type: Documentary first, then series, then movie
                tmdb_media_type = file_data.get("tmdb_media_type")
                genre_ids = file_data.get("genre_ids") or []
                if 99 in genre_ids:  # Documentary genre takes priority (even for docuseries)
                    file_data["media_type"] = "documentary"
                elif tmdb_media_type == "tv" or file_info["media_type"] == "series":
                    file_data["media_type"] = "series"
                else:
                    file_data["media_type"] = "movie"
            else:
                file_data["media_type"] = "movie"
            files.append(file_data)

        return {
            "scanned_path": meta.get("scanned_path"),
            "scanned_at": meta.get("scanned_at"),
            "files": files,
        }
    except Exception as e:
        print(f"[SUPABASE] Error loading inventory: {e}. Falling back to local.")
        return _load_inventory_local()


def load_inventory() -> dict:
    """Charge l'inventaire (Supabase si configuré, sinon fichier local)."""
    if is_supabase_enabled():
        return _load_inventory_supabase()
    return _load_inventory_local()


def _save_inventory_local(scanned_path: str, files: list) -> None:
    """Enregistre l'inventaire dans le fichier JSON local."""
    data = {
        "scanned_path": scanned_path,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    INVENTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")


def _save_inventory_supabase(scanned_path: str, files: list) -> None:
    """Enregistre l'inventaire dans Supabase avec batch upsert (une seule requête)."""
    client = get_supabase_client()
    if not client:
        return _save_inventory_local(scanned_path, files)

    try:
        now = datetime.now(timezone.utc).isoformat()

        # Build all rows first for batch upsert
        rows = []
        for file_data in files:
            # Convert audio_languages from "fre, jpn" string to ["fre", "jpn"] array
            audio_langs_raw = file_data.get("audio_languages", "")
            if isinstance(audio_langs_raw, str) and audio_langs_raw:
                audio_languages = [lang.strip() for lang in audio_langs_raw.split(",") if lang.strip() and lang.strip() != "?"]
            elif isinstance(audio_langs_raw, list):
                audio_languages = audio_langs_raw
            else:
                audio_languages = []

            # Extract audio info from first audio track if available
            audio_tracks = file_data.get("audio_tracks", [])
            first_audio = audio_tracks[0] if audio_tracks else {}

            row = {
                "path": file_data.get("path"),
                "name": file_data.get("name"),
                "size_bytes": file_data.get("size_bytes"),
                "mtime": file_data.get("mtime"),
                # Map scan field names to database column names
                "duration": file_data.get("duration_sec") or file_data.get("duration"),
                "width": file_data.get("width"),
                "height": file_data.get("height"),
                "codec": file_data.get("video_codec") or file_data.get("codec"),
                "codec_profile": file_data.get("codec_profile"),
                "audio_codec": first_audio.get("codec") or file_data.get("audio_codec"),
                "audio_channels": first_audio.get("channels") or file_data.get("audio_channels"),
                "bitrate": file_data.get("bit_rate") or file_data.get("bitrate"),
                "fps": file_data.get("fps"),
                "hdr": file_data.get("hdr", False),
                "container": file_data.get("format_name") or file_data.get("container"),
                "video_profile": file_data.get("video_profile"),
                "video_bitrate": file_data.get("video_bit_rate") or file_data.get("video_bitrate"),
                "audio_bitrate": first_audio.get("bit_rate") or file_data.get("audio_bitrate"),
                "audio_languages": audio_languages if audio_languages else None,
                "metadata_title": file_data.get("metadata_title"),
                "custom_group_key": file_data.get("custom_group_key"),
                "scanned_at": now,
                # ===== TOUS les champs TMDB =====
                "tmdb_id": file_data.get("tmdb_id"),
                "tmdb_title": file_data.get("tmdb_title"),
                "tmdb_year": file_data.get("tmdb_year"),
                "tmdb_media_type": file_data.get("tmdb_media_type"),
                "genre_ids": file_data.get("genre_ids"),
                "media_type": file_data.get("media_type"),
                "tmdb_original_title": file_data.get("tmdb_original_title"),
                "tmdb_overview": file_data.get("tmdb_overview"),
                "tmdb_backdrop_path": file_data.get("tmdb_backdrop_path"),
                "tmdb_popularity": file_data.get("tmdb_popularity"),
                "tmdb_vote_average": file_data.get("tmdb_vote_average"),
                "tmdb_vote_count": file_data.get("tmdb_vote_count"),
                "tmdb_original_language": file_data.get("tmdb_original_language"),
                "tmdb_adult": file_data.get("tmdb_adult"),
                "tmdb_release_date": file_data.get("tmdb_release_date"),
                "tmdb_origin_country": file_data.get("tmdb_origin_country"),
                "poster_url": file_data.get("poster_url"),
                # ===== Champs TVDB =====
                "tvdb_id": file_data.get("tvdb_id"),
                "tvdb_name": file_data.get("tvdb_name"),
                "tvdb_year": file_data.get("tvdb_year"),
            }
            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}
            rows.append(row)

        # Batch upsert - one request instead of N requests!
        if rows:
            client.table("movies").upsert(rows, on_conflict="path").execute()

        print(f"[SUPABASE] Saved {len(files)} movies to database (batch)")

        # Also save locally as backup
        _save_inventory_local(scanned_path, files)

    except Exception as e:
        import traceback
        print(f"[SUPABASE] Error saving inventory: {e}")
        print(f"[SUPABASE] Traceback: {traceback.format_exc()}")
        print(f"[SUPABASE] Falling back to local storage.")
        _save_inventory_local(scanned_path, files)


def _delete_files_from_supabase(paths: list) -> int:
    """Supprime des fichiers de Supabase par leur chemin."""
    client = get_supabase_client()
    if not client:
        print(f"[SUPABASE] No client available for delete")
        return 0
    if not paths:
        print(f"[SUPABASE] No paths to delete")
        return 0

    deleted = 0
    print(f"[SUPABASE] Attempting to delete {len(paths)} files: {paths}")

    for path in paths:
        try:
            # Delete one by one to ensure each delete works
            result = client.table("movies").delete().eq("path", path).execute()
            if result.data:
                deleted += len(result.data)
                print(f"[SUPABASE] Deleted: {path} (count: {len(result.data)})")
            else:
                print(f"[SUPABASE] No rows deleted for: {path}")
        except Exception as e:
            print(f"[SUPABASE] Error deleting {path}: {e}")

    print(f"[SUPABASE] Total deleted: {deleted}")
    return deleted


def _update_tmdb_batch_supabase(files_batch: list) -> None:
    """Met à jour TOUS les champs TMDB pour un batch de fichiers (sauvegarde complète)."""
    client = get_supabase_client()
    if not client or not files_batch:
        return

    updated = 0
    for f in files_batch:
        if f.get("tmdb_id") and f.get("path"):
            try:
                # Construire l'objet avec TOUS les champs TMDB disponibles
                update_data = {
                    "tmdb_id": f.get("tmdb_id"),
                    "tmdb_title": f.get("tmdb_title") or "",
                    "tmdb_year": f.get("tmdb_year") or "",
                    "tmdb_media_type": f.get("tmdb_media_type"),
                    "genre_ids": f.get("genre_ids") or [],
                    "media_type": f.get("media_type"),
                    "tmdb_original_title": f.get("tmdb_original_title"),
                    "tmdb_overview": f.get("tmdb_overview"),
                    "tmdb_backdrop_path": f.get("tmdb_backdrop_path"),
                    "tmdb_popularity": f.get("tmdb_popularity"),
                    "tmdb_vote_average": f.get("tmdb_vote_average"),
                    "tmdb_vote_count": f.get("tmdb_vote_count"),
                    "tmdb_original_language": f.get("tmdb_original_language"),
                    "tmdb_adult": f.get("tmdb_adult"),
                    "tmdb_release_date": f.get("tmdb_release_date"),
                    "tmdb_origin_country": f.get("tmdb_origin_country"),
                    "poster_url": f.get("poster_url"),
                }
                # Supprimer les valeurs None pour éviter les erreurs
                update_data = {k: v for k, v in update_data.items() if v is not None}
                client.table("movies").update(update_data).eq("path", f.get("path")).execute()
                updated += 1
            except Exception as e:
                print(f"[SUPABASE] Error updating {f.get('path')}: {e}")
    if updated > 0:
        print(f"[SUPABASE] Updated ALL TMDB fields for {updated} movies")


def save_inventory(scanned_path: str, files: list) -> None:
    """Enregistre l'inventaire (Supabase si configuré, sinon fichier local)."""
    if is_supabase_enabled():
        _save_inventory_supabase(scanned_path, files)
    else:
        _save_inventory_local(scanned_path, files)

    # Invalider le cache après sauvegarde pour forcer le rechargement
    inventory_cache.invalidate()
    series_cache.invalidate()
    print("[CACHE] Invalidated inventory and series caches after save")


def load_raw_inventory() -> dict:
    """Charge l'inventaire brut (liste seule, sans ffprobe ni API)."""
    if not RAW_INVENTORY_FILE.exists():
        return {"scanned_path": None, "scanned_at": None, "files": []}
    try:
        data = json.loads(RAW_INVENTORY_FILE.read_text(encoding="utf-8"))
        return {
            "scanned_path": data.get("scanned_path"),
            "scanned_at": data.get("scanned_at"),
            "files": data.get("files", []),
        }
    except Exception:
        return {"scanned_path": None, "scanned_at": None, "files": []}


def save_raw_inventory(scanned_path: str, files: list) -> None:
    """Enregistre l'inventaire brut (liste de fichiers sans ffprobe)."""
    data = {
        "scanned_path": scanned_path,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }
    RAW_INVENTORY_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=0), encoding="utf-8")

    # Invalider le cache après sauvegarde
    inventory_cache.invalidate()
    series_cache.invalidate()


def _load_poster_cache_local() -> dict:
    """Cache des affiches TMDB depuis fichier local."""
    if not POSTER_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(POSTER_CACHE_FILE.read_text(encoding="utf-8"))
        return data.get("cache", data) if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_poster_cache_supabase() -> dict:
    """Cache des affiches TMDB depuis Supabase."""
    client = get_supabase_client()
    if not client:
        return _load_poster_cache_local()

    try:
        response = client.table("poster_cache").select("*").execute()
        cache = {}
        for row in response.data or []:
            cache_key = row.get("cache_key")
            if cache_key:
                cache[cache_key] = {
                    "poster_path": row.get("poster_path"),
                    "poster_url": row.get("poster_url"),
                    "tmdb_id": row.get("tmdb_id"),
                    "title": row.get("tmdb_title"),
                    "year": row.get("tmdb_year"),
                }
        return cache
    except Exception as e:
        print(f"[SUPABASE] Error loading poster cache: {e}. Falling back to local.")
        return _load_poster_cache_local()


def load_poster_cache() -> dict:
    """Cache des affiches TMDB : clé = "query|year" -> { poster_path, tmdb_id?, title?, year? }."""
    if is_supabase_enabled():
        return _load_poster_cache_supabase()
    return _load_poster_cache_local()


def _save_poster_cache_local(cache: dict) -> None:
    """Enregistre le cache des affiches localement."""
    POSTER_CACHE_FILE.write_text(
        json.dumps({"cache": cache}, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


def _save_poster_cache_supabase(cache: dict) -> None:
    """Enregistre le cache des affiches dans Supabase (batch upsert)."""
    client = get_supabase_client()
    if not client:
        return _save_poster_cache_local(cache)

    try:
        # Build batch of rows
        rows = []
        for cache_key, entry in cache.items():
            if not isinstance(entry, dict):
                continue
            row = {
                "cache_key": cache_key,
                "poster_path": entry.get("poster_path"),
                "poster_url": entry.get("poster_url"),
                "tmdb_id": entry.get("tmdb_id"),
                "tmdb_title": entry.get("title"),
                "tmdb_year": entry.get("year"),
            }
            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}
            rows.append(row)

        # Batch upsert - one request instead of N!
        if rows:
            client.table("poster_cache").upsert(rows, on_conflict="cache_key").execute()

        # Also save locally as backup
        _save_poster_cache_local(cache)

    except Exception as e:
        print(f"[SUPABASE] Error saving poster cache: {e}. Falling back to local.")
        _save_poster_cache_local(cache)


def save_poster_cache(cache: dict) -> None:
    """Enregistre le cache des affiches."""
    if is_supabase_enabled():
        _save_poster_cache_supabase(cache)
    else:
        _save_poster_cache_local(cache)


def _poster_local_path(tmdb_id: int) -> Path:
    """Chemin du fichier affiche stocké localement (tmdb_id.jpg)."""
    return POSTERS_DIR / f"{tmdb_id}.jpg"


# ============================================
# SUPABASE STORAGE FOR POSTERS
# ============================================

POSTERS_BUCKET = "posters"


def _ensure_posters_bucket_exists() -> bool:
    """Crée le bucket 'posters' sur Supabase Storage s'il n'existe pas."""
    client = get_supabase_client()
    if not client:
        return False
    try:
        # Essayer de lister les buckets pour voir si 'posters' existe
        buckets = client.storage.list_buckets()
        bucket_names = [b.name for b in buckets]
        if POSTERS_BUCKET not in bucket_names:
            # Créer le bucket en mode public
            client.storage.create_bucket(POSTERS_BUCKET, options={"public": True})
            print(f"[SUPABASE] Created bucket '{POSTERS_BUCKET}'")
        return True
    except Exception as e:
        print(f"[SUPABASE] Error checking/creating bucket: {e}")
        return False


def _poster_exists_supabase(tmdb_id: int) -> bool:
    """Vérifie si le poster existe dans Supabase Storage."""
    client = get_supabase_client()
    if not client:
        return False
    try:
        # Lister les fichiers pour vérifier l'existence
        result = client.storage.from_(POSTERS_BUCKET).list(path="", options={"search": f"{tmdb_id}.jpg"})
        return any(f.get("name") == f"{tmdb_id}.jpg" for f in result)
    except Exception:
        return False


def _upload_poster_supabase(tmdb_id: int, image_bytes: bytes) -> str | None:
    """Upload un poster vers Supabase Storage. Retourne l'URL publique ou None."""
    client = get_supabase_client()
    if not client:
        return None
    try:
        _ensure_posters_bucket_exists()
        filename = f"{tmdb_id}.jpg"
        # Upsert: remplace si existe déjà
        client.storage.from_(POSTERS_BUCKET).upload(
            filename,
            image_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        # Récupérer l'URL publique
        url = client.storage.from_(POSTERS_BUCKET).get_public_url(filename)
        print(f"[SUPABASE] Uploaded poster {filename}")
        return url
    except Exception as e:
        print(f"[SUPABASE] Error uploading poster {tmdb_id}: {e}")
        return None


def _get_poster_url_supabase(tmdb_id: int) -> str | None:
    """Retourne l'URL publique du poster depuis Supabase Storage."""
    client = get_supabase_client()
    if not client:
        return None
    try:
        return client.storage.from_(POSTERS_BUCKET).get_public_url(f"{tmdb_id}.jpg")
    except Exception:
        return None


def _ensure_poster_downloaded(poster_path: str | None, tmdb_id: int | None, force_refresh: bool = False) -> bool:
    """
    Télécharge l'affiche TMDB et la stocke:
    - Sur Supabase Storage si configuré
    - En local dans static/posters/{tmdb_id}.jpg en fallback

    Ne re-télécharge que si:
    - force_refresh=True
    - Le fichier n'existe pas (ni local ni Supabase)

    Retourne True si le fichier existe, False sinon.
    """
    if not poster_path or not tmdb_id:
        return False

    local = _poster_local_path(tmdb_id)

    # Vérifier si déjà présent (sauf si refresh forcé)
    if not force_refresh:
        # Check local first (plus rapide)
        if local.exists():
            return True
        # Check Supabase si activé
        if is_supabase_enabled() and _poster_exists_supabase(tmdb_id):
            return True

    # Télécharger depuis TMDB
    url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path.startswith("/") else f"{TMDB_IMAGE_BASE}/{poster_path}"
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.get(url)
            if r.status_code == 200 and r.content:
                # Sauvegarder en local (fallback rapide)
                local.write_bytes(r.content)

                # Upload vers Supabase si activé
                if is_supabase_enabled():
                    _upload_poster_supabase(tmdb_id, r.content)

                return True
    except Exception as e:
        print(f"[POSTER] Error downloading {tmdb_id}: {e}")
    return False


def _poster_url_from_entry(entry: dict) -> str | None:
    """
    Retourne l'URL d'affichage du poster:
    1. URL locale (/static/posters/{id}.jpg) si fichier existe (plus rapide)
    2. Supabase Storage URL si configuré
    3. URL TMDB en fallback
    Retourne None si pas d'affiche.
    """
    poster_path = entry.get("poster_path")
    tmdb_id = entry.get("tmdb_id")
    if not poster_path:
        return None

    if tmdb_id:
        # Priorité 1: Fichier local (le plus rapide)
        if _poster_local_path(tmdb_id).exists():
            return f"/static/posters/{tmdb_id}.jpg"

        # Priorité 2: Supabase Storage (si fichier local n'existe pas)
        if is_supabase_enabled():
            supabase_url = _get_poster_url_supabase(tmdb_id)
            if supabase_url:
                return supabase_url

    # Fallback: URL TMDB directe
    return f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path.startswith("/") else f"{TMDB_IMAGE_BASE}/{poster_path}"


def get_ffprobe_path() -> str | None:
    path = shutil.which("ffprobe")
    return path if path else None


@app.get("/")
def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return FileResponse(html_path)
    return HTMLResponse("<p>Fichier index.html introuvable.</p>")


@app.get("/api/settings")
def api_get_settings():
    return load_settings()


@app.post("/api/settings")
def api_save_settings(body: dict = Body(...)):
    path = (body.get("video_path") or "").strip()
    series_path = (body.get("series_path") or "").strip() if "series_path" in body else None
    tmdb_key = body.get("tmdb_api_key")
    tvdb_key = body.get("tvdb_api_key")
    scan_mode = body.get("scan_mode")
    nzb_api_key = body.get("nzb_api_key")
    nzb_api_user = body.get("nzb_api_user")
    if tmdb_key is not None:
        tmdb_key = (tmdb_key or "").strip()
    if tvdb_key is not None:
        tvdb_key = (tvdb_key or "").strip()
    if scan_mode is not None and scan_mode not in ("direct", "async"):
        scan_mode = "direct"
    if nzb_api_key is not None:
        nzb_api_key = (nzb_api_key or "").strip()
    if nzb_api_user is not None:
        nzb_api_user = (nzb_api_user or "").strip()
    save_settings(
        video_path=path or None,
        series_path=series_path,
        tmdb_api_key=tmdb_key,
        tvdb_api_key=tvdb_key,
        scan_mode=scan_mode,
        nzb_api_key=nzb_api_key,
        nzb_api_user=nzb_api_user,
    )
    return load_settings()


# =============================================================================
# System Monitoring API
# =============================================================================

# Cache pour les stats système (évite les appels trop fréquents)
_system_stats_cache = {"data": None, "timestamp": 0}
_transcode_stats = {"active": False, "fps": 0, "speed": 0, "progress": 0, "bitrate": ""}

def update_transcode_stats(fps: float = 0, speed: float = 0, progress: float = 0, bitrate: str = "", active: bool = False):
    """Met à jour les stats de transcodage (appelé par le processus de transcode)."""
    global _transcode_stats
    _transcode_stats = {
        "active": active,
        "fps": fps,
        "speed": speed,
        "progress": progress,
        "bitrate": bitrate,
    }

@app.get("/api/system/stats")
async def get_system_stats():
    """
    Retourne les statistiques système en temps réel.
    CPU, mémoire, GPU (si disponible), et stats de transcodage.
    """
    import psutil
    import subprocess
    import time

    global _system_stats_cache

    # Cache de 500ms pour éviter les appels trop fréquents
    now = time.time()
    if _system_stats_cache["data"] and (now - _system_stats_cache["timestamp"]) < 0.5:
        return _system_stats_cache["data"]

    # CPU
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()

    # Mémoire
    memory = psutil.virtual_memory()

    # Processus FFmpeg actifs + calcul CPU hors Claude
    ffmpeg_processes = []
    claude_cpu = 0
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            name = proc.info['name'].lower() if proc.info['name'] else ''
            cpu = proc.info['cpu_percent'] or 0

            if 'ffmpeg' in name:
                ffmpeg_processes.append({
                    "pid": proc.info['pid'],
                    "cpu_percent": cpu,
                    "memory_percent": proc.info['memory_percent'],
                })

            # Exclure les processus Claude du calcul
            if 'claude' in name:
                claude_cpu += cpu
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    # CPU effectif (hors Claude)
    cpu_effective = max(0, cpu_percent - (claude_cpu / cpu_count)) if cpu_count else cpu_percent

    # GPU - macOS (powermetrics) ou NVIDIA (nvidia-smi)
    gpu_stats = None
    try:
        # Essayer nvidia-smi pour GPU NVIDIA
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 4:
                gpu_stats = {
                    "type": "nvidia",
                    "utilization": float(parts[0]),
                    "memory_used_mb": float(parts[1]),
                    "memory_total_mb": float(parts[2]),
                    "temperature": float(parts[3]),
                }
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Si pas de NVIDIA, détecter le GPU sur macOS via system_profiler
    if not gpu_stats:
        try:
            result = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=3
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                displays = data.get("SPDisplaysDataType", [])
                if displays:
                    gpu_info = displays[0]
                    gpu_name = gpu_info.get("sppci_model", "Apple GPU")
                    vram = gpu_info.get("spdisplays_vram", "")
                    vram_mb = None
                    if vram:
                        # Parse "8 GB" or "8192 MB"
                        parts = vram.split()
                        if len(parts) >= 2:
                            try:
                                val = int(parts[0])
                                if "GB" in parts[1].upper():
                                    vram_mb = val * 1024
                                else:
                                    vram_mb = val
                            except ValueError:
                                pass
                    gpu_stats = {
                        "type": "apple",
                        "name": gpu_name,
                        "utilization": None,
                        "memory_used_mb": None,
                        "memory_total_mb": vram_mb,
                        "available": True,
                    }
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Fallback minimal
            gpu_stats = {
                "type": "apple",
                "utilization": None,
                "memory_used_mb": None,
                "memory_total_mb": None,
                "available": True,
            }

    stats = {
        "timestamp": now,
        "cpu": {
            "percent": cpu_percent,
            "percent_effective": round(cpu_effective, 1),
            "count": cpu_count,
            "freq_mhz": cpu_freq.current if cpu_freq else None,
        },
        "memory": {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "percent": memory.percent,
            "available_gb": round(memory.available / (1024**3), 2),
        },
        "gpu": gpu_stats,
        "ffmpeg_processes": ffmpeg_processes,
        "transcode": _transcode_stats,
    }

    _system_stats_cache = {"data": stats, "timestamp": now}
    return stats


@app.get("/api/inventory")
def api_get_inventory(force_refresh: bool = False):
    """
    Retourne l'inventaire enregistré (sans rescan), avec poster_url Supabase pour chaque fichier.
    Utilise un cache en mémoire pour éviter les rechargements constants.

    Args:
        force_refresh: Si True, ignore le cache et recharge depuis la DB
    """
    cache_key = "inventory:all"

    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = inventory_cache.get(cache_key)
        if cached is not None:
            return cached

    # Load from database
    inv = load_inventory()
    files = inv.get("files", [])

    # Ajouter poster_url Supabase à chaque fichier qui a un tmdb_id
    supabase_url = (settings.supabase_url or "").strip().rstrip("/")
    for f in files:
        tmdb_id = f.get("tmdb_id")
        if tmdb_id and not f.get("poster_url"):
            if supabase_url:
                f["poster_url"] = f"{supabase_url}/storage/v1/object/public/posters/{tmdb_id}.jpg"
            elif _poster_local_path(tmdb_id).exists():
                f["poster_url"] = f"/static/posters/{tmdb_id}.jpg"

    # Cache result for 5 minutes
    inventory_cache.set(cache_key, inv, ttl=300)

    return inv


# =============================================================================
# SERIES API - Proper grouping by tvdb_id/series_name
# =============================================================================

def _get_cached_tmdb_tv_details(tmdb_id: int) -> dict | None:
    """Get TV show details from cache if available."""
    if not tmdb_id:
        return None
    cache = load_tmdb_details_cache()
    cache_key = f"tv_{tmdb_id}"
    cached = cache.get(cache_key)
    if cached and is_cache_valid(cached):
        return cached
    return None


def _get_series_group_key(f: dict) -> str:
    """
    Generate a unique key for grouping series episodes.
    Priority: tvdb_id > tmdb_id (for tv) > series_name
    """
    if f.get("tvdb_id"):
        return f"tvdb:{f['tvdb_id']}"
    if f.get("tmdb_id") and f.get("tmdb_media_type") == "tv":
        return f"tmdb:{f['tmdb_id']}"
    if f.get("series_name"):
        return f"name:{f['series_name'].lower().strip()}"
    return f"path:{f.get('path', '')}"


def _get_series_display_title(episodes: list[dict]) -> str:
    """
    Get the display title for a series.
    Priority: tmdb_original_title > tvdb_name > tmdb_title > series_name
    Always use original (English) title for consistency.
    """
    for ep in episodes:
        if ep.get("tmdb_original_title"):
            return ep["tmdb_original_title"]
    for ep in episodes:
        if ep.get("tvdb_name"):
            return ep["tvdb_name"]
    for ep in episodes:
        if ep.get("tmdb_title"):
            return ep["tmdb_title"]
    for ep in episodes:
        if ep.get("series_name"):
            # Capitalize series name properly
            return ep["series_name"].title()
    return "Unknown Series"


def _group_series_episodes(files: list[dict]) -> list[dict]:
    """
    Group all series episodes by tvdb_id/tmdb_id/series_name.
    Returns ONE entry per series with episode count and metadata.
    """
    # Filter only series
    series_files = [f for f in files if f.get("media_type") == "series"]

    # Group by series key
    groups: dict[str, list[dict]] = {}
    for f in series_files:
        key = _get_series_group_key(f)
        if key not in groups:
            groups[key] = []
        groups[key].append(f)

    # Build series list
    result = []
    supabase_url = (settings.supabase_url or "").strip().rstrip("/")

    for key, episodes in groups.items():
        # Get best metadata from any episode
        tvdb_id = None
        tmdb_id = None
        year = None
        poster_url = None
        backdrop_url = None
        overview = None
        status = None

        for ep in episodes:
            if not tvdb_id and ep.get("tvdb_id"):
                tvdb_id = ep["tvdb_id"]
            if not tmdb_id and ep.get("tmdb_id"):
                tmdb_id = ep["tmdb_id"]
            if not year and ep.get("tvdb_year"):
                year = ep["tvdb_year"]
            if not year and ep.get("tmdb_year"):
                year = ep["tmdb_year"]
            if not poster_url and ep.get("poster_url"):
                poster_url = ep["poster_url"]
            if not backdrop_url and ep.get("tmdb_backdrop_path"):
                backdrop_url = f"https://image.tmdb.org/t/p/w1280{ep['tmdb_backdrop_path']}"
            if not overview and ep.get("tmdb_overview"):
                overview = ep["tmdb_overview"]

        # Generate poster URL if missing but we have tmdb_id
        if not poster_url and tmdb_id:
            if supabase_url:
                poster_url = f"{supabase_url}/storage/v1/object/public/posters/{tmdb_id}.jpg"
            elif _poster_local_path(tmdb_id).exists():
                poster_url = f"/static/posters/{tmdb_id}.jpg"

        # Get display title (original/English)
        original_title = _get_series_display_title(episodes)

        # Get localized title if different
        localized_title = None
        for ep in episodes:
            if ep.get("tmdb_title") and ep["tmdb_title"] != original_title:
                localized_title = ep["tmdb_title"]
                break

        # Count seasons
        seasons = set()
        for ep in episodes:
            if ep.get("season") is not None:
                seasons.add(ep["season"])

        # Watched count
        watched_count = sum(1 for ep in episodes if ep.get("watched"))

        # Get TMDB details for additional info (year range, official episode count)
        first_air_year = year
        last_air_year = None
        total_episodes_tmdb = None
        status_tmdb = None

        if tmdb_id:
            tmdb_details = _get_cached_tmdb_tv_details(tmdb_id)
            if tmdb_details:
                # Extract year range
                first_air = tmdb_details.get("first_air_date", "")
                last_air = tmdb_details.get("last_air_date", "")
                if first_air:
                    first_air_year = first_air[:4]
                if last_air:
                    last_air_year = last_air[:4]
                # Official episode count from TMDB
                total_episodes_tmdb = tmdb_details.get("number_of_episodes")
                # Status from TMDB
                status_tmdb = tmdb_details.get("status")
                # Use TMDB overview if we don't have one
                if not overview and tmdb_details.get("overview"):
                    overview = tmdb_details.get("overview")
                # Use TMDB original name if we don't have tvdb_name
                if not original_title or original_title == "Unknown Series":
                    tmdb_original = tmdb_details.get("original_name")
                    if tmdb_original:
                        original_title = tmdb_original

        # Build year display string (e.g., "2005-2013" or "2020-")
        year_display = first_air_year
        if last_air_year and last_air_year != first_air_year:
            if status_tmdb == "Ended" or status_tmdb == "Canceled":
                year_display = f"{first_air_year}-{last_air_year}"
            else:
                year_display = f"{first_air_year}-"

        result.append({
            "id": key,  # Unique identifier for this series group
            "tvdb_id": tvdb_id,
            "tmdb_id": tmdb_id,
            "original_title": original_title,  # ALWAYS use for display/sort
            "title": localized_title or original_title,  # Localized if available
            "year": year_display,
            "first_air_year": first_air_year,
            "last_air_year": last_air_year,
            "overview": overview,
            "poster_url": poster_url,
            "backdrop_url": backdrop_url,
            "status": status_tmdb or status,
            "episode_count": len(episodes),  # Local file count
            "total_episodes": total_episodes_tmdb,  # Official TMDB count
            "season_count": len(seasons),
            "watched_count": watched_count,
            "seasons": sorted(seasons) if seasons else [],
        })

    # Sort by original_title (alphabetically, case-insensitive)
    result.sort(key=lambda s: (s["original_title"] or "").lower())

    return result


@app.get("/api/series")
def api_get_series(force_refresh: bool = False):
    """
    Get all series, properly grouped.
    Returns ONE entry per series (not per episode).
    Sorted alphabetically by original_title.
    Utilise un cache pour éviter les rechargements constants.

    Args:
        force_refresh: Si True, ignore le cache et recharge depuis la DB
    """
    cache_key = "series:all"

    # Check cache first (unless force refresh)
    if not force_refresh:
        cached = series_cache.get(cache_key)
        if cached is not None:
            return cached

    # Load from database
    inv = load_inventory()
    files = inv.get("files", [])
    series_list = _group_series_episodes(files)

    result = {
        "ok": True,
        "series": series_list,
        "total": len(series_list),
    }

    # Cache result for 5 minutes
    series_cache.set(cache_key, result, ttl=300)

    return result


@app.get("/api/series/{series_id:path}")
def api_get_series_detail(series_id: str):
    """
    Get a specific series with all its episodes grouped by season.
    series_id format: "tvdb:123" or "tmdb:456" or "name:breaking bad"
    """
    inv = load_inventory()
    files = inv.get("files", [])

    # Filter series files matching this ID
    series_files = [f for f in files if f.get("media_type") == "series"]
    matching_episodes = [f for f in series_files if _get_series_group_key(f) == series_id]

    if not matching_episodes:
        raise HTTPException(status_code=404, detail="Series not found")

    # Get series metadata
    original_title = _get_series_display_title(matching_episodes)

    # Get best metadata
    tvdb_id = None
    tmdb_id = None
    year = None
    poster_url = None
    backdrop_url = None
    overview = None

    for ep in matching_episodes:
        if not tvdb_id and ep.get("tvdb_id"):
            tvdb_id = ep["tvdb_id"]
        if not tmdb_id and ep.get("tmdb_id"):
            tmdb_id = ep["tmdb_id"]
        if not year and (ep.get("tvdb_year") or ep.get("tmdb_year")):
            year = ep.get("tvdb_year") or ep.get("tmdb_year")
        if not poster_url and ep.get("poster_url"):
            poster_url = ep["poster_url"]
        if not backdrop_url and ep.get("tmdb_backdrop_path"):
            backdrop_url = f"https://image.tmdb.org/t/p/w1280{ep['tmdb_backdrop_path']}"
        if not overview and ep.get("tmdb_overview"):
            overview = ep["tmdb_overview"]

    # Generate poster URL if missing
    supabase_url = (settings.supabase_url or "").strip().rstrip("/")
    if not poster_url and tmdb_id:
        if supabase_url:
            poster_url = f"{supabase_url}/storage/v1/object/public/posters/{tmdb_id}.jpg"
        elif _poster_local_path(tmdb_id).exists():
            poster_url = f"/static/posters/{tmdb_id}.jpg"

    # Localized title
    localized_title = None
    for ep in matching_episodes:
        if ep.get("tmdb_title") and ep["tmdb_title"] != original_title:
            localized_title = ep["tmdb_title"]
            break

    # Group episodes by season
    seasons_dict: dict[int, list[dict]] = {}
    for ep in matching_episodes:
        season_num = ep.get("season") or 0
        if season_num not in seasons_dict:
            seasons_dict[season_num] = []

        seasons_dict[season_num].append({
            "episode": ep.get("episode") or 0,
            "title": ep.get("metadata_title") or ep.get("name", ""),
            "filename": ep.get("name", ""),
            "filepath": ep.get("path", ""),
            "duration": ep.get("duration_sec"),
            "width": ep.get("width"),
            "height": ep.get("height"),
            "video_codec": ep.get("video_codec"),
            "hdr": ep.get("hdr", False),
            "watched": ep.get("watched", False),
            "progress": ep.get("progress", 0),
        })

    # Sort episodes within each season
    for season_num in seasons_dict:
        seasons_dict[season_num].sort(key=lambda e: e["episode"])

    # Build seasons list
    seasons_list = []
    for season_num in sorted(seasons_dict.keys()):
        episodes = seasons_dict[season_num]
        watched_in_season = sum(1 for e in episodes if e.get("watched"))
        seasons_list.append({
            "season": season_num,
            "episode_count": len(episodes),
            "watched_count": watched_in_season,
            "episodes": episodes,
        })

    return {
        "ok": True,
        "id": series_id,
        "tvdb_id": tvdb_id,
        "tmdb_id": tmdb_id,
        "original_title": original_title,
        "title": localized_title or original_title,
        "year": year,
        "overview": overview,
        "poster_url": poster_url,
        "backdrop_url": backdrop_url,
        "episode_count": len(matching_episodes),
        "season_count": len(seasons_list),
        "watched_count": sum(1 for ep in matching_episodes if ep.get("watched")),
        "seasons": seasons_list,
    }


@app.get("/api/raw-inventory")
def api_get_raw_inventory():
    """Retourne l'inventaire brut (liste seule, sans ffprobe). Pour mode asynchrone."""
    return load_raw_inventory()


@app.post("/api/delete-file")
def api_delete_file(body: dict = Body(...)):
    """
    Supprime un fichier du NAS et le retire de l'inventaire.
    Body: { "path": "/chemin/vers/fichier.mkv" }.
    Le chemin doit être sous le dossier vidéo ou séries configuré.
    """
    path_str = (body.get("path") or "").strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="path requis")
    path_obj = Path(path_str)
    if not path_obj.is_absolute():
        raise HTTPException(status_code=400, detail="Chemin absolu requis")

    settings = load_settings()
    video_path = settings.get("video_path", "").strip()
    series_path = settings.get("series_path", "").strip()

    if not video_path and not series_path:
        raise HTTPException(status_code=400, detail="Aucun dossier vidéo configuré")

    try:
        file_real = path_obj.resolve()
        allowed = False

        # Check if file is in video_path
        if video_path:
            base_real = Path(video_path).resolve()
            if base_real in file_real.parents or file_real == base_real:
                allowed = True

        # Check if file is in series_path
        if not allowed and series_path:
            series_real = Path(series_path).resolve()
            if series_real in file_real.parents or file_real == series_real:
                allowed = True

        if not allowed:
            raise HTTPException(
                status_code=403,
                detail="Le fichier doit être dans un dossier vidéo configuré",
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not path_obj.exists():
        inv = load_inventory()
        files = [f for f in (inv.get("files") or []) if (f.get("path") or "") != path_str]
        save_inventory(inv.get("scanned_path") or "", files)
        # Also delete from Supabase
        _delete_files_from_supabase([path_str])
        return {"ok": True, "removed_from_inventory": True}
    try:
        path_obj.unlink()
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Impossible de supprimer le fichier : {e}")
    inv = load_inventory()
    files = [f for f in (inv.get("files") or []) if (f.get("path") or "") != path_str]
    save_inventory(inv.get("scanned_path") or "", files)
    # Also delete from Supabase
    _delete_files_from_supabase([path_str])
    return {"ok": True}


@app.post("/api/files/delete")
def api_delete_files(body: dict = Body(...)):
    """
    Supprime plusieurs fichiers du NAS et les retire de l'inventaire.
    Body: { "paths": ["/chemin/vers/fichier1.mkv", "/chemin/vers/fichier2.mkv"] }.
    Tous les chemins doivent être sous le dossier vidéo configuré.
    """
    paths = body.get("paths") or []
    if not paths or not isinstance(paths, list):
        raise HTTPException(status_code=400, detail="paths requis (liste de chemins)")

    video_path = load_settings().get("video_path", "").strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="Dossier vidéo non configuré")

    base_real = Path(video_path).resolve()
    deleted = 0
    errors = []

    for path_str in paths:
        path_str = (path_str or "").strip()
        if not path_str:
            continue

        path_obj = Path(path_str)
        if not path_obj.is_absolute():
            errors.append(f"{path_str}: chemin absolu requis")
            continue

        try:
            file_real = path_obj.resolve()
            if base_real not in file_real.parents:
                errors.append(f"{path_str}: hors du dossier vidéo")
                continue
        except Exception as e:
            errors.append(f"{path_str}: {e}")
            continue

        if path_obj.exists():
            try:
                path_obj.unlink()
                deleted += 1
            except OSError as e:
                errors.append(f"{path_str}: {e}")
        else:
            deleted += 1  # Count as deleted if doesn't exist

    # Retirer tous les fichiers supprimés de l'inventaire
    inv = load_inventory()
    paths_set = set(paths)
    files = [f for f in (inv.get("files") or []) if f.get("path") not in paths_set]

    # Supprimer de Supabase explicitement
    if is_supabase_enabled():
        _delete_files_from_supabase(paths)

    save_inventory(inv.get("scanned_path") or "", files)

    return {
        "ok": len(errors) == 0,
        "deleted": deleted,
        "errors": errors if errors else None,
    }


@app.get("/api/sample-files")
def api_get_sample_files():
    """
    Récupère la liste des fichiers contenant 'sample' dans leur nom.
    Ces fichiers sont généralement des extraits/previews à supprimer.
    """
    inv = load_inventory()
    files = inv.get("files") or []

    sample_files = []
    for f in files:
        name = (f.get("name") or "").lower()
        if "sample" in name:
            sample_files.append({
                "path": f.get("path"),
                "name": f.get("name"),
                "size_bytes": f.get("size_bytes", 0),
                "duration_sec": f.get("duration_sec", 0),
            })

    return {
        "count": len(sample_files),
        "files": sample_files,
        "total_size_bytes": sum(f["size_bytes"] for f in sample_files),
    }


@app.post("/api/sample-files/delete")
def api_delete_sample_files():
    """
    Supprime tous les fichiers contenant 'sample' dans leur nom.
    """
    video_path = load_settings().get("video_path", "").strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="Dossier vidéo non configuré")

    inv = load_inventory()
    files = inv.get("files") or []

    sample_paths = []
    for f in files:
        name = (f.get("name") or "").lower()
        if "sample" in name:
            sample_paths.append(f.get("path"))

    if not sample_paths:
        return {"ok": True, "deleted": 0, "message": "Aucun fichier sample trouvé"}

    deleted = 0
    errors = []

    for path_str in sample_paths:
        try:
            path_obj = Path(path_str)
            if path_obj.exists():
                path_obj.unlink()
                deleted += 1
            else:
                deleted += 1  # Count as deleted if doesn't exist
        except Exception as e:
            errors.append(f"{path_str}: {e}")

    # Retirer les fichiers supprimés de l'inventaire
    paths_set = set(sample_paths)
    remaining_files = [f for f in files if f.get("path") not in paths_set]

    # Supprimer de Supabase explicitement
    if is_supabase_enabled():
        _delete_files_from_supabase(sample_paths)

    # Sauvegarder l'inventaire mis à jour (local backup)
    save_inventory(inv.get("scanned_path") or "", remaining_files)

    return {
        "ok": len(errors) == 0,
        "deleted": deleted,
        "errors": errors if errors else None,
    }


@app.get("/api/empty-folders")
def api_get_empty_folders():
    """
    Récupère la liste des dossiers vides sous le chemin vidéo configuré.
    Un dossier est considéré vide s'il ne contient aucun fichier (mais peut contenir des sous-dossiers vides).
    """
    video_path = load_settings().get("video_path", "").strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="Dossier vidéo non configuré")

    root_path = Path(video_path)
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Dossier vidéo introuvable")

    empty_folders = []

    def is_folder_empty(folder: Path) -> bool:
        """Vérifie si un dossier est vide (récursivement - pas de fichiers du tout)."""
        try:
            for item in folder.iterdir():
                if item.is_file():
                    return False
                if item.is_dir() and not is_folder_empty(item):
                    return False
            return True
        except PermissionError:
            return False

    def find_empty_folders(folder: Path):
        """Trouve tous les dossiers vides récursivement."""
        try:
            for item in folder.iterdir():
                if item.is_dir():
                    if is_folder_empty(item):
                        empty_folders.append(str(item))
                    else:
                        find_empty_folders(item)
        except PermissionError:
            pass

    find_empty_folders(root_path)

    # Trier par profondeur (plus profond en premier pour faciliter la suppression)
    empty_folders.sort(key=lambda x: x.count(os.sep), reverse=True)

    return {
        "count": len(empty_folders),
        "folders": empty_folders,
        "video_path": video_path,
    }


@app.post("/api/empty-folders/delete")
def api_delete_empty_folders():
    """
    Supprime tous les dossiers vides sous le chemin vidéo configuré.
    Les dossiers sont supprimés du plus profond au moins profond.
    """
    video_path = load_settings().get("video_path", "").strip()
    if not video_path:
        raise HTTPException(status_code=400, detail="Dossier vidéo non configuré")

    root_path = Path(video_path)
    if not root_path.exists():
        raise HTTPException(status_code=404, detail="Dossier vidéo introuvable")

    deleted = 0
    errors = []

    def is_folder_empty(folder: Path) -> bool:
        """Vérifie si un dossier est vide."""
        try:
            return not any(folder.iterdir())
        except PermissionError:
            return False

    def delete_empty_folders_recursive(folder: Path) -> int:
        """Supprime les dossiers vides récursivement, en remontant."""
        count = 0
        try:
            # D'abord traiter les sous-dossiers
            for item in list(folder.iterdir()):
                if item.is_dir():
                    count += delete_empty_folders_recursive(item)

            # Ensuite vérifier si ce dossier est maintenant vide
            if folder != root_path and is_folder_empty(folder):
                try:
                    folder.rmdir()
                    count += 1
                except OSError as e:
                    errors.append(f"{folder}: {e}")
        except PermissionError as e:
            errors.append(f"{folder}: Permission refusée")

        return count

    deleted = delete_empty_folders_recursive(root_path)

    return {
        "ok": len(errors) == 0,
        "deleted": deleted,
        "errors": errors if errors else None,
    }


# =============================================================================
# File Normalization API
# =============================================================================

@app.get("/api/normalize/preview/{video_path:path}")
def api_normalize_preview(video_path: str):
    """
    Generate a preview of the normalized filename for a video file.

    The video_path should be the full path to the video file.
    Returns the proposed filename and its components for user validation.
    """
    # URL decode the path
    from urllib.parse import unquote
    decoded_path = unquote(video_path)

    # Find the file in inventory
    inv = load_inventory()
    files = inv.get("files") or []

    file_data = None
    for f in files:
        if f.get("path") == decoded_path:
            file_data = f
            break

    if not file_data:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")

    # Check if file exists
    if not Path(decoded_path).exists():
        raise HTTPException(status_code=404, detail="Le fichier n'existe plus sur le disque")

    # Get TMDB data if available
    tmdb_data = None
    if file_data.get("tmdb_id"):
        tmdb_data = {
            "original_title": file_data.get("tmdb_original_title") or file_data.get("tmdb_title"),
            "title": file_data.get("tmdb_title"),
            "year": file_data.get("tmdb_year"),
        }
    # For series, also check tvdb_name
    elif file_data.get("tvdb_name"):
        tmdb_data = {
            "original_title": file_data.get("tvdb_name"),
            "title": file_data.get("tvdb_name"),
            "year": file_data.get("tvdb_year"),
        }

    # Generate preview
    normalizer = get_normalizer()
    preview = normalizer.generate_preview(decoded_path, file_data, tmdb_data)

    return {
        "ok": True,
        **preview,
    }


@app.post("/api/normalize/execute")
def api_normalize_execute(body: dict = Body(...)):
    """
    Execute the file rename operation.

    Body: {
        path: string,           # Original file path
        components: {           # Filename components
            title: string,
            year: string,
            season: string,     # For series: "S01"
            episode: string,    # For series: "E05"
            episode_title: string,
            resolution: string,
            video_codec: string,
            audio_codec: string,
            audio_channels: string,
            hdr: string,
            source: string,
            extension: string
        },
        dry_run: boolean        # If true, only simulate the rename
    }
    """
    file_path = (body.get("path") or "").strip()
    components = body.get("components") or {}
    dry_run = body.get("dry_run", False)

    if not file_path:
        raise HTTPException(status_code=400, detail="Chemin du fichier requis")

    if not components.get("title"):
        raise HTTPException(status_code=400, detail="Le titre est requis")

    # Execute rename
    normalizer = get_normalizer()
    result = normalizer.execute_rename(file_path, components, dry_run)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result.get("error", "Erreur lors du renommage"))

    # Update inventory if rename was successful and not dry run
    if result["success"] and not dry_run:
        inv = load_inventory()
        files = list(inv.get("files") or [])
        for f in files:
            if f.get("path") == file_path:
                # Update path and name in inventory
                f["path"] = result["new_path"]
                f["name"] = Path(result["new_path"]).name
                break
        save_inventory(inv.get("scanned_path") or "", files)

    return {
        "ok": True,
        **result,
    }


@app.post("/api/normalize/batch-preview")
def api_normalize_batch_preview(body: dict = Body(...)):
    """
    Generate previews for multiple files at once.

    Body: { paths: string[] }
    Returns: { previews: array of preview objects }
    """
    paths = body.get("paths") or []
    if not paths:
        raise HTTPException(status_code=400, detail="Au moins un chemin requis")

    inv = load_inventory()
    files_map = {f.get("path"): f for f in (inv.get("files") or [])}

    normalizer = get_normalizer()
    previews = []

    for file_path in paths:
        file_data = files_map.get(file_path)
        if not file_data:
            previews.append({
                "original_path": file_path,
                "error": "Fichier non trouvé dans l'inventaire",
            })
            continue

        if not Path(file_path).exists():
            previews.append({
                "original_path": file_path,
                "error": "Le fichier n'existe plus sur le disque",
            })
            continue

        # Get TMDB/TVDB data
        tmdb_data = None
        if file_data.get("tmdb_id"):
            tmdb_data = {
                "original_title": file_data.get("tmdb_original_title") or file_data.get("tmdb_title"),
                "title": file_data.get("tmdb_title"),
                "year": file_data.get("tmdb_year"),
            }
        elif file_data.get("tvdb_name"):
            tmdb_data = {
                "original_title": file_data.get("tvdb_name"),
                "title": file_data.get("tvdb_name"),
                "year": file_data.get("tvdb_year"),
            }

        try:
            preview = normalizer.generate_preview(file_path, file_data, tmdb_data)
            previews.append(preview)
        except Exception as e:
            previews.append({
                "original_path": file_path,
                "error": str(e),
            })

    return {
        "ok": True,
        "previews": previews,
    }


# =============================================================================
# Offline Normalizer API (no TMDB/TVDB dependency)
# =============================================================================

@app.post("/api/normalize-offline/analyze")
def api_normalize_offline_analyze(body: dict = Body(...)):
    """
    Analyze directory and detect movies/series WITHOUT API.
    Based purely on filename/folder analysis.

    Body: { root_path: string }
    """
    print(f"[DEBUG] Received body: {body}")
    print(f"[DEBUG] Body type: {type(body)}")
    print(f"[DEBUG] Body keys: {body.keys() if isinstance(body, dict) else 'N/A'}")
    path = (body.get("root_path") or body.get("path") or "").strip()
    print(f"[DEBUG] Extracted path: '{path}'")
    if not path:
        raise HTTPException(status_code=400, detail="Chemin requis")

    normalizer = get_offline_normalizer()
    try:
        analysis = normalizer.analyze_directory(path)
        return {"ok": True, **analysis}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'analyse: {str(e)}")


@app.post("/api/normalize-offline/plan")
def api_normalize_offline_plan(body: dict = Body(...)):
    """
    Generate normalization plan from analysis.

    Body: {
        analysis: object (from /analyze),
        options: {
            keep_technical_tags: boolean (default true),
            create_season_folders: boolean (default true)
        }
    }
    """
    analysis = body.get("analysis")
    options = body.get("options", {})

    if not analysis:
        raise HTTPException(status_code=400, detail="Analyse requise")

    normalizer = get_offline_normalizer()
    try:
        operations = normalizer.generate_plan(analysis, options)
        # Calculate summary
        folders_to_create = len([o for o in operations if o.get("type") == "create_folder"])
        files_to_move = len([o for o in operations if o.get("type") == "move_file"])
        folders_to_cleanup = len([o for o in operations if o.get("type") == "delete_folder_if_empty"])
        return {
            "ok": True,
            "operations": operations,
            "summary": {
                "folders_to_create": folders_to_create,
                "files_to_move": files_to_move,
                "folders_to_cleanup": folders_to_cleanup,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de génération du plan: {str(e)}")


@app.post("/api/normalize-offline/execute")
def api_normalize_offline_execute(body: dict = Body(...)):
    """
    Execute normalization plan. MODIFIES FILESYSTEM.

    Body: {
        operations: array (from /plan),
        dry_run: boolean (default true for safety)
    }
    """
    operations = body.get("operations")
    dry_run = body.get("dry_run", True)  # Default to dry_run for safety

    if not operations:
        raise HTTPException(status_code=400, detail="Opérations requises")

    normalizer = get_offline_normalizer()
    try:
        result = normalizer.execute_plan(operations, dry_run=dry_run)

        # Invalidate caches after modifications (if not dry_run)
        if not dry_run and result.get("success", 0) > 0:
            inventory_cache.invalidate()
            series_cache.invalidate()

        # Format response for frontend (expects success_count, failed_count)
        return {
            "ok": True,
            "success_count": result.get("success", 0),
            "failed_count": result.get("failed", 0),
            "errors": result.get("errors", []),
            "dry_run": result.get("dry_run", dry_run),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur d'exécution: {str(e)}")


# =============================================================================
# Cache Management API
# =============================================================================

@app.get("/api/cache/stats")
def api_cache_stats():
    """Get cache statistics."""
    return {
        "ok": True,
        "inventory_cache": inventory_cache.get_stats(),
        "series_cache": series_cache.get_stats(),
        "general_cache": cache.get_stats(),
    }


@app.post("/api/cache/invalidate")
def api_cache_invalidate(body: dict = Body(...)):
    """
    Invalidate cache.

    Body: {
        cache_type: "inventory" | "series" | "all" (default "all"),
        pattern: string (optional, for partial invalidation)
    }
    """
    cache_type = body.get("cache_type", "all")
    pattern = body.get("pattern")

    invalidated = 0

    if cache_type in ("inventory", "all"):
        invalidated += inventory_cache.invalidate(pattern)

    if cache_type in ("series", "all"):
        invalidated += series_cache.invalidate(pattern)

    if cache_type == "all":
        invalidated += cache.invalidate(pattern)

    return {
        "ok": True,
        "invalidated": invalidated,
    }


@app.post("/api/cache/refresh")
def api_cache_refresh():
    """
    Force refresh all caches by invalidating and reloading.
    Useful after manual database changes.
    """
    # Invalidate all caches
    inventory_cache.invalidate()
    series_cache.invalidate()
    cache.invalidate()

    # Trigger reload by calling the endpoints
    api_get_inventory(force_refresh=True)
    api_get_series(force_refresh=True)

    return {
        "ok": True,
        "message": "Caches rafraîchis",
    }


@app.post("/api/set-file-group")
def api_set_file_group(body: dict = Body(...)):
    """
    Définit ou supprime le groupement personnalisé d'un fichier.
    Body: { path: string, custom_group_key: string | null }
    Si custom_group_key est null, supprime le groupement personnalisé.
    """
    path_str = (body.get("path") or "").strip()
    custom_group_key = body.get("custom_group_key")
    if not path_str:
        raise HTTPException(status_code=400, detail="Chemin requis")
    inv = load_inventory()
    files = list(inv.get("files") or [])
    found = False
    for f in files:
        if f.get("path") == path_str:
            if custom_group_key is None:
                f.pop("custom_group_key", None)
            else:
                f["custom_group_key"] = str(custom_group_key).strip()
            found = True
            break
    if not found:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")
    save_inventory(inv.get("scanned_path") or "", files)
    return {"ok": True}


@app.post("/api/merge-groups")
def api_merge_groups(body: dict = Body(...)):
    """
    Fusionne plusieurs fichiers dans un même groupe personnalisé.
    Body: { paths: string[], group_key: string }
    Tous les fichiers spécifiés auront le même custom_group_key.
    """
    paths = body.get("paths") or []
    group_key = (body.get("group_key") or "").strip()
    if not paths or not group_key:
        raise HTTPException(status_code=400, detail="Chemins et clé de groupe requis")
    inv = load_inventory()
    files = list(inv.get("files") or [])
    updated = 0
    for f in files:
        if f.get("path") in paths:
            f["custom_group_key"] = group_key
            updated += 1
    if updated == 0:
        raise HTTPException(status_code=404, detail="Aucun fichier trouvé dans l'inventaire")
    save_inventory(inv.get("scanned_path") or "", files)
    return {"ok": True, "updated": updated}


@app.post("/api/rescan-file")
def api_rescan_file(body: dict = Body(...)):
    """
    Re-scanne un fichier avec ffprobe pour mettre à jour ses métadonnées.
    Body: { path: string }
    """
    from scanner import run_ffprobe, parse_full_metadata
    path_str = (body.get("path") or "").strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="Chemin requis")
    path_obj = Path(path_str)
    if not path_obj.exists():
        raise HTTPException(status_code=404, detail="Fichier introuvable")

    # Exécuter ffprobe
    probe = run_ffprobe(path_obj)
    new_metadata = parse_full_metadata(path_obj, probe)

    # Mettre à jour l'inventaire
    inv = load_inventory()
    files = list(inv.get("files") or [])
    found = False
    for i, f in enumerate(files):
        if f.get("path") == path_str:
            # Préserver les champs personnalisés
            custom_group_key = f.get("custom_group_key")
            tmdb_id = f.get("tmdb_id")
            tmdb_title = f.get("tmdb_title")
            tmdb_year = f.get("tmdb_year")
            # Remplacer par les nouvelles métadonnées
            files[i] = new_metadata
            # Restaurer les champs personnalisés
            if custom_group_key:
                files[i]["custom_group_key"] = custom_group_key
            if tmdb_id:
                files[i]["tmdb_id"] = tmdb_id
                files[i]["tmdb_title"] = tmdb_title
                files[i]["tmdb_year"] = tmdb_year
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")

    save_inventory(inv.get("scanned_path") or "", files)
    return {"ok": True, "file": new_metadata}


@app.post("/api/tmdb-search")
def api_tmdb_search(body: dict = Body(...)):
    """
    Recherche TMDB et retourne les 10 premiers résultats pour sélection manuelle.
    Body: { query: string, year?: string }
    """
    query = (body.get("query") or "").strip()
    year = (body.get("year") or "").strip() or None

    if not query:
        raise HTTPException(status_code=400, detail="Requête de recherche requise")

    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Clé API TMDB requise")

    headers: dict = {}
    params: dict = {"query": query, "language": "fr-FR"}
    if year:
        params["year"] = year
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(TMDB_SEARCH_URL, params=params, headers=headers or None)
            if r.status_code != 200:
                return {"ok": False, "results": [], "error": "Erreur TMDB"}

            data = r.json() if r.content else {}
            results = data.get("results") or []

            # Retourner les 10 premiers avec les infos utiles
            return {
                "ok": True,
                "results": [
                    {
                        "tmdb_id": m.get("id"),
                        "title": m.get("title"),
                        "original_title": m.get("original_title"),
                        "year": (m.get("release_date") or "")[:4] or None,
                        "overview": (m.get("overview") or "")[:200],
                        "poster_path": m.get("poster_path"),
                        "poster_url": f"https://image.tmdb.org/t/p/w185{m.get('poster_path')}" if m.get("poster_path") else None,
                    }
                    for m in results[:10]
                ],
            }
    except Exception as e:
        return {"ok": False, "results": [], "error": str(e)}


@app.post("/api/tmdb-select")
def api_tmdb_select(body: dict = Body(...)):
    """
    Associe manuellement un film à un ID TMDB sélectionné par l'utilisateur.
    Body: { path: string, tmdb_id: number, tmdb_title: string, tmdb_year: string, tmdb_original_title?: string, poster_url?: string }
    """
    path_str = (body.get("path") or "").strip()
    tmdb_id = body.get("tmdb_id")
    tmdb_title = (body.get("tmdb_title") or "").strip()
    tmdb_year = (body.get("tmdb_year") or "").strip()
    tmdb_original_title = (body.get("tmdb_original_title") or "").strip()
    poster_url = (body.get("poster_url") or "").strip()

    if not path_str:
        raise HTTPException(status_code=400, detail="Chemin requis")
    if not tmdb_id:
        raise HTTPException(status_code=400, detail="ID TMDB requis")

    # Mettre à jour l'inventaire
    inv = load_inventory()
    files = list(inv.get("files") or [])
    found = False
    for f in files:
        if f.get("path") == path_str:
            f["tmdb_id"] = tmdb_id
            f["tmdb_title"] = tmdb_title
            f["tmdb_year"] = tmdb_year
            # Stocker le titre original et l'affiche pour le bon affichage dans la grille
            if tmdb_original_title:
                f["tmdb_original_title"] = tmdb_original_title
            if poster_url:
                f["poster_url"] = poster_url
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")

    save_inventory(inv.get("scanned_path") or "", files)
    return {
        "ok": True,
        "tmdb_id": tmdb_id,
        "tmdb_title": tmdb_title,
        "tmdb_year": tmdb_year,
        "tmdb_original_title": tmdb_original_title,
        "poster_url": poster_url,
    }


@app.post("/api/tvdb-select")
def api_tvdb_select(body: dict = Body(...)):
    """
    Associe manuellement une série à un ID TheTVDB sélectionné par l'utilisateur.
    Body: { path: string, tvdb_id: number, tvdb_name: string, tvdb_year: string }
    Met à jour tous les fichiers qui partagent le même series_name.
    """
    path_str = (body.get("path") or "").strip()
    tvdb_id = body.get("tvdb_id")
    tvdb_name = (body.get("tvdb_name") or "").strip()
    tvdb_year = (body.get("tvdb_year") or "").strip()
    tvdb_poster = (body.get("tvdb_poster") or "").strip()

    if not path_str:
        raise HTTPException(status_code=400, detail="Chemin requis")
    if not tvdb_id:
        raise HTTPException(status_code=400, detail="ID TheTVDB requis")

    # Charger l'inventaire
    inv = load_inventory()
    files = list(inv.get("files") or [])

    # Trouver le fichier source et son series_name
    source_file = None
    for f in files:
        if f.get("path") == path_str:
            source_file = f
            break

    if not source_file:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")

    # Extraire le series_name du fichier source
    source_series_name = source_file.get("series_name") or ""
    if not source_series_name:
        # Essayer de parser le nom du fichier
        file_info = detect_media_type(Path(path_str))
        source_series_name = file_info.get("series_name", "")

    # Extraire le dossier racine de la série (2 niveaux au-dessus du fichier)
    # Ex: /Volumes/series/Beverly Hills, 90210/Season 1/episode.mkv -> /Volumes/series/Beverly Hills, 90210
    source_path = Path(path_str)
    source_series_folder = None
    # Remonter jusqu'à trouver le dossier de la série (généralement 2-3 niveaux)
    for parent in source_path.parents:
        if parent.name and not parent.name.lower().startswith("season") and parent.parent.name:
            # C'est probablement le dossier de la série
            source_series_folder = str(parent)
            break

    updated_count = 0
    updated_paths = []

    # Mettre à jour tous les fichiers de la même série
    # Critères: même series_name OU même dossier racine de série
    for f in files:
        file_series_name = f.get("series_name") or ""
        if not file_series_name:
            # Parser le fichier pour obtenir le series_name
            file_path = f.get("path", "")
            if file_path:
                file_info = detect_media_type(Path(file_path))
                file_series_name = file_info.get("series_name", "")

        # Vérifier si le fichier est dans le même dossier de série
        file_path = f.get("path", "")
        same_folder = False
        if source_series_folder and file_path:
            same_folder = file_path.startswith(source_series_folder + "/")

        # Vérifier si c'est la même série (même series_name OU même dossier OU même path)
        should_update = (
            f.get("path") == path_str or
            same_folder or
            (source_series_name and file_series_name and
             file_series_name.lower() == source_series_name.lower())
        )

        if should_update:
            f["tvdb_id"] = tvdb_id
            f["tvdb_name"] = tvdb_name
            f["tvdb_year"] = tvdb_year
            if tvdb_poster:
                f["poster_url"] = tvdb_poster
            # Marquer comme série identifiée
            f["media_type"] = "series"
            # Effacer les titres TMDB qui pourraient avoir priorité sur tvdb_name
            # pour que le titre corrigé soit bien affiché
            # (on garde tmdb_id pour le casting, recommandations, etc.)
            if "tmdb_original_title" in f:
                del f["tmdb_original_title"]
            if "tmdb_title" in f:
                del f["tmdb_title"]
            updated_count += 1
            updated_paths.append(f.get("path", ""))

    save_inventory(inv.get("scanned_path") or "", files)

    return {
        "ok": True,
        "tvdb_id": tvdb_id,
        "tvdb_name": tvdb_name,
        "tvdb_year": tvdb_year,
        "updated_count": updated_count,
        "updated_paths": updated_paths,
    }


@app.post("/api/reidentify-tmdb")
def api_reidentify_tmdb(body: dict = Body(...)):
    """
    Ré-identifie un fichier avec TMDB (automatique).
    Body: { path: string }
    """
    path_str = (body.get("path") or "").strip()
    if not path_str:
        raise HTTPException(status_code=400, detail="Chemin requis")

    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="Clé API TMDB requise")

    query, year = _query_from_file_path(path_str)
    if not query:
        raise HTTPException(status_code=400, detail="Impossible d'extraire un titre du chemin")

    result = _tmdb_search_sync(query, year)

    # Mettre à jour l'inventaire
    inv = load_inventory()
    files = list(inv.get("files") or [])
    found = False
    for f in files:
        if f.get("path") == path_str:
            if result.get("tmdb_id"):
                f["tmdb_id"] = result["tmdb_id"]
                f["tmdb_title"] = result.get("title") or ""
                f["tmdb_year"] = result.get("year") or ""
                f["tmdb_media_type"] = result.get("tmdb_media_type")
                f["genre_ids"] = result.get("genre_ids") or []
                # Calculate media_type based on TMDB + file patterns
                f["media_type"] = _calculate_media_type(f)
            else:
                # Effacer l'identification TMDB si non trouvé
                f.pop("tmdb_id", None)
                f.pop("tmdb_title", None)
                f.pop("tmdb_year", None)
                f.pop("tmdb_media_type", None)
                f.pop("genre_ids", None)
            found = True
            break

    if not found:
        raise HTTPException(status_code=404, detail="Fichier non trouvé dans l'inventaire")

    save_inventory(inv.get("scanned_path") or "", files)
    return {
        "ok": True,
        "tmdb_id": result.get("tmdb_id"),
        "tmdb_title": result.get("title"),
        "tmdb_year": result.get("year"),
    }


@app.post("/api/reset-all-tmdb")
def api_reset_all_tmdb():
    """
    Réinitialise toutes les données TMDB pour tous les fichiers.
    Permet de relancer l'enrichissement avec un algorithme corrigé.
    """
    client = get_supabase_client()
    if client:
        try:
            # Reset TMDB fields in Supabase
            client.table("movies").update({
                "tmdb_id": None,
                "tmdb_title": None,
                "tmdb_year": None,
            }).neq("path", "").execute()
            print("[SUPABASE] Reset all TMDB data")
        except Exception as e:
            print(f"[SUPABASE] Error resetting TMDB data: {e}")

    # Also reset local inventory
    inv = load_inventory()
    files = list(inv.get("files") or [])
    for f in files:
        f.pop("tmdb_id", None)
        f.pop("tmdb_title", None)
        f.pop("tmdb_year", None)
    _save_inventory_local(inv.get("scanned_path") or "", files)

    return {"ok": True, "message": f"Reset TMDB data for {len(files)} files"}


@app.post("/api/undo-group")
def api_undo_group(body: dict = Body(...)):
    """
    Annule le groupement personnalisé d'un ou plusieurs fichiers.
    Body: { paths: string[] }
    """
    paths = body.get("paths") or []
    if not paths:
        raise HTTPException(status_code=400, detail="Chemins requis")

    inv = load_inventory()
    files = list(inv.get("files") or [])
    updated = 0
    for f in files:
        if f.get("path") in paths and f.get("custom_group_key"):
            f.pop("custom_group_key", None)
            updated += 1

    if updated == 0:
        raise HTTPException(status_code=404, detail="Aucun fichier avec groupement personnalisé trouvé")

    save_inventory(inv.get("scanned_path") or "", files)
    return {"ok": True, "updated": updated}


def _scan_stream_generator(paths: list[str]):
    """
    Scan incrémental : conserve les données TMDB des fichiers existants.
    - Fichiers inchangés (même path, size, mtime) : garde toutes les métadonnées
    - Nouveaux fichiers : scan ffprobe complet
    - Fichiers supprimés : retirés de l'inventaire
    """
    yield json.dumps({"type": "started", "total_expected": -1, "paths": paths}, ensure_ascii=False) + "\n"

    # Load existing inventory for incremental update
    existing_inv = load_inventory()
    existing_files = {f["path"]: f for f in (existing_inv.get("files") or [])}
    yield json.dumps({"type": "info", "message": f"Inventaire existant: {len(existing_files)} fichiers"}, ensure_ascii=False) + "\n"

    inventory: list = []
    stats = {"kept": 0, "new": 0, "updated": 0}
    last_saved = 0
    SAVE_BATCH_SIZE = 50
    primary_path = paths[0] if paths else ""

    try:
        for path in paths:
            if not path or not Path(path).is_dir():
                continue
            yield json.dumps({"type": "scanning_path", "path": path}, ensure_ascii=False) + "\n"

            for row in scan_and_build_inventory_stream(path):
                file_path = row["path"]
                existing = existing_files.get(file_path)

                status = "new"
                if existing:
                    # File exists in inventory - check if unchanged
                    same_size = existing.get("size_bytes") == row.get("size_bytes")
                    # Keep existing TMDB data and other enriched fields
                    if same_size:
                        # Preserve TMDB data
                        row["tmdb_id"] = existing.get("tmdb_id")
                        row["tmdb_title"] = existing.get("tmdb_title")
                        row["tmdb_year"] = existing.get("tmdb_year")
                        row["poster_url"] = existing.get("poster_url")
                        row["custom_group_key"] = existing.get("custom_group_key")
                        stats["kept"] += 1
                        status = "kept"
                    else:
                        stats["updated"] += 1
                        status = "updated"
                    # Remove from existing dict (to track deleted files)
                    del existing_files[file_path]
                else:
                    stats["new"] += 1

                inventory.append(row)
                yield json.dumps({
                    "type": "file",
                    "data": row,
                    "current": len(inventory),
                    "status": status
                }, ensure_ascii=False) + "\n"

                if len(inventory) - last_saved >= SAVE_BATCH_SIZE:
                    save_inventory(primary_path, inventory)
                    last_saved = len(inventory)
                    yield json.dumps({"type": "progress_saved", "saved": last_saved}, ensure_ascii=False) + "\n"
    finally:
        if inventory and len(inventory) > last_saved:
            save_inventory(primary_path, inventory)

    # Files remaining in existing_files are deleted (no longer on disk)
    stats["deleted"] = len(existing_files)

    yield json.dumps({
        "type": "done",
        "total": len(inventory),
        "stats": stats,
        "message": f"Conservés: {stats['kept']}, Nouveaux: {stats['new']}, Modifiés: {stats['updated']}, Supprimés: {stats['deleted']}"
    }, ensure_ascii=False) + "\n"


@app.post("/api/scan")
def api_scan(body: dict = Body(...)):
    """Lance un scan du volume, enregistre tout dans inventory.json, retourne l'inventaire."""
    path = (body.get("path") or "").strip() or load_settings().get("video_path", "")
    if not path:
        raise HTTPException(
            status_code=400,
            detail="Indiquez le chemin du dossier vidéo dans Paramètres.",
        )
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Dossier introuvable : {path}")

    files = scan_and_build_inventory(path)
    save_inventory(path, files)
    inv = load_inventory()
    if len(inv.get("files") or []) == 0:
        inv["diagnostic"] = get_folder_diagnostic(path)
    return inv


@app.post("/api/scan-stream")
def api_scan_stream(body: dict = Body(...)):
    """Scan en flux : chaque fichier analysé est envoyé immédiatement (NDJSON). Mode direct = ffprobe."""
    settings_data = load_settings()
    path = (body.get("path") or "").strip() or settings_data.get("video_path", "")
    series_path = settings_data.get("series_path", "").strip()

    if not path:
        raise HTTPException(
            status_code=400,
            detail="Indiquez le chemin du dossier vidéo dans Paramètres.",
        )
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Dossier introuvable : {path}")

    # Build list of paths to scan
    paths_to_scan = [path]
    if series_path and Path(series_path).is_dir():
        paths_to_scan.append(series_path)

    return StreamingResponse(
        _scan_stream_generator(paths_to_scan),
        media_type="application/x-ndjson",
    )


def _scan_raw_stream_generator(paths: list[str]):
    """
    Liste uniquement les fichiers vidéo (path, name, size_bytes, mtime).
    Aucun ffprobe, aucune API — utilisable sans internet. Sauvegarde dans raw_inventory.json.
    Supporte plusieurs chemins.
    """
    # Start immediately - don't wait for count
    yield json.dumps({"type": "started", "total_expected": -1, "paths": paths}, ensure_ascii=False) + "\n"
    raw_list: list = []
    primary_path = paths[0] if paths else ""
    try:
        for path in paths:
            if not path or not Path(path).is_dir():
                continue
            yield json.dumps({"type": "scanning_path", "path": path}, ensure_ascii=False) + "\n"
            for row in scan_raw_stream(path):
                raw_list.append(row)
                yield json.dumps({"type": "file", "data": row, "current": len(raw_list)}, ensure_ascii=False) + "\n"
    finally:
        if raw_list:
            save_raw_inventory(primary_path, raw_list)
    yield json.dumps(
        {"type": "done", "total": len(raw_list)},
        ensure_ascii=False,
    ) + "\n"


@app.post("/api/scan-raw-stream")
def api_scan_raw_stream(body: dict = Body(...)):
    """Scan brut en flux : liste seule (path, name, size, mtime), sans ffprobe ni API. Sans internet."""
    settings_data = load_settings()
    path = (body.get("path") or "").strip() or settings_data.get("video_path", "")
    series_path = settings_data.get("series_path", "").strip()

    if not path:
        raise HTTPException(
            status_code=400,
            detail="Indiquez le chemin du dossier vidéo dans Paramètres.",
        )
    if not Path(path).is_dir():
        raise HTTPException(status_code=400, detail=f"Dossier introuvable : {path}")

    # Build list of paths to scan
    paths_to_scan = [path]
    if series_path and Path(series_path).is_dir():
        paths_to_scan.append(series_path)

    return StreamingResponse(
        _scan_raw_stream_generator(paths_to_scan),
        media_type="application/x-ndjson",
    )


def _build_from_raw_generator(resume: bool = True):
    """
    Charge raw_inventory, exécute ffprobe sur chaque fichier, yield chaque ligne.
    Sauvegarde progressivement tous les 50 fichiers pour éviter la perte de données.

    Si resume=True, reprend là où on s'est arrêté en sautant les fichiers déjà traités.
    """
    raw = load_raw_inventory()
    files_raw = raw.get("files") or []
    scanned_path = raw.get("scanned_path") or ""
    if not files_raw:
        yield json.dumps(
            {"type": "done", "total": 0, "total_expected": 0, "error": "Aucun inventaire brut. Lancez d'abord un scan asynchrone."},
            ensure_ascii=False,
        ) + "\n"
        return

    # Load existing inventory to resume from where we left off
    existing_inventory = load_inventory()
    existing_files = existing_inventory.get("files") or []
    existing_paths = {f.get("path") for f in existing_files} if resume else set()

    # Filter out already processed files
    files_to_process = [f for f in files_raw if f.get("path") not in existing_paths]
    skipped_count = len(files_raw) - len(files_to_process)

    total_raw = len(files_raw)
    total_to_process = len(files_to_process)

    yield json.dumps({
        "type": "started",
        "total_expected": total_to_process,
        "total_raw": total_raw,
        "skipped": skipped_count,
        "resumed": resume and skipped_count > 0
    }, ensure_ascii=False) + "\n"

    if total_to_process == 0:
        yield json.dumps({
            "type": "done",
            "total": skipped_count,
            "total_expected": 0,
            "message": f"Tous les {skipped_count} fichiers sont déjà traités."
        }, ensure_ascii=False) + "\n"
        return

    # Start with existing inventory (to merge with new files)
    inventory: list = list(existing_files) if resume else []
    new_count = 0
    last_saved = len(inventory)
    SAVE_BATCH_SIZE = 50  # Save every 50 files

    try:
        for row in build_inventory_from_raw_stream(files_to_process):
            inventory.append(row)
            new_count += 1
            yield json.dumps({
                "type": "file",
                "data": row,
                "current": new_count,
                "total": total_to_process,
                "total_inventory": len(inventory)
            }, ensure_ascii=False) + "\n"

            # Progressive save every SAVE_BATCH_SIZE files
            if len(inventory) - last_saved >= SAVE_BATCH_SIZE:
                if scanned_path:
                    save_inventory(scanned_path, inventory)
                last_saved = len(inventory)
                yield json.dumps({"type": "progress_saved", "saved": len(inventory)}, ensure_ascii=False) + "\n"
    finally:
        # Final save
        if inventory and scanned_path and len(inventory) > last_saved:
            save_inventory(scanned_path, inventory)

    yield json.dumps({
        "type": "done",
        "total": len(inventory),
        "new_files": new_count,
        "total_expected": total_to_process
    }, ensure_ascii=False) + "\n"


@app.post("/api/build-inventory-from-raw")
def api_build_inventory_from_raw(body: dict = Body(default={})):
    """
    À partir de raw_inventory.json (liste seule), exécute ffprobe sur chaque fichier
    et enregistre l'inventaire complet dans inventory.json. Sans re-scanner le dossier.
    Utilisable sans internet (ffprobe local).

    Options:
    - resume: bool (default True) - Reprend là où on s'est arrêté, saute les fichiers déjà traités.
    """
    resume = body.get("resume", True)
    return StreamingResponse(
        _build_from_raw_generator(resume=resume),
        media_type="application/x-ndjson",
    )


TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
TMDB_SEARCH_MULTI_URL = "https://api.themoviedb.org/3/search/multi"
TMDB_MOVIE_URL = "https://api.themoviedb.org/3/movie"
TMDB_TV_URL = "https://api.themoviedb.org/3/tv"
DOCUMENTARY_GENRE_ID = 99  # TMDB genre ID for documentaries
TMDB_DETAILS_CACHE_FILE = BASE_DIR / "tmdb_details_cache.json"


def _calculate_media_type(file_data: dict) -> str:
    """
    Calculate the media_type based on TMDB data and file patterns.
    Priority:
    1. If TMDB genre includes Documentary (99) → "documentary" (even if it's a TV series like "13 novembre")
    2. If TMDB says "tv" OR filename has series patterns (S01E01, etc.) → "series"
    3. Otherwise → "movie"
    """
    genre_ids = file_data.get("genre_ids") or []
    tmdb_media_type = file_data.get("tmdb_media_type")

    # Documentary takes priority (even for docuseries like "13 novembre")
    if DOCUMENTARY_GENRE_ID in genre_ids:
        return "documentary"

    # Check TMDB media type for TV shows
    if tmdb_media_type == "tv":
        return "series"

    # Check file patterns for series (S01E01, etc.)
    path_str = file_data.get("path", "")
    if path_str:
        file_info = detect_media_type(Path(path_str))
        if file_info.get("media_type") == "series":
            return "series"

    return "movie"
TMDB_DETAILS_CACHE_DAYS = 7  # Cache movie details for 7 days


def load_tmdb_details_cache() -> dict:
    """Load TMDB movie details cache from file."""
    if not TMDB_DETAILS_CACHE_FILE.exists():
        return {}
    try:
        data = json.loads(TMDB_DETAILS_CACHE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_tmdb_details_cache(cache: dict) -> None:
    """Save TMDB movie details cache to file."""
    TMDB_DETAILS_CACHE_FILE.write_text(
        json.dumps(cache, ensure_ascii=False, indent=0),
        encoding="utf-8",
    )


def is_cache_valid(cache_entry: dict, max_days: int = TMDB_DETAILS_CACHE_DAYS) -> bool:
    """Check if a cache entry is still valid based on timestamp."""
    cached_at = cache_entry.get("_cached_at")
    if not cached_at:
        return False
    try:
        cached_time = datetime.fromisoformat(cached_at.replace("Z", "+00:00"))
        age = datetime.now(timezone.utc) - cached_time
        return age.days < max_days
    except Exception:
        return False

# Mots à retirer des noms de fichiers pour obtenir un titre de film TMDB
_TMDB_STRIP_WORDS = re.compile(
    r"\b(720p|1080p|2160p|4k|uhd|hd|sd|bluray|blu-ray|webrip|web-dl|webdl|web|hdtv|brrip|dvdrip|"
    r"x264|x265|hevc|h264|h265|avc|aac|ac3|dts|flac|eac3|truehd|atmos|"
    r"french|english|multi|vff|vfq|vo|vostfr|vf|en|fr|sub|subs|truefrench|dubbed|"
    r"internal|2in1|sample|proper|repack|"
    r"remux|extended|uncut|remastered|theatrical|directors\.?cut|final\.?cut|redux|"
    r"complete|trilogy|collection|criterion|masters|restored|reconstructed|"
    r"yts|yify|rarbg|eztv|ettv|sparks|fgt|evo|geckos|amiable|bokutox|ntg|sigma|don|cinefile|playhd|playbd|"
    r"kralikmarko|trollhd|lazy|ion10|"
    r"amzn|nf|hmax|dsnp|atvp|pcok|hulu|ddp|ddp5\.1|ddp7\.1|5\.1|7\.1)\b",
    re.IGNORECASE,
)


def _extract_title_year_simple(filename: str) -> tuple[str | None, str | None]:
    """
    Extraction SIMPLE type Plex : titre + année depuis le nom de fichier.
    Pas de normalisation complexe.

    Gère les cas comme "Blade.Runner.2049.2017.1080p" où 2049 fait partie du titre.
    """
    if not filename:
        return None, None

    # Retirer l'extension
    name = re.sub(r"\.(mkv|mp4|avi|mov|wmv|flv|webm|m4v|ts|iso)$", "", filename, flags=re.IGNORECASE)

    # Retirer les préfixes de numérotation (01, 02, etc.)
    name = re.sub(r"^(\d{1,2})[\s._-]+", "", name)

    # Tags techniques qui indiquent qu'on a dépassé le titre
    tech_tags = r"(?:1080p|720p|2160p|4k|uhd|bluray|bdrip|brrip|webrip|web-dl|webdl|hdtv|dvdrip|hdrip|x264|x265|hevc|h264|h265|avc|remux|proper|repack|internal|limited|unrated|extended|directors|criterion)"

    # Méthode 1: Format "Title (Year)" - ex: "Avatar (2009)"
    match = re.search(r"^(.+?)\s*\((\d{4})\)", name)
    if match:
        return match.group(1).strip(), match.group(2)

    # Méthode 2: Chercher une année SUIVIE d'un tag technique
    # Ex: "Blade.Runner.2049.2017.1080p" -> titre="Blade Runner 2049", année="2017"
    match = re.search(rf"^(.+?)[.\s_-]((?:19|20)\d{{2}})[.\s_-]+{tech_tags}", name, re.IGNORECASE)
    if match:
        title = match.group(1).replace(".", " ").replace("_", " ").strip()
        return title, match.group(2)

    # Méthode 3: Format "Title.Year" simple sans tag technique après
    # Mais seulement si l'année est <= année courante + 1 (pas 2049 qui est dans le futur)
    import datetime
    current_year = datetime.datetime.now().year
    match = re.search(r"^(.+?)[.\s_-]((?:19|20)\d{2})(?:[.\s_-]|$)", name)
    if match:
        year_found = int(match.group(2))
        # Si l'année est réaliste (pas dans le futur lointain), on l'utilise
        if year_found <= current_year + 1:
            title = match.group(1).replace(".", " ").replace("_", " ").strip()
            return title, match.group(2)
        # Sinon, l'année fait partie du titre - chercher une autre année après
        rest_of_name = name[match.end(1):]
        match2 = re.search(r"[.\s_-]((?:19|20)\d{2})(?:[.\s_-]|$)", rest_of_name)
        if match2:
            year2 = int(match2.group(1))
            if year2 <= current_year + 1:
                # Inclure le "faux" numéro dans le titre
                title = name[:match.end(1) + match2.start(1)].replace(".", " ").replace("_", " ").strip()
                return title, match2.group(1)

    # Méthode 4: Pas d'année trouvée - on prend le nom tel quel (points -> espaces)
    title = name.replace(".", " ").replace("_", " ").strip()
    # Couper au premier tag technique évident
    for tag in ["1080p", "720p", "2160p", "4k", "bluray", "webrip", "web-dl", "hdtv", "dvdrip", "x264", "x265", "hevc"]:
        idx = title.lower().find(tag)
        if idx > 0:
            title = title[:idx].strip(" -")
            break
    return title if title else None, None


def _normalize_query_for_tmdb(query: str, year: str | None = None) -> str:
    """Simple: juste nettoyer les espaces, TMDB fait le reste."""
    if not query:
        return ""
    return re.sub(r"\s+", " ", query).strip()[:100]


def _shorten_query_for_tmdb(query: str, max_words: int = 4) -> str:
    """Réduit la requête aux N premiers mots pour un fallback TMDB."""
    if not query or not query.strip():
        return ""
    words = query.strip().split()
    return " ".join(words[:max_words]) if words else ""


def _clean_edition_suffixes(title: str) -> str:
    """Nettoie les suffixes d'édition/version du titre pour améliorer la recherche TMDB."""
    if not title:
        return title
    # Patterns d'édition à retirer (insensible à la casse)
    edition_patterns = [
        r"\s*[-–—]\s*Directors?\s*Cut.*$",
        r"\s*[-–—]\s*Extended.*$",
        r"\s*[-–—]\s*Theatrical.*$",
        r"\s*[-–—]\s*Unrated.*$",
        r"\s*[-–—]\s*Special\s*Edition.*$",
        r"\s*[-–—]\s*Remastered.*$",
        r"\s*\d+(?:th|st|nd|rd)\s*Anniversary.*$",
        r"\s*Anniversary\s*Edition.*$",
        r"\s*Collector'?s?\s*Edition.*$",
        r"\s*Ultimate\s*Edition.*$",
        r"\s*Final\s*Cut.*$",
        r"\s*Redux.*$",
        r"\s*Complete\s*Collection.*$",
        r"\s*Criterion.*$",
        r"\s*\[.*\]$",  # Tout entre crochets à la fin
        r"\s*\(.*\)$",  # Tout entre parenthèses à la fin (sauf année)
    ]
    cleaned = title
    for pattern in edition_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_year_from_string(s: str) -> str | None:
    """Extrait une année 4 chiffres (19xx ou 20xx) d'une chaîne."""
    if not s:
        return None
    match = re.search(r"\b(19\d{2}|20\d{2})\b", s)
    return match.group(1) if match else None


_GENERIC_FOLDERS = frozenset(
    {"videos", "movies", "films", "film", "video", "films hd", "bluray", "series",
     "downloads", "download", "dl", "torrents", "torrent", "incoming", "complete",
     "incomplete", "new", "recent", "temp", "tmp", "media", "multimedia", "hd", "4k",
     "uhd", "1080p", "720p", "x264", "x265", "hevc", "h264", "bluray", "dvdrip",
     "webrip", "web-dl", "hdtv", "remux", "rartv", "yts", "yify", "eztv"}
)


def _looks_like_film_title(name: str) -> bool:
    """Vérifie si un nom ressemble à un titre de film (contient des mots, pas juste des tags techniques)."""
    if not name or len(name) < 3:
        return False
    # Trop court (moins de 2 mots et pas d'année) = probablement pas un titre
    words = re.sub(r"[.\-_]", " ", name).split()
    # Si c'est un seul mot court sans année, c'est probablement un dossier générique
    if len(words) == 1 and len(name) <= 10 and not re.search(r"\b(19|20)\d{2}\b", name):
        return False
    # Vérifier s'il y a au moins un mot qui n'est pas un tag technique
    tech_tags = {"1080p", "720p", "2160p", "4k", "uhd", "hdr", "x264", "x265", "hevc",
                 "h264", "bluray", "webrip", "web", "dl", "hdtv", "remux", "yts", "yify",
                 "rartv", "eztv", "french", "english", "multi", "vff", "vostfr", "ac3",
                 "dts", "aac", "mp3", "mkv", "mp4", "avi", "s01", "s02", "s03", "e01"}
    real_words = [w for w in words if w.lower() not in tech_tags and len(w) > 1]
    return len(real_words) >= 1


def _query_from_file_path(path_str: str) -> tuple[str, str | None]:
    """
    Extraction SIMPLE type Plex : titre + année depuis le nom de fichier.
    PAS de normalisation complexe - TMDB fait le matching intelligent.
    """
    p = Path(path_str)
    filename = p.name  # Nom du fichier avec extension

    # Méthode Plex: extraire titre + année simplement
    title, year = _extract_title_year_simple(filename)

    # Si pas trouvé dans le nom de fichier, essayer le dossier parent
    if not title:
        parent_name = (p.parent.name or "").strip()
        if parent_name and parent_name.lower() not in _GENERIC_FOLDERS:
            title, year = _extract_title_year_simple(parent_name)

    return title or "", year


def _has_non_latin_chars(text: str) -> bool:
    """Détecte caractères non-latins (arabe, chinois, cyrillique, etc.)."""
    if not text:
        return False
    for char in text:
        code = ord(char)
        # Arabe: 0x0600-0x06FF, Chinois: 0x4E00-0x9FFF, Cyrillique: 0x0400-0x04FF
        # Japonais Hiragana/Katakana: 0x3040-0x30FF, Coréen: 0xAC00-0xD7AF
        if (0x0600 <= code <= 0x06FF or  # Arabic
            0x4E00 <= code <= 0x9FFF or  # Chinese
            0x0400 <= code <= 0x04FF or  # Cyrillic
            0x3040 <= code <= 0x30FF or  # Japanese
            0xAC00 <= code <= 0xD7AF):   # Korean
            return True
    return False


def _title_similarity(search_title: str, result_title: str) -> float:
    """
    Calcule la similarité entre deux titres (0.0 à 1.0).
    Compare les mots en commun.
    """
    if not search_title or not result_title:
        return 0.0

    # Normaliser: lowercase, enlever ponctuation
    def normalize(t):
        t = t.lower()
        t = re.sub(r"[^\w\s]", " ", t)
        return set(t.split())

    search_words = normalize(search_title)
    result_words = normalize(result_title)

    if not search_words or not result_words:
        return 0.0

    common = search_words & result_words
    # Jaccard similarity
    union = search_words | result_words
    return len(common) / len(union) if union else 0.0


def _validate_tmdb_result(result: dict, search_query: str, expected_type: str | None = None) -> bool:
    """
    Validation STRICTE du résultat TMDB.

    Rejette si:
    - Titre en langue non-latine quand recherche en latin
    - Aucun mot en commun avec la recherche (ni titre ni titre original)
    - Type ne correspond pas (si spécifié)

    Args:
        result: Résultat TMDB
        search_query: Titre recherché
        expected_type: "movie", "tv", ou None pour accepter les deux
    """
    result_title = result.get("title") or result.get("name") or ""
    original_title = result.get("original_title") or result.get("original_name") or ""
    result_type = result.get("media_type")  # "movie" ou "tv"

    # 1. Vérifier le type si spécifié
    if expected_type and result_type != expected_type:
        print(f"[TMDB VALIDATION] Rejected: type mismatch ({result_type} vs {expected_type})")
        return False

    # 2. Vérifier langue - rejeter non-latin si recherche en latin
    search_has_non_latin = _has_non_latin_chars(search_query)
    result_has_non_latin = _has_non_latin_chars(result_title)
    original_has_non_latin = _has_non_latin_chars(original_title) if original_title else True

    if result_has_non_latin and not search_has_non_latin:
        # Vérifier aussi le titre original (souvent en anglais)
        if not original_title or original_has_non_latin:
            print(f"[TMDB VALIDATION] Rejected: non-latin result '{result_title}' for latin search '{search_query}'")
            return False

    # 3. Vérifier similarité titre
    # Comparer avec titre local ET titre original (crucial pour les traductions)
    sim_title = _title_similarity(search_query, result_title)
    sim_original = _title_similarity(search_query, original_title) if original_title else 0.0
    best_sim = max(sim_title, sim_original)

    # Log pour debug
    if sim_original > sim_title:
        print(f"[TMDB VALIDATION] Using original title: '{original_title}' (sim={sim_original:.2f}) vs local '{result_title}' (sim={sim_title:.2f})")

    if best_sim < 0.1:  # Moins de 10% de mots en commun
        print(f"[TMDB VALIDATION] Rejected: low similarity ({best_sim:.2f}) - '{search_query}' vs '{result_title}' / '{original_title}'")
        return False

    print(f"[TMDB VALIDATION] Accepted: '{result_title}' (sim={best_sim:.2f})")
    return True


def _tmdb_search_sync(query: str, year: str | None, cache: dict | None = None, api_key: str | None = None, expected_type: str | None = None) -> dict:
    """
    Recherche TMDB avec VALIDATION STRICTE des résultats.

    Args:
        query: Titre à rechercher
        year: Année (optionnel)
        cache: Cache des posters (optionnel)
        api_key: Clé API TMDB (optionnel, chargée depuis settings si non fournie)
        expected_type: "movie", "tv", ou None pour recherche multi
    """
    if not (query or "").strip():
        print(f"[TMDB SEARCH] Empty query, returning empty")
        return {}

    title = query.strip()
    year_param = (year or "").strip() or None
    cache_key = f"{title}|{year_param or ''}|{expected_type or 'multi'}"
    print(f"[TMDB SEARCH] Query: '{title}' year={year_param} type={expected_type} cache_key='{cache_key}'")

    # Vérifier le cache
    if cache is None:
        cache = load_poster_cache()
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("tmdb_id"):
        print(f"[TMDB SEARCH] Cache hit: {cached.get('title')}")
        return {
            "tmdb_id": cached.get("tmdb_id"),
            "title": cached.get("title"),
            "year": cached.get("year"),
            "poster_path": cached.get("poster_path"),
            "tmdb_media_type": cached.get("tmdb_media_type"),
            "genre_ids": cached.get("genre_ids"),
        }

    # Charger la clé API
    if api_key is None:
        settings_data = load_settings()
        api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {}

    # Préparer la requête
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"

    def _do_search(q: str, y: str | None, search_type: str | None = None) -> list:
        """Helper pour faire une recherche TMDB."""
        p = {"query": q, "language": "fr-FR"}
        if y:
            p["year"] = y
            if search_type == "tv":
                p["first_air_date_year"] = y
                del p["year"]
        if len(api_key) <= 50:
            p["api_key"] = api_key

        # Choisir l'endpoint selon le type attendu
        if search_type == "tv":
            url = "https://api.themoviedb.org/3/search/tv"
        elif search_type == "movie":
            url = TMDB_SEARCH_URL  # /search/movie
        else:
            url = TMDB_SEARCH_MULTI_URL  # /search/multi

        r = httpx.get(url, params=p, headers=headers or None, timeout=10.0)
        if r.status_code != 200:
            return []
        results = (r.json() if r.content else {}).get("results") or []

        # Pour search/multi, filtrer les "person"
        if search_type is None:
            results = [r for r in results if r.get("media_type") in ("movie", "tv")]
        else:
            # Ajouter media_type pour search/tv et search/movie
            for r in results:
                if "media_type" not in r:
                    r["media_type"] = search_type

        return results

    try:
        results = []
        search_type = expected_type  # "movie", "tv", ou None

        # Requête 1: titre complet + année
        print(f"[TMDB SEARCH] Query: '{title}' year={year_param} type={search_type}")
        results = _do_search(title, year_param, search_type)

        # Requête 2: titre complet sans année
        if not results and year_param:
            results = _do_search(title, None, search_type)

        # Requête 3: titre nettoyé (sans suffixes d'édition) + année
        cleaned_title = _clean_edition_suffixes(title)
        if not results and cleaned_title and cleaned_title != title:
            results = _do_search(cleaned_title, year_param, search_type)
            # Requête 4: titre nettoyé sans année
            if not results and year_param:
                results = _do_search(cleaned_title, None, search_type)

        # Requête 5: premiers mots seulement (fallback ultime)
        if not results:
            short_title = _shorten_query_for_tmdb(cleaned_title or title, max_words=3)
            if short_title and short_title != title and short_title != cleaned_title:
                results = _do_search(short_title, year_param, search_type)
                if not results and year_param:
                    results = _do_search(short_title, None, search_type)

        # Pas de résultat = pas de match
        if not results:
            print(f"[TMDB SEARCH] No results for '{title}'")
            return {}

        # VALIDATION ET SÉLECTION DU MEILLEUR RÉSULTAT
        def _get_result_year(r):
            d = r.get("release_date") or r.get("first_air_date") or ""
            return d[:4] if d else None

        def _score_result(r):
            """Score le résultat avec validation stricte."""
            # D'abord valider - si invalide, score = -1000
            if not _validate_tmdb_result(r, title, expected_type):
                return -1000

            score = 0
            result_year = _get_result_year(r)
            result_type = r.get("media_type")
            result_title = r.get("title") or r.get("name") or ""
            original_title = r.get("original_title") or r.get("original_name") or ""

            # Bonus si l'année correspond exactement
            if year_param and result_year == year_param:
                score += 100

            # Bonus de similarité du titre - comparer avec titre local ET original
            # Prendre le meilleur score (crucial pour les traductions comme "L'Agence tous risques" → "The A-Team")
            sim_local = _title_similarity(title, result_title)
            sim_original = _title_similarity(title, original_title) if original_title else 0.0
            best_sim = max(sim_local, sim_original)

            # Donner BEAUCOUP de poids à la similarité (0-200 points)
            # Une correspondance exacte (1.0) donne 200 points, ce qui domine tout
            score += best_sim * 200

            # Bonus si le type correspond à l'attendu
            if expected_type and result_type == expected_type:
                score += 30

            # Bonus de popularité (normaliser entre 0-10) - faible impact
            pop = r.get("popularity") or 0
            score += min(pop / 10, 10)

            return score

        # Trier par score et prendre le meilleur VALIDE
        results_sorted = sorted(results, key=_score_result, reverse=True)

        # Prendre le premier résultat avec score positif
        first = None
        for r in results_sorted:
            if _score_result(r) > -1000:
                first = r
                break

        if not first:
            print(f"[TMDB SEARCH] All results rejected for '{title}'")
            return {}

        print(f"[TMDB SEARCH] Selected: {first.get('title') or first.get('name')} (ID: {first.get('id')})")
        # Handle both movie (release_date) and TV (first_air_date)
        release_date = first.get("release_date") or first.get("first_air_date") or ""
        release_year = release_date[:4] if release_date else None
        poster_path = first.get("poster_path")
        backdrop_path = first.get("backdrop_path")
        # Get title (movie) or name (tv)
        tmdb_title = first.get("title") or first.get("name")
        original_title = first.get("original_title") or first.get("original_name")
        # Get media type and genres from TMDB
        tmdb_media_type = first.get("media_type")  # "movie" or "tv"
        genre_ids = first.get("genre_ids") or []

        # Stocker TOUTES les données TMDB disponibles pour usage futur
        entry = {
            "tmdb_id": first.get("id"),
            "title": tmdb_title,
            "year": release_year,
            "poster_path": poster_path,
            "tmdb_media_type": tmdb_media_type,
            "genre_ids": genre_ids,
            # Données supplémentaires TMDB
            "original_title": original_title,
            "overview": first.get("overview"),
            "backdrop_path": backdrop_path,
            "popularity": first.get("popularity"),
            "vote_average": first.get("vote_average"),
            "vote_count": first.get("vote_count"),
            "original_language": first.get("original_language"),
            "adult": first.get("adult"),
            "release_date": release_date,
            # Pour les séries TV
            "origin_country": first.get("origin_country"),
        }

        # Mettre en cache (données complètes)
        if cache is not None:
            cache[cache_key] = entry.copy()
        return entry
    except Exception:
        return {}


def _tmdb_test_key_sync() -> tuple[bool, str | None]:
    """
    Teste la clé TMDB avec une recherche « Matrix ». Retourne (True, None) si OK,
    (False, message_erreur) sinon.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return False, "Clé API TMDB requise. Paramètres → Clé API TMDB (v3)."
    params: dict = {"query": "Matrix"}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.get(TMDB_SEARCH_URL, params=params, headers=headers or None)
            if r.status_code == 401:
                return False, "Clé API TMDB invalide ou expirée. Paramètres → Tester la clé."
            if r.status_code != 200:
                return False, f"TMDB a répondu {r.status_code}. Vérifiez la clé API."
            data = r.json() if r.content else {}
            if not isinstance(data.get("results"), list):
                return False, "Réponse TMDB inattendue. Vérifiez la clé API."
            return True, None
    except httpx.TimeoutException:
        return False, "TMDB injoignable (délai dépassé). Vérifiez votre connexion internet."
    except Exception as e:
        return False, f"Erreur TMDB : {e!s}"


def _enrich_tmdb_stream_generator(force: bool = False, media_type_filter: str | None = None):
    """
    Générateur streaming pour l'enrichissement TMDB.
    Envoie: started -> progress (par fichier) -> done
    Sauvegarde l'inventaire tous les 50 fichiers pour éviter de perdre la progression.
    Si force=False, skip les fichiers déjà identifiés (qui ont déjà un tmdb_id).
    Si force=True, ré-enrichit TOUS les fichiers pour mettre à jour les genres.
    Si media_type_filter est spécifié ("series", "movie", "documentary"), filtre les fichiers.
    """
    import time
    inv = load_inventory()
    files = list(inv.get("files") or [])

    # Filtrer par media_type si spécifié
    if media_type_filter and media_type_filter != "all":
        files = [f for f in files if f.get("media_type") == media_type_filter]

    total = len(files)

    if not files:
        yield json.dumps({"type": "started", "total": 0}, ensure_ascii=False) + "\n"
        yield json.dumps({"type": "done", "ok": True, "enriched": 0, "total": 0}, ensure_ascii=False) + "\n"
        return

    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        yield json.dumps({"type": "error", "detail": "Clé API TMDB requise. Paramètres → Clé API TMDB (v3)."}, ensure_ascii=False) + "\n"
        return

    ok, err = _tmdb_test_key_sync()
    if not ok:
        yield json.dumps({"type": "error", "detail": err or "Clé API TMDB invalide."}, ensure_ascii=False) + "\n"
        return

    # Précharger le cache une seule fois (évite N requêtes Supabase)
    poster_cache = load_poster_cache()

    # Si force=True, traiter tous les fichiers; sinon seulement ceux sans tmdb_id
    if force:
        to_enrich = files
        skipped = 0
    else:
        already_enriched = [f for f in files if f.get("tmdb_id")]
        to_enrich = [f for f in files if not f.get("tmdb_id")]
        skipped = len(already_enriched)

    yield json.dumps({
        "type": "started",
        "total": len(to_enrich),
        "total_files": total,
        "skipped": skipped,
        "already_enriched": skipped
    }, ensure_ascii=False) + "\n"

    if not to_enrich:
        yield json.dumps({
            "type": "done",
            "ok": True,
            "enriched": 0,
            "total": total,
            "skipped": skipped,
            "message": f"Tous les {total} fichiers sont déjà identifiés."
        }, ensure_ascii=False) + "\n"
        return

    enriched = 0
    scanned_path = inv.get("scanned_path") or ""
    recently_enriched = []  # Track files enriched in current batch

    for i, f in enumerate(to_enrich):
        path_str = f.get("path") or ""
        found = False

        # Détecter le type de média depuis le nom de fichier
        file_info = detect_media_type(Path(path_str)) if path_str else {}
        detected_type = file_info.get("media_type", "movie")
        series_name = file_info.get("series_name")

        # Pour les séries, utiliser series_name comme requête de recherche
        if detected_type == "series" and series_name:
            query = series_name
            year = None  # Les séries ont rarement l'année dans le nom de fichier
            expected_type = "tv"
            print(f"[ENRICH] Series detected: '{series_name}' S{file_info.get('season', '?')}E{file_info.get('episode', '?')}")
        else:
            query, year = _query_from_file_path(path_str)
            expected_type = "movie" if detected_type == "movie" else None

        if query:
            result = _tmdb_search_sync(query, year, cache=poster_cache, api_key=api_key, expected_type=expected_type)
            if result.get("tmdb_id"):
                # Stocker TOUTES les données TMDB dans les métadonnées du fichier
                f["tmdb_id"] = result["tmdb_id"]
                f["tmdb_title"] = result.get("title") or ""
                f["tmdb_year"] = result.get("year") or ""
                f["tmdb_media_type"] = result.get("tmdb_media_type")
                f["genre_ids"] = result.get("genre_ids") or []
                # Données supplémentaires TMDB
                f["tmdb_original_title"] = result.get("original_title")
                f["tmdb_overview"] = result.get("overview")
                f["tmdb_backdrop_path"] = result.get("backdrop_path")
                f["tmdb_popularity"] = result.get("popularity")
                f["tmdb_vote_average"] = result.get("vote_average")
                f["tmdb_vote_count"] = result.get("vote_count")
                f["tmdb_original_language"] = result.get("original_language")
                f["tmdb_adult"] = result.get("adult")
                f["tmdb_release_date"] = result.get("release_date")
                f["tmdb_origin_country"] = result.get("origin_country")
                f["tmdb_poster_path"] = result.get("poster_path")
                f["tmdb_backdrop_path"] = result.get("backdrop_path")
                # Générer l'URL du poster pour compatibilité
                f["poster_url"] = f"https://image.tmdb.org/t/p/w342{result['poster_path']}" if result.get("poster_path") else None
                # Calculate media_type based on TMDB + file patterns
                f["media_type"] = _calculate_media_type(f)
                enriched += 1
                found = True
                recently_enriched.append(f)

        # Envoyer la progression avec données du fichier si trouvé
        progress_msg = {
            "type": "progress",
            "current": i + 1,
            "total": len(to_enrich),
            "enriched": enriched,
            "file": Path(path_str).name if path_str else "",
            "found": found,
        }
        if found:
            # Inclure les données TMDB pour mise à jour incrémentale côté frontend
            progress_msg["file_data"] = {
                "path": path_str,
                "tmdb_id": f.get("tmdb_id"),
                "tmdb_title": f.get("tmdb_title"),
                "tmdb_year": f.get("tmdb_year"),
                "media_type": f.get("media_type"),
                "genre_ids": f.get("genre_ids"),
                "tmdb_vote_average": f.get("tmdb_vote_average"),
                "tmdb_overview": f.get("tmdb_overview"),
                "poster_url": f.get("poster_url"),
                "tmdb_original_title": f.get("tmdb_original_title"),
                "tmdb_release_date": f.get("tmdb_release_date"),
            }
        yield json.dumps(progress_msg, ensure_ascii=False) + "\n"

        # Sauvegarder batch TMDB tous les 50 fichiers enrichis (rapide - seulement les champs TMDB)
        if len(recently_enriched) >= 50:
            _update_tmdb_batch_supabase(recently_enriched)
            save_poster_cache(poster_cache)  # Sauvegarder le cache aussi
            recently_enriched = []

        time.sleep(0.02)  # 20ms entre chaque requête TMDB

    # Sauvegarde finale des fichiers restants
    if recently_enriched:
        _update_tmdb_batch_supabase(recently_enriched)

    # Sauvegarder le cache poster final
    save_poster_cache(poster_cache)

    # Sauvegarde locale complète en backup
    _save_inventory_local(scanned_path, files)
    yield json.dumps({
        "type": "done",
        "ok": True,
        "enriched": enriched,
        "total": len(to_enrich),
        "skipped": skipped,
        "total_files": total
    }, ensure_ascii=False) + "\n"


@app.post("/api/enrich-tmdb")
def api_enrich_tmdb(body: dict = Body(default={})):
    """
    Enrichit l'inventaire avec les identifiants TMDB (streaming NDJSON).
    Envoie la progression en temps réel pour chaque fichier traité.
    Body: {
        force?: boolean - si true, ré-enrichit même les fichiers déjà identifiés
        media_type?: string - "series", "movie", "documentary" ou "all" (défaut: "all")
    }
    """
    force = body.get("force", False)
    media_type_filter = body.get("media_type") or "all"
    return StreamingResponse(
        _enrich_tmdb_stream_generator(force=force, media_type_filter=media_type_filter),
        media_type="application/x-ndjson",
    )


@app.get("/api/tmdb/search")
async def api_tmdb_search(
    query: str = Query(..., min_length=1),
    year: str | None = Query(None),
):
    """
    Recherche un film sur TMDB par titre (et optionnellement année).
    Nécessite une clé API TMDB (v3) ou token Bearer (v4) dans Paramètres.
    Retourne le meilleur résultat : { tmdb_id, title, year, poster_path } ou { error }.
    Le titre est normalisé (nom de fichier -> titre de film) avant envoi à TMDB.
    """
    query_stripped = query.strip()
    year_stripped = (year or "").strip() or ""
    cache_key = f"{query_stripped}|{year_stripped}"

    cache = load_poster_cache()
    cached = cache.get(cache_key)
    if isinstance(cached, dict) and cached.get("poster_path"):
        _ensure_poster_downloaded(cached.get("poster_path"), cached.get("tmdb_id"))
        poster_url = _poster_url_from_entry(cached)
        return {
            "tmdb_id": cached.get("tmdb_id"),
            "title": cached.get("title"),
            "year": cached.get("year"),
            "poster_path": cached.get("poster_path"),
            "poster_url": poster_url,
        }

    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {
            "tmdb_id": None,
            "title": None,
            "year": None,
            "poster_path": None,
            "poster_url": None,
            "error": "Clé API non configurée. Paramètres → Clé API TMDB (v3).",
        }
    search_query = _normalize_query_for_tmdb(query_stripped, year)
    if not search_query:
        search_query = query.strip()[:100]
    year_param = (year or "").strip() or None
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 1) Requête normalisée + année
            params: dict = {"query": search_query}
            if year_param:
                params["year"] = year_param
            if "Authorization" not in headers:
                params["api_key"] = api_key
            r = await client.get(TMDB_SEARCH_URL, params=params, headers=headers or None)
            data = r.json() if r.content else {}
            if r.status_code != 200:
                msg = data.get("status_message") or data.get("errors", [str(r.status_code)])
                if isinstance(msg, list):
                    msg = msg[0] if msg else str(r.status_code)
                return {
                    "tmdb_id": None,
                    "title": None,
                    "year": None,
                    "poster_path": None,
                    "poster_url": None,
                    "error": f"TMDB: {msg}",
                }
            results = data.get("results") or []
            # 2) Si aucun résultat : réessayer sans année
            if not results and year_param:
                params2: dict = {"query": search_query}
                if "Authorization" not in headers:
                    params2["api_key"] = api_key
                r2 = await client.get(TMDB_SEARCH_URL, params=params2, headers=headers or None)
                data2 = r2.json() if r2.content else {}
                results = data2.get("results") or []
            # 3) Si toujours rien : requête courte (4 premiers mots)
            if not results:
                short_q = _shorten_query_for_tmdb(search_query, 4)
                if short_q and short_q != search_query:
                    params3: dict = {"query": short_q}
                    if year_param:
                        params3["year"] = year_param
                    if "Authorization" not in headers:
                        params3["api_key"] = api_key
                    r3 = await client.get(TMDB_SEARCH_URL, params=params3, headers=headers or None)
                    data3 = r3.json() if r3.content else {}
                    results = data3.get("results") or []
            if not results:
                return {"tmdb_id": None, "title": None, "year": None, "poster_path": None, "poster_url": None}
            first = results[0]
            release = (first.get("release_date") or "")[:4] or None
            poster_path = first.get("poster_path")
            entry = {
                "poster_path": poster_path,
                "tmdb_id": first.get("id"),
                "title": first.get("title"),
                "year": release,
            }
            cache = load_poster_cache()
            cache[cache_key] = entry
            save_poster_cache(cache)
            _ensure_poster_downloaded(poster_path, entry.get("tmdb_id"))
            poster_url = _poster_url_from_entry(entry)
            return {
                "tmdb_id": entry["tmdb_id"],
                "title": entry["title"],
                "year": entry["year"],
                "poster_path": poster_path,
                "poster_url": poster_url,
            }
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        return {
            "tmdb_id": None,
            "title": None,
            "year": None,
            "poster_path": None,
            "poster_url": None,
            "error": str(e),
        }


@app.get("/api/tmdb/search-multi")
async def api_tmdb_search_multi(
    query: str = Query(..., min_length=1),
    year: str | None = Query(None),
    limit: int = Query(10, ge=1, le=20),
):
    """
    Recherche TMDB et retourne PLUSIEURS résultats pour que l'utilisateur choisisse.
    Retourne: { results: [{ tmdb_id, title, year, poster_path, poster_url, overview }, ...] }
    """
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"results": [], "error": "Clé API TMDB non configurée"}

    headers: dict = {}
    params: dict = {"query": query.strip()}
    if year:
        params["year"] = year.strip()
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(TMDB_SEARCH_URL, params=params, headers=headers or None)
            data = r.json() if r.content else {}
            if r.status_code != 200:
                return {"results": [], "error": f"TMDB error: {r.status_code}"}

            results = []
            for movie in (data.get("results") or [])[:limit]:
                poster_path = movie.get("poster_path")
                poster_url = f"{TMDB_IMAGE_BASE}{poster_path}" if poster_path else None
                results.append({
                    "tmdb_id": movie.get("id"),
                    "title": movie.get("title"),
                    "year": (movie.get("release_date") or "")[:4] or None,
                    "poster_path": poster_path,
                    "poster_url": poster_url,
                    "overview": (movie.get("overview") or "")[:200],
                })
            return {"results": results}
    except Exception as e:
        return {"results": [], "error": str(e)}


@app.post("/api/tmdb/assign")
async def api_tmdb_assign(
    path: str = Body(..., embed=True),
    tmdb_id: int = Body(..., embed=True),
    tmdb_title: str = Body(None, embed=True),
    tmdb_year: str = Body(None, embed=True),
    poster_url: str = Body(None, embed=True),
):
    """
    Assigne manuellement un film TMDB à un fichier.
    Met à jour la base Supabase OU le fichier local selon la configuration.
    """
    update_data = {
        "tmdb_id": tmdb_id,
        "tmdb_title": tmdb_title or "",
        "tmdb_year": tmdb_year or "",
    }
    if poster_url:
        update_data["poster_url"] = poster_url

    # Try Supabase first
    client = get_supabase_client()
    if client:
        try:
            client.table("movies").update(update_data).eq("path", path).execute()
            return {"ok": True, "tmdb_id": tmdb_id, "tmdb_title": tmdb_title}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    # Fallback: update local inventory file
    try:
        inv = load_inventory()
        files = inv.get("files") or []
        updated = False
        for f in files:
            if f.get("path") == path:
                f["tmdb_id"] = tmdb_id
                f["tmdb_title"] = tmdb_title or ""
                f["tmdb_year"] = tmdb_year or ""
                if poster_url:
                    f["poster_url"] = poster_url
                # Also update custom_group_key to group by tmdb_id
                f["custom_group_key"] = f"tmdb:{tmdb_id}"
                updated = True
                break

        if updated:
            save_inventory(inv.get("scanned_path") or "", files)
            return {"ok": True, "tmdb_id": tmdb_id, "tmdb_title": tmdb_title}
        else:
            return {"ok": False, "error": f"Fichier non trouvé dans l'inventaire: {path}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.get("/api/poster-cache")
def api_get_poster_cache():
    """
    Retourne le cache des affiches (query|year -> poster_path, poster_url, etc.)
    poster_url pointe vers l'image stockée localement ou TMDB. Pré-remplit l'UI sans rappeler l'API.
    """
    raw = load_poster_cache()
    out = {}
    for key, entry in raw.items():
        if isinstance(entry, dict):
            e = dict(entry)
            e["poster_url"] = _poster_url_from_entry(entry)
            out[key] = e
        else:
            out[key] = entry
    return {"cache": out}


@app.post("/api/posters/migrate-to-supabase")
def api_migrate_posters_to_supabase():
    """
    Migre tous les posters locaux vers Supabase Storage.
    Streaming NDJSON pour suivre la progression.
    """
    def migrate_generator():
        if not is_supabase_enabled():
            yield json.dumps({"type": "error", "detail": "Supabase non configuré."}) + "\n"
            return

        # Lister tous les posters locaux
        local_posters = list(POSTERS_DIR.glob("*.jpg"))
        total = len(local_posters)
        yield json.dumps({"type": "started", "total": total}) + "\n"

        if total == 0:
            yield json.dumps({"type": "done", "migrated": 0, "total": 0}) + "\n"
            return

        _ensure_posters_bucket_exists()
        migrated = 0
        errors = 0

        for i, poster_path in enumerate(local_posters):
            tmdb_id = poster_path.stem  # "12345" from "12345.jpg"
            try:
                image_bytes = poster_path.read_bytes()
                result = _upload_poster_supabase(int(tmdb_id), image_bytes)
                if result:
                    migrated += 1
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                print(f"[MIGRATE] Error migrating {tmdb_id}: {e}")

            if (i + 1) % 50 == 0 or i == total - 1:
                yield json.dumps({
                    "type": "progress",
                    "current": i + 1,
                    "total": total,
                    "migrated": migrated,
                    "errors": errors
                }) + "\n"

        yield json.dumps({
            "type": "done",
            "migrated": migrated,
            "errors": errors,
            "total": total
        }) + "\n"

    return StreamingResponse(migrate_generator(), media_type="application/x-ndjson")


@app.post("/api/database/test-delete")
def api_test_supabase_delete(body: dict = Body(...)):
    """
    Test endpoint pour vérifier la suppression Supabase.
    Body: { "path": "/chemin/vers/fichier" }
    """
    path = body.get("path", "").strip()
    if not path:
        raise HTTPException(status_code=400, detail="path requis")

    client = get_supabase_client()
    if not client:
        return {"error": "No Supabase client", "deleted": 0}

    try:
        # First check if the file exists in Supabase
        check = client.table("movies").select("path").eq("path", path).execute()
        exists = len(check.data) > 0 if check.data else False

        if not exists:
            return {"error": "File not found in Supabase", "path": path, "deleted": 0}

        # Try to delete
        result = client.table("movies").delete().eq("path", path).execute()
        deleted_count = len(result.data) if result.data else 0

        # Verify deletion
        verify = client.table("movies").select("path").eq("path", path).execute()
        still_exists = len(verify.data) > 0 if verify.data else False

        return {
            "path": path,
            "existed_before": exists,
            "deleted_count": deleted_count,
            "still_exists_after": still_exists,
            "success": not still_exists,
        }
    except Exception as e:
        return {"error": str(e), "path": path}


@app.get("/api/database/migration-sql")
def api_get_migration_sql():
    """
    Retourne le SQL de migration à exécuter dans la console Supabase.
    Vérifie aussi quelles colonnes existent déjà.
    """
    # SQL de migration complet
    migration_sql = """-- Migration: Ajouter TOUS les champs TMDB à la table movies
-- Exécuter dans Supabase > SQL Editor

ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_media_type TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS genre_ids INTEGER[];
ALTER TABLE movies ADD COLUMN IF NOT EXISTS media_type TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_original_title TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_overview TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_backdrop_path TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_popularity FLOAT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_vote_average FLOAT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_vote_count INTEGER;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_original_language TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_adult BOOLEAN;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_release_date TEXT;
ALTER TABLE movies ADD COLUMN IF NOT EXISTS tmdb_origin_country TEXT[];
ALTER TABLE movies ADD COLUMN IF NOT EXISTS poster_url TEXT;

-- Index pour performances
CREATE INDEX IF NOT EXISTS idx_movies_media_type ON movies(media_type);
CREATE INDEX IF NOT EXISTS idx_movies_genre_ids ON movies USING GIN(genre_ids);"""

    # Vérifier les colonnes existantes si Supabase est configuré
    existing_columns = []
    missing_columns = []
    required_columns = [
        "tmdb_media_type", "genre_ids", "media_type", "tmdb_original_title",
        "tmdb_overview", "tmdb_backdrop_path", "tmdb_popularity", "tmdb_vote_average",
        "tmdb_vote_count", "tmdb_original_language", "tmdb_adult", "tmdb_release_date",
        "tmdb_origin_country", "poster_url"
    ]

    if is_supabase_enabled():
        client = get_supabase_client()
        if client:
            try:
                # Tester chaque colonne en essayant de la sélectionner
                for col in required_columns:
                    try:
                        client.table("movies").select(col).limit(1).execute()
                        existing_columns.append(col)
                    except Exception:
                        missing_columns.append(col)
            except Exception:
                pass

    return {
        "sql": migration_sql,
        "existing_columns": existing_columns,
        "missing_columns": missing_columns,
        "needs_migration": len(missing_columns) > 0,
        "supabase_url": settings.supabase_url if is_supabase_enabled() else None,
        "message": f"{len(missing_columns)} colonnes manquantes à ajouter." if missing_columns
            else "Toutes les colonnes TMDB sont présentes."
    }


@app.post("/api/posters/refresh/{tmdb_id}")
def api_refresh_poster(tmdb_id: int):
    """
    Force le re-téléchargement d'un poster spécifique depuis TMDB.
    Utile si l'attribution TMDB a changé.
    """
    # Chercher le poster_path dans le cache
    cache = load_poster_cache()
    poster_path = None
    for key, entry in cache.items():
        if isinstance(entry, dict) and entry.get("tmdb_id") == tmdb_id:
            poster_path = entry.get("poster_path")
            break

    if not poster_path:
        # Chercher dans l'inventaire
        inv = load_inventory()
        for f in inv.get("files", []):
            if f.get("tmdb_id") == tmdb_id:
                # On doit refaire une recherche TMDB pour obtenir le poster_path
                query = f.get("tmdb_title") or f.get("name", "")
                year = f.get("tmdb_year")
                result = _tmdb_search_sync(query, year, cache=cache)
                poster_path = result.get("poster_path") if result else None
                break

    if not poster_path:
        return {"ok": False, "error": "Poster non trouvé dans le cache."}

    # Forcer le refresh
    success = _ensure_poster_downloaded(poster_path, tmdb_id, force_refresh=True)
    if success:
        url = _get_poster_url_supabase(tmdb_id) if is_supabase_enabled() else f"/static/posters/{tmdb_id}.jpg"
        return {"ok": True, "poster_url": url}
    return {"ok": False, "error": "Échec du téléchargement."}


@app.get("/api/posters/stats")
def api_posters_stats():
    """
    Retourne les statistiques des posters stockés.
    """
    local_count = len(list(POSTERS_DIR.glob("*.jpg")))
    local_size_bytes = sum(f.stat().st_size for f in POSTERS_DIR.glob("*.jpg"))

    stats = {
        "local": {
            "count": local_count,
            "size_bytes": local_size_bytes,
            "size_mb": round(local_size_bytes / (1024 * 1024), 1)
        },
        "supabase": {
            "enabled": is_supabase_enabled(),
            "bucket": POSTERS_BUCKET if is_supabase_enabled() else None
        }
    }

    return stats


@app.get("/api/tmdb/test")
async def api_tmdb_test():
    """
    Teste la clé TMDB avec une recherche « Matrix ».
    Retourne toujours 200 avec { ok: true } ou { ok: false, error: "..." } (jamais 500).
    """
    try:
        try:
            settings_data = load_settings()
        except Exception as e:
            return {"ok": False, "error": f"Paramètres inaccessibles : {e!s}"}
        api_key = (settings_data.get("tmdb_api_key") or "").strip()
        if not api_key:
            return {"ok": False, "error": "Clé API non configurée. Paramètres → Clé API TMDB (v3)."}
        params: dict = {"query": "Matrix"}
        headers: dict | None = None
        if len(api_key) > 50:
            headers = {"Authorization": f"Bearer {api_key}"}
        else:
            params["api_key"] = api_key
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.get(TMDB_SEARCH_URL, params=params, headers=headers)
            try:
                data = r.json() if r.content else {}
            except json.JSONDecodeError:
                return {"ok": False, "error": f"TMDB a répondu un format inattendu (status {r.status_code})."}
            if r.status_code != 200:
                msg = data.get("status_message") or data.get("status_code") or str(r.status_code)
                return {"ok": False, "error": f"TMDB: {msg}"}
            if not (data.get("results")):
                return {"ok": False, "error": "Aucun résultat (clé peut-être invalide)."}
            return {"ok": True, "message": "Clé TMDB valide."}
    except httpx.TimeoutException:
        return {"ok": False, "error": "Délai dépassé (TMDB injoignable)."}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"Réseau: {e!s}"}
    except Exception as e:
        return {"ok": False, "error": f"Erreur: {e!s}"}


# =============================================================================
# TheTVDB API v4 Integration
# =============================================================================
# TheTVDB is a community-driven TV database with excellent series identification.
# Requires attribution: "TV information and images are provided by TheTVDB.com,
# but we are not endorsed or certified by TheTVDB.com or its affiliates."

TVDB_API_BASE = "https://api4.thetvdb.com/v4"
_tvdb_token: str | None = None
_tvdb_token_expires: float = 0


async def _get_tvdb_token(api_key: str) -> str | None:
    """
    Get a valid TVDB JWT token. Tokens are cached and refreshed as needed.
    TheTVDB API v4 uses JWT authentication with 30-day expiry.
    """
    global _tvdb_token, _tvdb_token_expires
    import time

    # Return cached token if still valid (with 1 hour buffer)
    if _tvdb_token and time.time() < _tvdb_token_expires - 3600:
        return _tvdb_token

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                f"{TVDB_API_BASE}/login",
                json={"apikey": api_key},
                headers={"Content-Type": "application/json"}
            )
            if r.status_code != 200:
                print(f"[TVDB] Login failed: {r.status_code} - {r.text[:200]}")
                return None

            data = r.json()
            token = data.get("data", {}).get("token")
            if not token:
                print(f"[TVDB] No token in response: {data}")
                return None

            _tvdb_token = token
            # TVDB tokens expire in 30 days, but we refresh more often
            _tvdb_token_expires = time.time() + (7 * 24 * 3600)  # Refresh weekly
            print(f"[TVDB] Successfully authenticated")
            return _tvdb_token

    except Exception as e:
        print(f"[TVDB] Login error: {e}")
        return None


async def tvdb_search_series(query: str, year: str | None = None) -> list[dict]:
    """
    Search for TV series on TheTVDB.
    Returns list of matching series with id, name, year, overview, poster.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("tvdb_api_key") or "").strip()
    if not api_key:
        return []

    token = await _get_tvdb_token(api_key)
    if not token:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {"query": query, "type": "series"}
            if year:
                params["year"] = year

            r = await client.get(
                f"{TVDB_API_BASE}/search",
                params=params,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json"
                }
            )

            if r.status_code != 200:
                print(f"[TVDB] Search failed: {r.status_code}")
                return []

            data = r.json()
            results = []
            for item in data.get("data", []):
                # Extract year from first_air_time or year field
                item_year = item.get("year")
                if not item_year and item.get("first_air_time"):
                    item_year = item["first_air_time"][:4]

                results.append({
                    "tvdb_id": item.get("tvdb_id") or item.get("id"),
                    "name": item.get("name") or item.get("translations", {}).get("eng"),
                    "original_name": item.get("name"),
                    "year": item_year,
                    "overview": item.get("overview") or item.get("overviews", {}).get("eng", ""),
                    "poster": item.get("image_url") or item.get("thumbnail"),
                    "status": item.get("status"),
                    "primary_language": item.get("primary_language"),
                })
            return results

    except Exception as e:
        print(f"[TVDB] Search error: {e}")
        return []


async def tvdb_get_series_details(tvdb_id: int) -> dict | None:
    """
    Get detailed information about a TV series from TheTVDB.
    Includes extended info with seasons, episodes, artwork.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("tvdb_api_key") or "").strip()
    if not api_key:
        return None

    token = await _get_tvdb_token(api_key)
    if not token:
        return None

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Get extended series data with episodes
            r = await client.get(
                f"{TVDB_API_BASE}/series/{tvdb_id}/extended",
                params={"meta": "episodes"},
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json"
                }
            )

            if r.status_code != 200:
                print(f"[TVDB] Series details failed: {r.status_code}")
                return None

            data = r.json().get("data", {})
            return {
                "tvdb_id": data.get("id"),
                "name": data.get("name"),
                "original_name": data.get("name"),
                "year": data.get("year") or (data.get("firstAired") or "")[:4],
                "overview": data.get("overview"),
                "status": data.get("status", {}).get("name"),
                "poster": data.get("image"),
                "backdrop": next((a.get("image") for a in data.get("artworks", []) if a.get("type") == 3), None),
                "genres": [g.get("name") for g in data.get("genres", [])],
                "networks": [n.get("name") for n in data.get("networks", [])],
                "seasons": data.get("seasons", []),
                "episodes": data.get("episodes", []),
                "first_aired": data.get("firstAired"),
                "last_aired": data.get("lastAired"),
                "runtime": data.get("averageRuntime"),
            }

    except Exception as e:
        print(f"[TVDB] Series details error: {e}")
        return None


@app.get("/api/tvdb/test")
async def api_tvdb_test():
    """
    Test TVDB API key by authenticating and searching for a test series.
    Returns { ok: true } if successful, { ok: false, error: "..." } otherwise.
    """
    try:
        settings_data = load_settings()
        api_key = (settings_data.get("tvdb_api_key") or "").strip()

        if not api_key:
            return {"ok": False, "error": "Clé API TheTVDB non configurée."}

        # Test authentication
        token = await _get_tvdb_token(api_key)
        if not token:
            return {"ok": False, "error": "Authentification TheTVDB échouée. Vérifiez votre clé API."}

        # Test search
        results = await tvdb_search_series("Breaking Bad")
        if not results:
            return {"ok": False, "error": "Recherche TheTVDB échouée. Token peut-être invalide."}

        return {
            "ok": True,
            "message": f"TheTVDB connecté. Test: {results[0].get('name', 'Breaking Bad')}",
            "results_count": len(results)
        }

    except Exception as e:
        return {"ok": False, "error": f"Erreur: {e!s}"}


@app.get("/api/tvdb/search")
async def api_tvdb_search(query: str = Query(...), year: str | None = Query(None)):
    """
    Search for TV series on TheTVDB.
    """
    if not query.strip():
        return {"ok": False, "results": [], "error": "Requête vide."}

    results = await tvdb_search_series(query.strip(), year)
    return {"ok": True, "results": results}


@app.get("/api/tvdb/series/{tvdb_id}")
async def api_tvdb_series_details(tvdb_id: int):
    """
    Get detailed information about a TV series from TheTVDB.
    """
    details = await tvdb_get_series_details(tvdb_id)
    if not details:
        return {"error": f"Série TheTVDB #{tvdb_id} non trouvée."}
    return details


@app.get("/api/tmdb/movie/{tmdb_id}/full")
async def api_tmdb_movie_full(tmdb_id: int):
    """
    Get complete movie details from TMDB including credits, videos, recommendations, similar.
    Uses append_to_response for a single API call. Results are cached for 7 days.
    Returns the full movie data for Plex-like detail pages.
    """
    cache_key = str(tmdb_id)
    cache = load_tmdb_details_cache()

    # Check cache first
    cached = cache.get(cache_key)
    if cached and is_cache_valid(cached):
        # Remove internal cache metadata before returning
        result = {k: v for k, v in cached.items() if not k.startswith("_")}
        return result

    # Get API key
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {
            "error": "Clé API TMDB non configurée. Paramètres → Clé API TMDB.",
            "id": tmdb_id,
        }

    # Build request with append_to_response for all data in one call
    params: dict = {
        "append_to_response": "credits,videos,recommendations,similar",
        "language": "fr-FR",  # French language for titles/overviews
    }
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{TMDB_MOVIE_URL}/{tmdb_id}"
            r = await client.get(url, params=params, headers=headers or None)

            if r.status_code == 404:
                return {"error": f"Film TMDB #{tmdb_id} non trouvé.", "id": tmdb_id}

            if r.status_code != 200:
                data = r.json() if r.content else {}
                msg = data.get("status_message") or f"Erreur TMDB: {r.status_code}"
                return {"error": msg, "id": tmdb_id}

            data = r.json() if r.content else {}

            # Add cache timestamp and save
            data["_cached_at"] = datetime.now(timezone.utc).isoformat()
            cache[cache_key] = data
            save_tmdb_details_cache(cache)

            # Remove cache metadata before returning
            result = {k: v for k, v in data.items() if not k.startswith("_")}
            return result

    except httpx.TimeoutException:
        return {"error": "TMDB injoignable (délai dépassé).", "id": tmdb_id}
    except Exception as e:
        return {"error": f"Erreur: {e!s}", "id": tmdb_id}


@app.get("/api/tmdb/movie/{tmdb_id}/credits")
async def api_tmdb_movie_credits(tmdb_id: int):
    """Get movie credits (cast and crew) from TMDB."""
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"error": "Clé API TMDB non configurée.", "id": tmdb_id}

    params: dict = {"language": "fr-FR"}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{TMDB_MOVIE_URL}/{tmdb_id}/credits"
            r = await client.get(url, params=params, headers=headers or None)
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}
            return r.json() if r.content else {}
    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


@app.get("/api/tmdb/movie/{tmdb_id}/videos")
async def api_tmdb_movie_videos(tmdb_id: int):
    """Get movie videos (trailers, teasers) from TMDB."""
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"error": "Clé API TMDB non configurée.", "id": tmdb_id}

    params: dict = {"language": "fr-FR"}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Try French first
            url = f"{TMDB_MOVIE_URL}/{tmdb_id}/videos"
            r = await client.get(url, params=params, headers=headers or None)
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}
            data = r.json() if r.content else {}

            # If no French videos, try English
            if not data.get("results"):
                params["language"] = "en-US"
                r = await client.get(url, params=params, headers=headers or None)
                if r.status_code == 200:
                    data = r.json() if r.content else {}

            return data
    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


@app.get("/api/tmdb/movie/{tmdb_id}/recommendations")
async def api_tmdb_movie_recommendations(tmdb_id: int, page: int = Query(1)):
    """Get movie recommendations from TMDB."""
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"error": "Clé API TMDB non configurée.", "id": tmdb_id}

    params: dict = {"language": "fr-FR", "page": page}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{TMDB_MOVIE_URL}/{tmdb_id}/recommendations"
            r = await client.get(url, params=params, headers=headers or None)
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}
            return r.json() if r.content else {}
    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


@app.get("/api/tmdb/movie/{tmdb_id}/similar")
async def api_tmdb_movie_similar(tmdb_id: int, page: int = Query(1)):
    """Get similar movies from TMDB."""
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"error": "Clé API TMDB non configurée.", "id": tmdb_id}

    params: dict = {"language": "fr-FR", "page": page}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{TMDB_MOVIE_URL}/{tmdb_id}/similar"
            r = await client.get(url, params=params, headers=headers or None)
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}
            return r.json() if r.content else {}
    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


# =============================================================================
# TMDB TV Show Endpoints
# =============================================================================

@app.get("/api/tmdb/tv/{tmdb_id}/full")
async def api_tmdb_tv_full(tmdb_id: int):
    """
    Get complete TV show details from TMDB including credits, videos, recommendations, similar.
    Uses append_to_response for a single API call. Results are cached for 7 days.
    """
    cache_key = f"tv_{tmdb_id}"
    cache = load_tmdb_details_cache()

    # Check cache first
    cached = cache.get(cache_key)
    if cached and is_cache_valid(cached):
        result = {k: v for k, v in cached.items() if not k.startswith("_")}
        return result

    # Get API key
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {
            "error": "Clé API TMDB non configurée. Paramètres → Clé API TMDB.",
            "id": tmdb_id,
        }

    # Build request with append_to_response for all data in one call
    params: dict = {
        "append_to_response": "credits,videos,recommendations,similar,external_ids",
        "language": "fr-FR",
    }
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"{TMDB_TV_URL}/{tmdb_id}"
            r = await client.get(url, params=params, headers=headers or None)

            if r.status_code == 404:
                return {"error": f"Série TV TMDB #{tmdb_id} non trouvée.", "id": tmdb_id}
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}

            data = r.json() if r.content else {}

            # Add media_type for frontend
            data["media_type"] = "tv"

            # Cache the result
            data["_cached_at"] = datetime.now(timezone.utc).isoformat()
            cache[cache_key] = data
            save_tmdb_details_cache(cache)

            # Return without cache metadata
            return {k: v for k, v in data.items() if not k.startswith("_")}

    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


@app.get("/api/tmdb/tv/{tmdb_id}/season/{season_number}")
async def api_tmdb_tv_season(tmdb_id: int, season_number: int):
    """
    Get details for a specific season of a TV show, including all episodes.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("tmdb_api_key") or "").strip()
    if not api_key:
        return {"error": "Clé API TMDB non configurée.", "id": tmdb_id}

    params: dict = {"language": "fr-FR"}
    headers: dict = {}
    if len(api_key) > 50:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        params["api_key"] = api_key

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{TMDB_TV_URL}/{tmdb_id}/season/{season_number}"
            r = await client.get(url, params=params, headers=headers or None)
            if r.status_code == 404:
                return {"error": f"Saison {season_number} non trouvée.", "id": tmdb_id, "season": season_number}
            if r.status_code != 200:
                return {"error": f"Erreur TMDB: {r.status_code}", "id": tmdb_id}
            return r.json() if r.content else {}
    except Exception as e:
        return {"error": str(e), "id": tmdb_id}


@app.get("/api/tmdb/media/{media_type}/{tmdb_id}/full")
async def api_tmdb_media_full(media_type: str, tmdb_id: int):
    """
    Unified endpoint that handles both movies and TV shows.
    media_type should be 'movie' or 'tv'.
    """
    if media_type == "tv":
        return await api_tmdb_tv_full(tmdb_id)
    else:
        return await api_tmdb_movie_full(tmdb_id)


@app.get("/api/ffmpeg")
def api_ffmpeg_status():
    ffprobe = get_ffprobe_path()
    installed = ffprobe is not None
    try:
        brew = shutil.which("brew") is not None
    except Exception:
        brew = False
    return {
        "installed": installed,
        "ffprobe_path": ffprobe or "",
        "hint_brew": brew,
        "hint_message": "brew install ffmpeg" if brew else "Installez ffmpeg (https://ffmpeg.org/download.html) puis redémarrez l'app.",
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/supabase/status")
def api_supabase_status():
    """
    Retourne le statut de la connexion Supabase.
    Utile pour vérifier si le stockage cloud est configuré.
    """
    supabase_url = settings.supabase_url.strip() if settings.supabase_url else ""
    has_url = bool(supabase_url)
    has_key = bool(settings.supabase_key.strip() if settings.supabase_key else "")

    if not has_url or not has_key:
        return {
            "enabled": False,
            "connected": False,
            "message": "Supabase non configuré. Définissez SUPABASE_URL et SUPABASE_KEY dans l'environnement.",
            "storage": "local",
        }

    client = get_supabase_client()
    if client:
        # Test connection with a simple query
        try:
            client.table("scan_metadata").select("id").limit(1).execute()
            return {
                "enabled": True,
                "connected": True,
                "url": supabase_url,
                "message": "Supabase connecté.",
                "storage": "supabase",
            }
        except Exception as e:
            return {
                "enabled": True,
                "connected": False,
                "url": supabase_url,
                "message": f"Erreur de connexion: {str(e)}",
                "storage": "local",
            }

    return {
        "enabled": True,
        "connected": False,
        "message": "Impossible de créer le client Supabase.",
        "storage": "local",
    }


# =============================================================================
# NZB Search API (omgwtfnzbs)
# =============================================================================

# API omgwtfnzbs - format: /json/?search=...&user=...&api=...
NZB_API_BASE = "https://api.omgwtfnzbs.org"
NZB_CATEGORIES = {
    "movies": "15,16,31,35,17,18",  # Movies: All
    "movies_sd": "15",              # Movies: SD
    "movies_hd": "16",              # Movies: HD
    "movies_4k": "31",              # Movies: UHD
    "movies_bluray": "35",          # Movies: Full BR
    "movies_dvd": "17",             # Movies: DVD
}


@app.get("/api/nzb/test")
async def api_nzb_test():
    """Teste la connexion à l'API omgwtfnzbs avec les credentials configurés."""
    settings_data = load_settings()
    api_key = (settings_data.get("nzb_api_key") or "").strip()
    api_user = (settings_data.get("nzb_api_user") or "").strip()

    if not api_key or not api_user:
        return {"ok": False, "error": "Clé API et nom d'utilisateur NZB requis (Paramètres)."}

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params = {
                "search": "Matrix",
                "user": api_user,
                "api": api_key,
                "catid": NZB_CATEGORIES["movies"],
            }
            r = await client.get(f"{NZB_API_BASE}/json/", params=params)
            if r.status_code != 200:
                return {"ok": False, "error": f"Erreur HTTP {r.status_code}: {r.text[:200]}"}

            # Parser le JSON avec gestion d'erreur
            try:
                data = r.json() if r.content else []
            except Exception as json_err:
                return {"ok": False, "error": f"Réponse JSON invalide: {r.text[:200]}"}

            # Gérer les différents formats de réponse
            if isinstance(data, dict):
                if data.get("error"):
                    return {"ok": False, "error": data.get("error")}
                if data.get("notice"):
                    return {"ok": False, "error": data.get("notice")}
                # Certaines API retournent un dict avec des résultats
                return {"ok": True, "message": "Connexion NZB réussie.", "results_count": 0}

            if isinstance(data, list):
                return {"ok": True, "message": "Connexion NZB réussie.", "results_count": len(data)}

            return {"ok": False, "error": f"Format de réponse inattendu: {type(data).__name__}"}

    except httpx.TimeoutException:
        return {"ok": False, "error": "Délai dépassé (API NZB injoignable)."}
    except httpx.RequestError as e:
        return {"ok": False, "error": f"Erreur réseau: {e!s}"}
    except Exception as e:
        return {"ok": False, "error": f"Erreur inattendue: {e!s}"}


@app.get("/api/nzb/search")
async def api_nzb_search(
    query: str = Query(..., min_length=1),
    category: str = Query("movies_hd"),
):
    """
    Recherche NZB sur omgwtfnzbs.
    Categories: movies, movies_hd, movies_sd, movies_4k
    Retourne une liste de résultats avec titre, taille, qualité, etc.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("nzb_api_key") or "").strip()
    api_user = (settings_data.get("nzb_api_user") or "").strip()

    if not api_key or not api_user:
        raise HTTPException(status_code=400, detail="Clé API et nom d'utilisateur NZB requis (Paramètres).")

    cat_id = NZB_CATEGORIES.get(category, NZB_CATEGORIES["movies_hd"])

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            params = {
                "search": query.strip(),
                "user": api_user,
                "api": api_key,
                "catid": cat_id,
            }
            r = await client.get(f"{NZB_API_BASE}/json/", params=params)
            if r.status_code != 200:
                raise HTTPException(status_code=502, detail=f"Erreur API NZB: HTTP {r.status_code}")

            data = r.json() if r.content else []

            # Gérer les erreurs de l'API
            if isinstance(data, dict):
                if data.get("error"):
                    raise HTTPException(status_code=400, detail=data.get("error"))
                if data.get("notice"):
                    return {"results": [], "notice": data.get("notice")}

            if not isinstance(data, list):
                return {"results": []}

            # Transformer les résultats (jusqu'à 500 résultats selon l'API)
            results = []
            for item in data:
                # Extraire les infos de qualité du titre
                title = item.get("release") or item.get("title") or ""
                title_upper = title.upper()
                size_bytes = int(item.get("sizebytes") or 0)

                # Détecter la qualité depuis le titre
                quality = "SD"
                if any(q in title_upper for q in ["2160P", "4K", "UHD"]):
                    quality = "4K"
                elif any(q in title_upper for q in ["1080P", "BLURAY", "BLU-RAY", "BDRIP"]):
                    quality = "1080p"
                elif "720P" in title_upper:
                    quality = "720p"
                elif "480P" in title_upper:
                    quality = "480p"

                # Détecter HDR (HDR10, HDR10+)
                hdr = any(h in title_upper for h in ["HDR10+", "HDR10", "HDR", "HLG"])

                # Détecter Dolby Vision séparément
                dv = any(d in title_upper for d in ["DOLBY.VISION", "DOLBYVISION", "DV", "DOVI", "DOVi"])
                # Éviter les faux positifs (DVD contient DV)
                if dv and "DVD" in title_upper and "DV" in title_upper:
                    # Vérifier si c'est vraiment DV ou juste DVD
                    import re
                    dv = bool(re.search(r'\bDV\b|\bDOVI\b|DOLBY\.?VISION', title_upper))

                # Détecter le codec
                codec = ""
                if any(c in title_upper for c in ["HEVC", "X265", "H265", "H.265"]):
                    codec = "HEVC"
                elif any(c in title_upper for c in ["H264", "X264", "H.264", "AVC"]):
                    codec = "H.264"
                elif "AV1" in title_upper:
                    codec = "AV1"
                elif "MPEG" in title_upper:
                    codec = "MPEG"

                # Détecter l'audio
                audio = ""
                if any(a in title_upper for a in ["ATMOS", "TRUEHD.ATMOS"]):
                    audio = "Atmos"
                elif "TRUEHD" in title_upper:
                    audio = "TrueHD"
                elif any(a in title_upper for a in ["DTS-HD", "DTSHD", "DTS.HD"]):
                    audio = "DTS-HD"
                elif "DTS" in title_upper:
                    audio = "DTS"
                elif any(a in title_upper for a in ["DD5.1", "DD+", "DDP", "AC3"]):
                    audio = "DD"

                results.append({
                    "id": item.get("nzbid") or item.get("id") or "",
                    "title": title,
                    "size_bytes": size_bytes,
                    "size_human": _format_size(size_bytes),
                    "quality": quality,
                    "hdr": hdr,
                    "dv": dv,
                    "codec": codec,
                    "audio": audio,
                    "category": item.get("cattext") or item.get("categoryname") or "",
                    "usenet_date": item.get("usenetage") or item.get("usenetdate") or "",
                    "nzb_url": item.get("getnzb") or "",
                    "details_url": item.get("details") or "",
                    "imdb_id": item.get("imdb") or "",
                })

            return {"results": results}

    except HTTPException:
        raise
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Délai dépassé (API NZB).")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Erreur réseau: {e!s}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e!s}")


def _format_size(size_bytes: int) -> str:
    """Formate une taille en bytes en format lisible."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


@app.get("/api/nzb/download/{nzb_id}")
async def api_nzb_download(nzb_id: str):
    """
    Récupère l'URL de téléchargement pour un NZB spécifique.
    Retourne l'URL du fichier NZB à télécharger.
    """
    settings_data = load_settings()
    api_key = (settings_data.get("nzb_api_key") or "").strip()
    api_user = (settings_data.get("nzb_api_user") or "").strip()

    if not api_key or not api_user:
        raise HTTPException(status_code=400, detail="Clé API et nom d'utilisateur NZB requis.")

    # URL de téléchargement NZB (format: /nzb/?id=...&user=...&api=...)
    nzb_download_url = f"{NZB_API_BASE}/nzb/?id={nzb_id}&user={api_user}&api={api_key}"

    return {
        "ok": True,
        "nzb_id": nzb_id,
        "download_url": nzb_download_url,
    }


# =============================================================================
# Video Player API - Streaming et Persistance
# =============================================================================

PLAYBACK_STATE_FILE = BASE_DIR / "playback_state.json"
WATCH_HISTORY_FILE = BASE_DIR / "watch_history.json"


def _load_playback_states_local() -> dict:
    """Charge les états de lecture depuis le fichier JSON local."""
    if PLAYBACK_STATE_FILE.exists():
        try:
            return json.loads(PLAYBACK_STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _load_playback_states_supabase() -> dict:
    """Charge les états de lecture depuis Supabase."""
    client = get_supabase_client()
    if not client:
        return _load_playback_states_local()

    try:
        response = client.table("playback_state").select("*").execute()
        states = {}
        for row in response.data or []:
            video_path = row.get("video_path")
            if video_path:
                states[video_path] = {
                    "position": row.get("position", 0),
                    "duration": row.get("duration", 0),
                    "volume": row.get("volume", 1.0),
                    "playback_rate": row.get("playback_rate", 1.0),
                    "subtitle_track": row.get("subtitle_track"),
                    "audio_track": row.get("audio_track"),
                    "watched": row.get("watched", False),
                    "last_played": row.get("last_played"),
                }
        return states
    except Exception as e:
        print(f"[SUPABASE] Error loading playback states: {e}. Falling back to local.")
        return _load_playback_states_local()


def load_playback_states() -> dict:
    """Charge les états de lecture."""
    if is_supabase_enabled():
        return _load_playback_states_supabase()
    return _load_playback_states_local()


def _save_playback_states_local(states: dict) -> None:
    """Sauvegarde les états de lecture localement."""
    PLAYBACK_STATE_FILE.write_text(json.dumps(states, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_playback_states_supabase(states: dict) -> None:
    """Sauvegarde les états de lecture dans Supabase."""
    client = get_supabase_client()
    if not client:
        return _save_playback_states_local(states)

    try:
        for video_path, state in states.items():
            row = {
                "video_path": video_path,
                "position": state.get("position", 0),
                "duration": state.get("duration", 0),
                "volume": state.get("volume", 1.0),
                "playback_rate": state.get("playback_rate", 1.0),
                "subtitle_track": state.get("subtitle_track"),
                "audio_track": state.get("audio_track"),
                "watched": state.get("watched", False),
                "last_played": state.get("last_played"),
            }
            # Remove None values
            row = {k: v for k, v in row.items() if v is not None}
            client.table("playback_state").upsert(row, on_conflict="video_path").execute()

        # Also save locally as backup
        _save_playback_states_local(states)

    except Exception as e:
        print(f"[SUPABASE] Error saving playback states: {e}. Falling back to local.")
        _save_playback_states_local(states)


def save_playback_states(states: dict) -> None:
    """Sauvegarde les états de lecture."""
    if is_supabase_enabled():
        _save_playback_states_supabase(states)
    else:
        _save_playback_states_local(states)


def _load_watch_history_local() -> list:
    """Charge l'historique de visionnage local."""
    if WATCH_HISTORY_FILE.exists():
        try:
            return json.loads(WATCH_HISTORY_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _load_watch_history_supabase() -> list:
    """Charge l'historique de visionnage depuis Supabase."""
    client = get_supabase_client()
    if not client:
        return _load_watch_history_local()

    try:
        response = client.table("watch_history").select("*").order("timestamp", desc=True).limit(100).execute()
        history = []
        for row in response.data or []:
            history.append({
                "video_id": row.get("video_path"),
                "position": row.get("position", 0),
                "duration": row.get("duration", 0),
                "watched": row.get("watched", False),
                "timestamp": row.get("timestamp"),
            })
        return history
    except Exception as e:
        print(f"[SUPABASE] Error loading watch history: {e}. Falling back to local.")
        return _load_watch_history_local()


def load_watch_history() -> list:
    """Charge l'historique de visionnage."""
    if is_supabase_enabled():
        return _load_watch_history_supabase()
    return _load_watch_history_local()


def _save_watch_history_local(history: list) -> None:
    """Sauvegarde l'historique de visionnage localement."""
    WATCH_HISTORY_FILE.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_watch_history_supabase(history: list) -> None:
    """Sauvegarde l'historique de visionnage dans Supabase."""
    client = get_supabase_client()
    if not client:
        return _save_watch_history_local(history)

    try:
        # Only insert new entries (history is ordered by timestamp desc)
        if history:
            # Insert the most recent entry
            entry = history[0]
            row = {
                "video_path": entry.get("video_id"),
                "position": entry.get("position", 0),
                "duration": entry.get("duration", 0),
                "watched": entry.get("watched", False),
                "timestamp": entry.get("timestamp"),
            }
            row = {k: v for k, v in row.items() if v is not None}
            client.table("watch_history").insert(row).execute()

        # Also save locally as backup
        _save_watch_history_local(history)

    except Exception as e:
        print(f"[SUPABASE] Error saving watch history: {e}. Falling back to local.")
        _save_watch_history_local(history)


def save_watch_history(history: list) -> None:
    """Sauvegarde l'historique de visionnage."""
    if is_supabase_enabled():
        _save_watch_history_supabase(history)
    else:
        _save_watch_history_local(history)


@app.get("/api/playback/{video_id:path}")
async def get_playback_state(video_id: str):
    """Récupère l'état de lecture d'une vidéo."""
    states = load_playback_states()
    state = states.get(video_id, {})
    return {
        "video_id": video_id,
        "position": state.get("position", 0),
        "duration": state.get("duration", 0),
        "volume": state.get("volume", 1.0),
        "playback_rate": state.get("playback_rate", 1.0),
        "subtitle_track": state.get("subtitle_track"),
        "audio_track": state.get("audio_track"),
        "watched": state.get("watched", False),
        "last_played": state.get("last_played"),
    }


@app.post("/api/playback/{video_id:path}")
async def save_playback_state(
    video_id: str,
    position: float = Body(0),
    duration: float = Body(0),
    volume: float = Body(1.0),
    playback_rate: float = Body(1.0),
    subtitle_track: int | None = Body(None),
    audio_track: int | None = Body(None),
    watched: bool = Body(False),
):
    """Sauvegarde l'état de lecture d'une vidéo."""
    states = load_playback_states()

    now = datetime.now(timezone.utc).isoformat()
    states[video_id] = {
        "position": position,
        "duration": duration,
        "volume": volume,
        "playback_rate": playback_rate,
        "subtitle_track": subtitle_track,
        "audio_track": audio_track,
        "watched": watched,
        "last_played": now,
    }

    save_playback_states(states)

    # Ajouter à l'historique
    history = load_watch_history()
    # Supprimer l'entrée existante si présente
    history = [h for h in history if h.get("video_id") != video_id]
    # Ajouter en tête
    history.insert(0, {
        "video_id": video_id,
        "position": position,
        "duration": duration,
        "watched": watched,
        "timestamp": now,
    })
    # Garder les 100 derniers
    history = history[:100]
    save_watch_history(history)

    return {"ok": True}


@app.delete("/api/playback/{video_id:path}")
async def delete_playback_state(video_id: str):
    """Supprime l'état de lecture d'une vidéo."""
    states = load_playback_states()
    if video_id in states:
        del states[video_id]
        save_playback_states(states)
    return {"ok": True}


@app.get("/api/watch-history")
async def get_watch_history(limit: int = Query(20)):
    """Récupère l'historique de visionnage."""
    history = load_watch_history()
    return {"history": history[:limit]}


@app.post("/api/mark-watched/{video_id:path}")
async def mark_as_watched(video_id: str, watched: bool = Body(True)):
    """Marque une vidéo comme vue ou non vue."""
    states = load_playback_states()
    if video_id not in states:
        states[video_id] = {}
    states[video_id]["watched"] = watched
    states[video_id]["last_played"] = datetime.now(timezone.utc).isoformat()
    save_playback_states(states)
    return {"ok": True, "watched": watched}


@app.get("/api/continue-watching")
async def get_continue_watching(limit: int = Query(20)):
    """
    Récupère les vidéos en cours de visionnage (position > 0, pas terminées).
    Retourne les fichiers avec leur position de lecture pour "Continuer à regarder".
    """
    states = load_playback_states()
    inv = load_inventory()
    files = inv.get("files") or []

    # Créer un index des fichiers par path
    files_by_path = {f.get("path"): f for f in files}

    continue_watching = []
    for video_path, state in states.items():
        position = state.get("position", 0)
        duration = state.get("duration", 0)
        watched = state.get("watched", False)

        # Inclure seulement les vidéos en cours (position > 10s, pas terminées < 95%)
        if position > 10 and duration > 0 and not watched:
            progress_percent = (position / duration) * 100
            if progress_percent < 95:
                file_info = files_by_path.get(video_path)
                if file_info:
                    continue_watching.append({
                        "path": video_path,
                        "position": position,
                        "duration": duration,
                        "progress_percent": round(progress_percent, 1),
                        "last_played": state.get("last_played"),
                        "file": file_info,
                    })

    # Trier par dernière lecture (plus récent en premier)
    continue_watching.sort(
        key=lambda x: x.get("last_played") or "",
        reverse=True
    )

    return {"items": continue_watching[:limit]}


@app.get("/api/recently-added")
async def get_recently_added(limit: int = Query(20)):
    """
    Récupère les films récemment ajoutés à la bibliothèque.
    Triés par date de scan (plus récent en premier).
    """
    inv = load_inventory()
    files = inv.get("files") or []

    # Trier par date de création/scan (si disponible) ou par ordre inverse
    # Les fichiers les plus récents sont généralement à la fin de la liste
    # On utilise scanned_at ou created_at s'ils existent
    def get_scan_date(f):
        return f.get("scanned_at") or f.get("created_at") or f.get("mtime") or ""

    sorted_files = sorted(files, key=get_scan_date, reverse=True)

    return {"items": sorted_files[:limit]}


# Formats que les navigateurs peuvent lire nativement
BROWSER_NATIVE_CONTAINERS = {"mp4", "m4v", "webm", "ogg", "ogv"}
BROWSER_NATIVE_VIDEO_CODECS = {"h264", "avc1", "vp8", "vp9", "av1"}
BROWSER_NATIVE_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis", "flac"}


def needs_transcoding(video_path: Path) -> tuple[bool, dict]:
    """
    Vérifie si une vidéo nécessite un transcodage pour être lue dans le navigateur.
    Retourne (needs_transcode, video_info).
    """
    import subprocess

    ext = video_path.suffix.lower().lstrip(".")

    # Vérifier le conteneur
    if ext not in BROWSER_NATIVE_CONTAINERS:
        return True, {"reason": f"container_{ext}"}

    # Analyser les codecs avec ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return True, {"reason": "ffprobe_error"}

        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        if streams:
            video_codec = streams[0].get("codec_name", "").lower()
            # HEVC/H.265 n'est pas supporté sur tous les navigateurs
            if video_codec in ["hevc", "h265", "vc1", "mpeg2video", "mpeg4", "msmpeg4v3"]:
                return True, {"reason": f"codec_{video_codec}", "codec": video_codec}

        return False, {"reason": "native"}

    except Exception as e:
        # En cas d'erreur, transcoder par sécurité
        return True, {"reason": f"error_{str(e)}"}


def get_video_duration(video_path: Path) -> float:
    """Récupère la durée de la vidéo en secondes."""
    import subprocess
    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_format", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return float(data.get("format", {}).get("duration", 0))
    except:
        pass
    return 0


@app.get("/api/video/stream")
async def stream_video(
    request: Request,
    path: str = Query(...),
    start: float = Query(0, description="Position de départ en secondes"),
    quality: str = Query("720p", description="Qualité: original, 1080p, 720p, 480p, 360p"),
):
    """
    Stream une vidéo avec transcoding automatique si nécessaire.
    Transcode en H.264/AAC MP4 pour les formats non supportés par le navigateur.
    Supporte le hardware acceleration (VideoToolbox, NVENC, VAAPI).
    """
    import mimetypes
    import shutil

    print(f"[STREAM] Request for: {path}, start={start}, quality={quality}")

    video_path = Path(path)
    if not video_path.exists():
        # Diagnostic détaillé
        parent = video_path.parent
        print(f"[STREAM] File not found: {path}")
        print(f"[STREAM] Parent exists: {parent.exists()}")
        if parent.exists():
            print(f"[STREAM] Parent contents: {list(parent.iterdir())[:5]}...")

        # Vérifier si c'est un problème de volume non monté
        parts = video_path.parts
        if len(parts) > 2 and parts[1] == "Volumes":
            volume_path = Path("/Volumes") / parts[2]
            if not volume_path.exists():
                raise HTTPException(
                    status_code=404,
                    detail=f"Volume non monté: {volume_path}. Montez le NAS et réessayez."
                )

        raise HTTPException(status_code=404, detail=f"Fichier non trouvé: {path}")

    needs_transcode, info = needs_transcoding(video_path)

    # Si qualité != original, forcer le transcoding pour redimensionner
    if quality != "original" and quality in QUALITY_PRESETS:
        needs_transcode = True
        info["reason"] = f"quality_{quality}"

    print(f"[STREAM] Needs transcode: {needs_transcode}, info: {info}")

    if not needs_transcode:
        # Stream direct sans transcoding
        mime_type, _ = mimetypes.guess_type(str(video_path))
        if not mime_type or not mime_type.startswith("video"):
            mime_type = "video/mp4"
        print(f"[STREAM] Direct stream with mime type: {mime_type}")
        return FileResponse(
            path=str(video_path),
            media_type=mime_type,
            filename=video_path.name,
        )

    # Vérifier que FFmpeg est disponible
    if not shutil.which("ffmpeg"):
        print("[STREAM] FFmpeg not found!")
        raise HTTPException(status_code=500, detail="FFmpeg n'est pas installé sur le serveur")

    # Transcoding nécessaire - utiliser FFmpeg avec qualité et HW acceleration
    print(f"[STREAM] Starting transcode for: {video_path}")
    return await transcode_video(video_path, start, quality, request)


async def transcode_video_simple(video_path: Path, start: float, quality: str):
    """
    Mode de secours : transcodage simple sans accélération hardware.
    Utilisé quand le transcodage avancé échoue.
    """
    import asyncio

    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["720p"])
    target_height = preset.get("height", 720)
    target_bitrate = preset.get("bitrate", "4M")

    # Commande FFmpeg ultra-simple
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-nostdin",
    ]

    if start > 0:
        ffmpeg_cmd.extend(["-ss", str(start)])

    ffmpeg_cmd.extend([
        "-i", str(video_path),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "28",
        "-vf", f"scale=-2:{target_height or 720},format=yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-ac", "2",
        "-movflags", "frag_keyframe+empty_moov+default_base_moof",
        "-f", "mp4",
        "pipe:1"
    ])

    print(f"[TRANSCODE-SIMPLE] Command: {' '.join(ffmpeg_cmd)}")

    async def generate():
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        bytes_sent = 0
        try:
            while True:
                chunk = await process.stdout.read(64 * 1024)
                if not chunk:
                    break
                bytes_sent += len(chunk)
                yield chunk

            await process.wait()
            stderr = await process.stderr.read()
            if stderr:
                print(f"[TRANSCODE-SIMPLE] stderr: {stderr.decode('utf-8', errors='ignore')}")
            print(f"[TRANSCODE-SIMPLE] Done: {bytes_sent:,} bytes")

        except Exception as e:
            print(f"[TRANSCODE-SIMPLE] Error: {e}")
            process.kill()
            raise
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    return StreamingResponse(
        generate(),
        media_type="video/mp4",
        headers={"Accept-Ranges": "none", "Cache-Control": "no-cache"}
    )


async def transcode_video(video_path: Path, start: float, quality: str, request: Request):
    """
    Transcode une vidéo en temps réel avec FFmpeg.
    Produit un stream MP4 avec H.264/AAC compatible navigateur.
    Supporte l'accélération matérielle complète (décodage + encodage GPU).
    Gère le HDR avec tone mapping vers SDR.
    """
    import asyncio
    import os

    # Variable d'environnement pour forcer le mode simple
    if os.environ.get("TRANSCODE_SIMPLE", "").lower() in ("1", "true", "yes"):
        print("[TRANSCODE] Using SIMPLE mode (TRANSCODE_SIMPLE=1)")
        return await transcode_video_simple(video_path, start, quality)

    # Récupérer le preset de qualité
    preset = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["720p"])
    target_height = preset.get("height")
    target_bitrate = preset.get("bitrate")

    # Détecter les encodeurs/décodeurs hardware disponibles
    hw_encoders = get_hw_encoders()

    # Détecter les propriétés de la vidéo (résolution, HDR, etc.)
    video_props = detect_video_properties(video_path)
    is_hdr = video_props.get("is_hdr", False)
    is_dolby_vision = video_props.get("is_dolby_vision", False)
    source_height = video_props.get("height", 0)
    source_codec = video_props.get("codec", "")

    # Pour qualité "original", calculer un bitrate adapté à la résolution source
    if quality == "original" and not target_bitrate:
        if source_height >= 2160:
            target_bitrate = "25M"  # 4K
        elif source_height >= 1440:
            target_bitrate = "15M"  # 1440p
        elif source_height >= 1080:
            target_bitrate = "10M"  # 1080p
        else:
            target_bitrate = "6M"   # 720p et moins

    print(f"[TRANSCODE] Source: {video_props['width']}x{source_height}, codec={source_codec}, HDR={is_hdr}, DV={is_dolby_vision}")

    # Construire la commande FFmpeg
    ffmpeg_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "warning",
        "-nostdin",
    ]

    # === DÉCODAGE HARDWARE ===
    use_hw_decode = False
    hw_decode_method = None

    # NVIDIA CUDA/CUVID - décodage GPU pour HEVC/H264
    if hw_encoders.get("cuvid") and hw_encoders.get("nvenc"):
        if source_codec in ["hevc", "h265", "h264", "avc"]:
            ffmpeg_cmd.extend(["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"])
            use_hw_decode = True
            hw_decode_method = "cuda"
            print(f"[TRANSCODE] Using NVIDIA CUDA hardware decoding")

    # macOS VideoToolbox
    elif hw_encoders.get("videotoolbox_dec") and hw_encoders.get("videotoolbox"):
        if source_codec in ["hevc", "h265", "h264", "avc"]:
            ffmpeg_cmd.extend(["-hwaccel", "videotoolbox"])
            use_hw_decode = True
            hw_decode_method = "videotoolbox"
            print(f"[TRANSCODE] Using VideoToolbox hardware decoding")

    # Linux VAAPI
    elif hw_encoders.get("vaapi_dec") and hw_encoders.get("vaapi"):
        if source_codec in ["hevc", "h265", "h264", "avc"]:
            ffmpeg_cmd.extend(["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"])
            use_hw_decode = True
            hw_decode_method = "vaapi"
            print(f"[TRANSCODE] Using VAAPI hardware decoding")

    # Intel QuickSync
    elif hw_encoders.get("qsv_dec") and hw_encoders.get("qsv"):
        if source_codec in ["hevc", "h265", "h264", "avc"]:
            ffmpeg_cmd.extend(["-hwaccel", "qsv"])
            use_hw_decode = True
            hw_decode_method = "qsv"
            print(f"[TRANSCODE] Using Intel QSV hardware decoding")

    # Position de départ si spécifiée (avant -i pour seeking rapide)
    if start > 0:
        ffmpeg_cmd.extend(["-ss", str(start)])

    ffmpeg_cmd.extend(["-i", str(video_path)])

    # === FILTRES VIDÉO ===
    video_filters = []

    # Pour le décodage CUDA, on doit télécharger les frames du GPU si on applique des filtres CPU
    needs_hwdownload = hw_decode_method == "cuda"

    # Tone mapping HDR → SDR (nécessaire pour l'affichage navigateur)
    if is_hdr:
        # Désactiver le décodage hardware pour HDR car le tone mapping nécessite des filtres CPU
        if needs_hwdownload:
            video_filters.append("hwdownload")
            video_filters.append("format=nv12")
            needs_hwdownload = False  # Déjà géré

        # Essayer zscale (meilleure qualité) sinon tonemap basique
        if hw_encoders.get("zscale_filter"):
            video_filters.extend([
                "zscale=t=linear:npl=100",
                "format=gbrpf32le",
                "zscale=p=bt709",
                "tonemap=hable:desat=0",
                "zscale=t=bt709:m=bt709:r=tv",
                "format=yuv420p"
            ])
            print(f"[TRANSCODE] Using zscale HDR tone mapping (high quality)")
        elif hw_encoders.get("tonemap_filter"):
            # Tone mapping basique - conversion colorspace + tonemap
            video_filters.extend([
                "format=gbrpf32le",
                "tonemap=hable:param=1.0:desat=0",
                "format=yuv420p"
            ])
            print(f"[TRANSCODE] Using basic HDR tone mapping")
        else:
            # Pas de filtre de tone mapping disponible
            # Solution de repli : convertir en 8-bit et ajuster les niveaux manuellement
            # eq=gamma=0.9:saturation=1.2 aide à compenser le manque de tone mapping
            video_filters.extend([
                "format=yuv420p",
                "eq=gamma=0.85:saturation=1.3:contrast=1.1"  # Compensation basique HDR
            ])
            print(f"[TRANSCODE] WARNING: No HDR tone mapping filters (zscale/tonemap) available!")
            print(f"[TRANSCODE] Using basic gamma/saturation compensation - colors won't be perfect")
            print(f"[TRANSCODE] For better quality, install FFmpeg with zscale: brew install ffmpeg --with-zimg")

    # Redimensionnement si nécessaire
    if target_height and source_height > 0 and source_height != target_height:
        if needs_hwdownload:
            # CUDA scaling sur GPU
            video_filters.append(f"scale_cuda=-2:{target_height}")
        else:
            video_filters.append(f"scale=-2:{target_height}")

    # Assurer format pixel compatible (si pas déjà fait par tone mapping)
    if not is_hdr and not any("format=yuv420p" in f for f in video_filters):
        video_filters.append("format=yuv420p")

    # Appliquer les filtres
    if video_filters:
        filter_str = ",".join(video_filters)
        ffmpeg_cmd.extend(["-vf", filter_str])
        print(f"[TRANSCODE] Video filters: {filter_str}")

    # === ENCODAGE VIDÉO ===
    if hw_encoders.get("nvenc"):
        # NVIDIA NVENC - excellent pour 4K
        ffmpeg_cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",  # p1(fastest) to p7(slowest/best)
            "-profile:v", "high",
            "-rc", "vbr",
            "-rc-lookahead", "32",
        ])
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", target_bitrate, "-maxrate", target_bitrate, "-bufsize", f"{int(target_bitrate.replace('M', '')) * 2}M"])
        else:
            ffmpeg_cmd.extend(["-cq", "23"])
        print(f"[TRANSCODE] Using NVENC hardware encoder")

    elif hw_encoders.get("videotoolbox"):
        # macOS VideoToolbox
        ffmpeg_cmd.extend([
            "-c:v", "h264_videotoolbox",
            "-profile:v", "high",
            "-level", "5.1",  # Support 4K
        ])
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", target_bitrate])
        else:
            ffmpeg_cmd.extend(["-q:v", "65"])
        print(f"[TRANSCODE] Using VideoToolbox hardware encoder")

    elif hw_encoders.get("vaapi"):
        # Linux VAAPI (Intel/AMD)
        ffmpeg_cmd.extend([
            "-c:v", "h264_vaapi",
            "-profile:v", "high",
        ])
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", target_bitrate])
        else:
            ffmpeg_cmd.extend(["-qp", "23"])
        print(f"[TRANSCODE] Using VAAPI hardware encoder")

    elif hw_encoders.get("qsv"):
        # Intel QuickSync
        ffmpeg_cmd.extend([
            "-c:v", "h264_qsv",
            "-preset", "medium",
            "-profile:v", "high",
        ])
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", target_bitrate])
        else:
            ffmpeg_cmd.extend(["-global_quality", "23"])
        print(f"[TRANSCODE] Using Intel QSV hardware encoder")

    else:
        # Software encoding (libx264) - fallback
        ffmpeg_cmd.extend([
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-profile:v", "high",
            "-level", "5.1",
        ])
        if target_bitrate:
            ffmpeg_cmd.extend(["-b:v", target_bitrate])
        else:
            ffmpeg_cmd.extend(["-crf", "23"])
        print(f"[TRANSCODE] Using software encoder (libx264)")

    # Audio : convertir en AAC stéréo (compatible navigateur)
    # Pour FLAC/DTS/TrueHD, on force la conversion
    ffmpeg_cmd.extend([
        "-c:a", "aac",
        "-b:a", "192k",  # Qualité audio améliorée
        "-ar", "48000",   # Fréquence standard pour vidéo
        "-ac", "2",       # Stéréo (downmix si multicanal)
    ])

    # MP4 fragmenté pour streaming progressif (permet lecture avant fin du téléchargement)
    ffmpeg_cmd.extend([
        "-movflags", "frag_keyframe+empty_moov+default_base_moof+faststart",
        "-f", "mp4",
        "pipe:1"  # Sortie vers stdout
    ])

    print(f"[TRANSCODE] Starting: quality={quality}, target={target_height}p, bitrate={target_bitrate}, HDR={is_hdr}, DV={is_dolby_vision}")
    print(f"[TRANSCODE] HW decode={hw_decode_method}, filters={video_filters}")
    print(f"[TRANSCODE] Command: {' '.join(ffmpeg_cmd)}")

    async def generate():
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stderr_chunks = []

        async def read_stderr():
            """Lire stderr en arrière-plan pour le logging."""
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                stderr_chunks.append(line.decode('utf-8', errors='ignore'))

        try:
            # Démarrer la lecture de stderr en parallèle
            stderr_task = asyncio.create_task(read_stderr())

            bytes_sent = 0
            first_chunk_timeout = 10.0  # Timeout pour le premier chunk (FFmpeg doit démarrer)

            # Attendre le premier chunk avec timeout
            try:
                first_chunk = await asyncio.wait_for(
                    process.stdout.read(32 * 1024),
                    timeout=first_chunk_timeout
                )
                if first_chunk:
                    bytes_sent += len(first_chunk)
                    yield first_chunk
            except asyncio.TimeoutError:
                # FFmpeg n'a rien produit - probablement une erreur
                process.kill()
                await stderr_task
                stderr_output = ''.join(stderr_chunks)
                print(f"[TRANSCODE] TIMEOUT - FFmpeg stderr: {stderr_output}")
                raise Exception(f"FFmpeg timeout: {stderr_output[:500]}")

            # Continuer à lire le reste
            while True:
                chunk = await process.stdout.read(64 * 1024)  # 64KB chunks pour performance
                if not chunk:
                    break
                bytes_sent += len(chunk)
                yield chunk

            # Attendre la fin
            await process.wait()
            await stderr_task

            stderr_output = ''.join(stderr_chunks)
            if stderr_output:
                print(f"[TRANSCODE] FFmpeg stderr: {stderr_output}")

            if bytes_sent == 0:
                print(f"[TRANSCODE] ERROR: No data produced! stderr: {stderr_output}")
            else:
                print(f"[TRANSCODE] Done. Sent {bytes_sent:,} bytes ({bytes_sent / 1024 / 1024:.1f} MB)")

        except asyncio.CancelledError:
            print("[TRANSCODE] Cancelled by client")
            process.kill()
            await process.wait()
            raise
        except Exception as e:
            print(f"[TRANSCODE] Error: {e}")
            process.kill()
            await process.wait()
            raise
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    return StreamingResponse(
        generate(),
        media_type="video/mp4",
        headers={
            "Content-Disposition": f"inline; filename=\"{video_path.stem}_transcoded.mp4\"",
            "Accept-Ranges": "none",  # Pas de seeking sur le stream transcodé
            "Cache-Control": "no-cache",
        }
    )


def detect_hw_encoders() -> dict:
    """Détecte les encodeurs et décodeurs hardware disponibles."""
    import subprocess
    import platform

    encoders = {
        "videotoolbox": False,  # macOS
        "nvenc": False,         # NVIDIA
        "vaapi": False,         # Linux Intel/AMD
        "qsv": False,           # Intel QuickSync
        # Décodeurs hardware
        "cuvid": False,         # NVIDIA CUDA decoder
        "videotoolbox_dec": False,  # macOS decoder
        "vaapi_dec": False,     # Linux decoder
        "qsv_dec": False,       # Intel decoder
    }

    try:
        # Vérifier les encodeurs
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout.lower()

        if "h264_videotoolbox" in output:
            encoders["videotoolbox"] = True
        if "h264_nvenc" in output:
            encoders["nvenc"] = True
        if "h264_vaapi" in output:
            encoders["vaapi"] = True
        if "h264_qsv" in output:
            encoders["qsv"] = True

        # Vérifier les décodeurs hardware
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True,
            text=True,
            timeout=10
        )
        dec_output = result.stdout.lower()

        if "hevc_cuvid" in dec_output or "h264_cuvid" in dec_output:
            encoders["cuvid"] = True
        if "hevc_videotoolbox" in dec_output or "h264_videotoolbox" in dec_output:
            encoders["videotoolbox_dec"] = True
        if "hevc_vaapi" in dec_output or "h264_vaapi" in dec_output:
            encoders["vaapi_dec"] = True
        if "hevc_qsv" in dec_output or "h264_qsv" in dec_output:
            encoders["qsv_dec"] = True

        # Vérifier les filtres disponibles (pour tone mapping HDR)
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            timeout=10
        )
        filters_output = result.stdout.lower()
        encoders["tonemap_filter"] = "tonemap" in filters_output
        encoders["zscale_filter"] = "zscale" in filters_output

    except Exception as e:
        print(f"[HW] Error detecting encoders: {e}")

    return encoders


def detect_video_properties(video_path: Path) -> dict:
    """Détecte les propriétés de la vidéo (résolution, HDR, codec)."""
    import subprocess

    props = {
        "width": 0,
        "height": 0,
        "codec": "",
        "is_hdr": False,
        "is_dolby_vision": False,
        "color_transfer": "",
        "color_primaries": "",
        "bit_depth": 8,
    }

    try:
        cmd = [
            "ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", "-select_streams", "v:0", str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return props

        data = json.loads(result.stdout)
        streams = data.get("streams", [])

        if streams:
            stream = streams[0]
            props["width"] = stream.get("width", 0)
            props["height"] = stream.get("height", 0)
            props["codec"] = stream.get("codec_name", "").lower()
            props["color_transfer"] = stream.get("color_transfer", "")
            props["color_primaries"] = stream.get("color_primaries", "")

            # Détecter le bit depth
            pix_fmt = stream.get("pix_fmt", "")
            if "10" in pix_fmt or "12" in pix_fmt or "16" in pix_fmt:
                props["bit_depth"] = 10

            # Détecter HDR (PQ = HDR10, HLG = HLG, ARIB = HLG japonais)
            hdr_transfers = ["smpte2084", "arib-std-b67", "bt2020-10", "bt2020-12"]
            if props["color_transfer"] in hdr_transfers:
                props["is_hdr"] = True

            # Détecter Dolby Vision via side_data
            side_data = stream.get("side_data_list", [])
            for sd in side_data:
                if sd.get("side_data_type") == "DOVI configuration record":
                    props["is_dolby_vision"] = True
                    props["is_hdr"] = True
                    break

        print(f"[VIDEO] Properties: {props['width']}x{props['height']}, codec={props['codec']}, HDR={props['is_hdr']}, DV={props['is_dolby_vision']}")

    except Exception as e:
        print(f"[VIDEO] Error detecting properties: {e}")

    return props


# Cache des encodeurs détectés
_hw_encoders_cache = None

def get_hw_encoders() -> dict:
    global _hw_encoders_cache
    if _hw_encoders_cache is None:
        _hw_encoders_cache = detect_hw_encoders()
        print(f"[HW] Detected capabilities: {_hw_encoders_cache}")
    return _hw_encoders_cache

def reset_hw_encoders_cache():
    """Réinitialise le cache des capacités hardware."""
    global _hw_encoders_cache
    _hw_encoders_cache = None
    print("[HW] Cache cleared, will re-detect on next request")


# Presets de qualité pour le transcoding
QUALITY_PRESETS = {
    "original": {"height": None, "bitrate": None, "label": "Original"},
    "1080p": {"height": 1080, "bitrate": "8M", "label": "1080p (8 Mbps)"},
    "720p": {"height": 720, "bitrate": "4M", "label": "720p (4 Mbps)"},
    "480p": {"height": 480, "bitrate": "2M", "label": "480p (2 Mbps)"},
    "360p": {"height": 360, "bitrate": "1M", "label": "360p (1 Mbps)"},
}


@app.get("/api/video/test-transcode")
async def test_transcode():
    """
    Teste si FFmpeg est disponible et retourne les capacités.
    """
    import subprocess
    import shutil

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        return {"ok": False, "error": "FFmpeg non trouvé dans PATH", "ffmpeg_path": None}

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        version_line = result.stdout.split('\n')[0] if result.stdout else "Unknown"

        hw_encoders = get_hw_encoders()

        # Déterminer le meilleur encodeur
        best_encoder = "software"
        if hw_encoders.get("nvenc"):
            best_encoder = "nvenc"
        elif hw_encoders.get("videotoolbox"):
            best_encoder = "videotoolbox"
        elif hw_encoders.get("vaapi"):
            best_encoder = "vaapi"
        elif hw_encoders.get("qsv"):
            best_encoder = "qsv"

        # Déterminer le meilleur décodeur
        best_decoder = "software"
        if hw_encoders.get("cuvid"):
            best_decoder = "cuda"
        elif hw_encoders.get("videotoolbox_dec"):
            best_decoder = "videotoolbox"
        elif hw_encoders.get("vaapi_dec"):
            best_decoder = "vaapi"
        elif hw_encoders.get("qsv_dec"):
            best_decoder = "qsv"

        # Capacités HDR
        hdr_capable = hw_encoders.get("tonemap_filter", False) or hw_encoders.get("zscale_filter", False)

        return {
            "ok": True,
            "ffmpeg_path": ffmpeg_path,
            "version": version_line,
            "can_transcode": True,
            "hw_encoders": hw_encoders,
            "best_encoder": best_encoder,
            "best_decoder": best_decoder,
            "hdr_tonemapping": hdr_capable,
            "quality_presets": list(QUALITY_PRESETS.keys()),
            "supports_4k_realtime": best_encoder != "software" and best_decoder != "software",
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "ffmpeg_path": ffmpeg_path}


@app.post("/api/video/refresh-hw-capabilities")
async def refresh_hw_capabilities():
    """
    Réinitialise le cache des capacités hardware et re-détecte.
    Utile après installation de nouveaux drivers GPU.
    """
    reset_hw_encoders_cache()
    hw_encoders = get_hw_encoders()
    return {
        "ok": True,
        "message": "Hardware capabilities refreshed",
        "hw_encoders": hw_encoders,
    }


@app.get("/api/video/transcode-info")
async def get_transcode_info(path: str = Query(...)):
    """
    Retourne les informations de transcodage pour une vidéo.
    Utile pour le frontend pour savoir si le seeking sera disponible.
    """
    video_path = Path(path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Fichier vidéo non trouvé")

    needs_transcode, info = needs_transcoding(video_path)
    duration = get_video_duration(video_path)

    return {
        "path": path,
        "needs_transcoding": needs_transcode,
        "info": info,
        "duration": duration,
        "seeking_supported": not needs_transcode,  # Seeking natif uniquement sans transcodage
    }


@app.get("/api/video/info")
async def get_video_info(path: str = Query(...)):
    """
    Récupère les informations détaillées d'une vidéo pour le player.
    Inclut les pistes audio, sous-titres, chapitres si disponibles.
    """
    import subprocess

    video_path = Path(path)
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Fichier vidéo non trouvé")

    # Utiliser ffprobe pour obtenir les infos détaillées
    try:
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            "-show_chapters",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="Erreur ffprobe")

        data = json.loads(result.stdout)

        # Extraire les infos pertinentes
        format_info = data.get("format", {})
        streams = data.get("streams", [])
        chapters = data.get("chapters", [])

        # Pistes vidéo
        video_tracks = []
        audio_tracks = []
        subtitle_tracks = []

        for i, stream in enumerate(streams):
            codec_type = stream.get("codec_type")
            if codec_type == "video":
                video_tracks.append({
                    "index": stream.get("index"),
                    "codec": stream.get("codec_name"),
                    "width": stream.get("width"),
                    "height": stream.get("height"),
                    "fps": stream.get("r_frame_rate"),
                    "bit_rate": stream.get("bit_rate"),
                })
            elif codec_type == "audio":
                tags = stream.get("tags", {})
                audio_tracks.append({
                    "index": stream.get("index"),
                    "codec": stream.get("codec_name"),
                    "channels": stream.get("channels"),
                    "sample_rate": stream.get("sample_rate"),
                    "language": tags.get("language", "und"),
                    "title": tags.get("title", ""),
                })
            elif codec_type == "subtitle":
                tags = stream.get("tags", {})
                subtitle_tracks.append({
                    "index": stream.get("index"),
                    "codec": stream.get("codec_name"),
                    "language": tags.get("language", "und"),
                    "title": tags.get("title", ""),
                    "forced": stream.get("disposition", {}).get("forced", 0) == 1,
                })

        # Chapitres
        chapter_list = []
        for ch in chapters:
            chapter_list.append({
                "id": ch.get("id"),
                "start": float(ch.get("start_time", 0)),
                "end": float(ch.get("end_time", 0)),
                "title": ch.get("tags", {}).get("title", f"Chapitre {ch.get('id', 0) + 1}"),
            })

        return {
            "path": str(video_path),
            "filename": video_path.name,
            "duration": float(format_info.get("duration", 0)),
            "size_bytes": int(format_info.get("size", 0)),
            "bit_rate": int(format_info.get("bit_rate", 0)),
            "format": format_info.get("format_name"),
            "video_tracks": video_tracks,
            "audio_tracks": audio_tracks,
            "subtitle_tracks": subtitle_tracks,
            "chapters": chapter_list,
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Timeout ffprobe")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e!s}")


# =============================================================================
# Folder Management & Merge Endpoints
# =============================================================================

@app.get("/api/folders/list")
async def list_folders():
    """
    Liste tous les dossiers dans les répertoires vidéo et séries.
    """
    folders = []

    try:
        user_settings = load_settings()

        # Lister les dossiers vidéo
        video_path_str = user_settings.get("video_path", "")
        if video_path_str:
            video_path = Path(video_path_str)
            if video_path.exists() and video_path.is_dir():
                for folder in video_path.iterdir():
                    if folder.is_dir() and not folder.name.startswith('.'):
                        # Compter les fichiers vidéo
                        video_files = list(folder.glob("**/*.mkv")) + list(folder.glob("**/*.mp4")) + \
                                      list(folder.glob("**/*.avi")) + list(folder.glob("**/*.mov"))

                        # Calculer la taille totale
                        total_size = sum(f.stat().st_size for f in video_files if f.exists())

                        folders.append({
                            "path": str(folder),
                            "name": folder.name,
                            "type": "video",
                            "fileCount": len(video_files),
                            "size": total_size,
                        })

        # Lister les dossiers séries
        series_path_str = user_settings.get("series_path", "")
        if series_path_str:
            series_path = Path(series_path_str)
            if series_path.exists() and series_path.is_dir():
                for folder in series_path.iterdir():
                    if folder.is_dir() and not folder.name.startswith('.'):
                        # Compter les fichiers vidéo
                        video_files = list(folder.glob("**/*.mkv")) + list(folder.glob("**/*.mp4")) + \
                                      list(folder.glob("**/*.avi")) + list(folder.glob("**/*.mov"))

                        # Calculer la taille totale
                        total_size = sum(f.stat().st_size for f in video_files if f.exists())

                        folders.append({
                            "path": str(folder),
                            "name": folder.name,
                            "type": "series",
                            "fileCount": len(video_files),
                            "size": total_size,
                        })

        return {
            "ok": True,
            "folders": folders,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du listage des dossiers: {e!s}")


@app.post("/api/folders/merge")
async def merge_folders(data: dict = Body(...)):
    """
    Fusionne plusieurs dossiers source vers un dossier cible.
    Déplace tous les fichiers des sources vers la cible, puis supprime les dossiers sources vides.
    """
    target = data.get("target")
    sources = data.get("sources", [])

    if not target or not sources:
        raise HTTPException(status_code=400, detail="target et sources sont requis")

    target_path = Path(target)
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=404, detail="Dossier cible introuvable")

    moved_files = 0
    errors = []

    try:
        for source_str in sources:
            source_path = Path(source_str)

            if not source_path.exists() or not source_path.is_dir():
                errors.append(f"Dossier source introuvable: {source_str}")
                continue

            if source_path == target_path:
                errors.append(f"Le dossier source ne peut pas être le même que la cible: {source_str}")
                continue

            # Déplacer tous les fichiers du source vers le target
            for item in source_path.rglob("*"):
                if item.is_file():
                    # Calculer le chemin relatif
                    rel_path = item.relative_to(source_path)
                    dest_path = target_path / rel_path

                    # Créer les dossiers parents si nécessaire
                    dest_path.parent.mkdir(parents=True, exist_ok=True)

                    # Si le fichier existe déjà dans la destination, ajouter un suffixe
                    if dest_path.exists():
                        counter = 1
                        stem = dest_path.stem
                        suffix = dest_path.suffix
                        while dest_path.exists():
                            dest_path = dest_path.parent / f"{stem}_{counter}{suffix}"
                            counter += 1

                    try:
                        shutil.move(str(item), str(dest_path))
                        moved_files += 1
                    except Exception as e:
                        errors.append(f"Erreur lors du déplacement de {item}: {e!s}")

            # Supprimer le dossier source s'il est vide
            try:
                # Supprimer les dossiers vides récursivement
                for dirpath, dirnames, filenames in os.walk(str(source_path), topdown=False):
                    if not filenames and not dirnames:
                        os.rmdir(dirpath)

                # Supprimer le dossier source principal s'il est vide
                if source_path.exists() and not any(source_path.iterdir()):
                    source_path.rmdir()
            except Exception as e:
                errors.append(f"Erreur lors de la suppression du dossier source {source_path}: {e!s}")

        return {
            "ok": True,
            "moved_files": moved_files,
            "errors": errors if errors else None,
            "message": f"{moved_files} fichiers déplacés avec succès",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la fusion: {e!s}")
