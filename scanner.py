"""
Scan vidéos: extraction maximale des métadonnées ffprobe, sauvegarde en inventaire persistant.

Le scan parcourt récursivement TOUS les sous-dossiers du chemin d'origine (rglob("*")),
pas seulement le premier niveau : chaque fichier vidéo trouvé à n'importe quelle profondeur
est inclus dans l'inventaire.

Mode asynchrone : scan_raw_stream ne fait qu'une liste de fichiers (path, name, size, mtime)
sans ffprobe ni API, utilisable sans internet. build_inventory_from_raw_stream enrichit
cette liste avec ffprobe ensuite (sans re-scanner le dossier).
"""
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

VIDEO_EXT = {".mkv", ".mp4", ".avi", ".mov", ".wmv", ".m4v", ".webm", ".mpg", ".mpeg"}

# Patterns pour détecter les épisodes de séries
SERIES_PATTERNS = [
    # S01E01, S1E1, s01e01
    re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})"),
    # 1x01, 01x01
    re.compile(r"(\d{1,2})x(\d{2,3})"),
    # 1-05, 01-05 (season-episode with dash, at start of filename)
    re.compile(r"^(\d{1,2})-(\d{2,3})(?:\s|$)"),
    # Season 1 Episode 1, Saison 1 Episode 1 (with dots, spaces, or underscores)
    re.compile(r"[Ss](?:eason|aison)[.\s_]*(\d{1,2})[.\s_]*[Ee](?:pisode)?[.\s_]*(\d{1,3})", re.IGNORECASE),
    # E01, Episode 01 (sans saison explicite)
    re.compile(r"[^a-zA-Z][Ee](?:pisode)?[.\s_]*(\d{1,3})(?:[^0-9]|$)"),
]

# Mots-clés pour documentaires (dans le chemin ou nom de fichier)
# Note: compared against lowercase path with dots replaced by spaces
DOCUMENTARY_KEYWORDS = [
    "documentary", "documentaire", "docuseries", "docu-series", "docu series",
    "national geographic", "natgeo", "bbc earth", "discovery channel",
    "history channel", "arte documentaire", "planet earth", "nature documentary",
    "blue planet", "cosmos", "wildlife", "our planet",
]


def _clean_title(name: str) -> str:
    """Nettoie un nom de fichier/dossier pour obtenir un titre propre."""
    # Remove common release tags
    cleaned = re.sub(
        r"[\s._-]*(dvdrip|bdrip|bluray|webrip|web-dl|hdtv|hdcam|hdrip|"
        r"1080p|720p|2160p|4k|uhd|"
        r"x264|x265|h264|h265|hevc|avc|xvid|divx|"
        r"aac|ac3|dts|mp3|flac|"
        r"proper|repack|internal|limited|unrated|extended|"
        r"sample|proof|"
        r"-[a-z0-9]+$)[\s._-]*",
        "", name, flags=re.IGNORECASE
    )
    # Remove season/episode patterns
    cleaned = re.sub(r"[\s._-]*s\d+[\s._-]*$", "", cleaned, flags=re.IGNORECASE)
    # Clean separators
    cleaned = cleaned.replace(".", " ").replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _get_series_folder_name(path: Path) -> str | None:
    """
    Cherche le nom de la série dans les dossiers parents.
    Remonte jusqu'à trouver un dossier qui ressemble à un nom de série propre.
    Ex: /Volumes/series/Beverly Hills, 90210/S01/episode.mkv -> "Beverly Hills, 90210"
    """
    # Check up to 3 parent levels
    for parent in list(path.parents)[:3]:
        folder_name = parent.name
        if not folder_name or folder_name in (".", ""):
            continue

        # Skip folders that look like season folders
        if re.match(r"^(season|saison|s)\s*\d+$", folder_name, re.IGNORECASE):
            continue

        # Skip folders with release tags (e.g., "Show.S01.DVDRip.XviD-GROUP")
        if re.search(r"(dvdrip|webrip|bluray|hdtv|xvid|x264|x265|-[a-z0-9]{2,8})$", folder_name, re.IGNORECASE):
            continue

        # This looks like a clean series folder name
        cleaned = _clean_title(folder_name)
        if cleaned and len(cleaned) > 2:
            return cleaned

    return None


def detect_media_type(path: Path) -> dict[str, Any]:
    """
    Détecte le type de média: movie, series, ou documentary.
    Retourne aussi season/episode si c'est une série.
    Utilise le nom du dossier parent si plus propre que le nom de fichier.
    """
    full_path = str(path).lower().replace(".", " ").replace("_", " ")
    filename = path.name

    result = {
        "media_type": "movie",
        "season": None,
        "episode": None,
        "series_name": None,
    }

    # Check for documentary keywords first
    for keyword in DOCUMENTARY_KEYWORDS:
        if keyword in full_path:
            result["media_type"] = "documentary"
            return result

    # Check for series patterns
    for i, pattern in enumerate(SERIES_PATTERNS):
        match = pattern.search(filename)
        if match:
            result["media_type"] = "series"
            groups = match.groups()

            if i == 4:  # Episode only pattern (no season) - index 4 after adding 1-05 pattern
                result["season"] = 1
                result["episode"] = int(groups[0])
            else:
                result["season"] = int(groups[0])
                result["episode"] = int(groups[1])

            # Try to get series name from folder first (usually cleaner)
            folder_name = _get_series_folder_name(path)

            # Extract series name from filename
            series_name_from_file = filename[:match.start()].strip()
            series_name_from_file = re.sub(r"[._-]+$", "", series_name_from_file)
            series_name_from_file = series_name_from_file.replace(".", " ").replace("_", " ").strip()
            series_name_from_file = _clean_title(series_name_from_file)

            # Choose the best series name:
            # 1. Folder name if it's clean and meaningful
            # 2. Filename-based name if folder is not useful
            if folder_name and len(folder_name) > 2:
                result["series_name"] = folder_name
            elif series_name_from_file and len(series_name_from_file) > 2:
                result["series_name"] = series_name_from_file
            else:
                # Last resort: try text after the episode number or parent folder
                after_pattern = filename[match.end():].strip()
                after_pattern = re.sub(r"^[\s._-]+", "", after_pattern)
                after_pattern = re.sub(r"\.[^.]+$", "", after_pattern)
                after_pattern = _clean_title(after_pattern)
                if after_pattern and len(after_pattern) > 2:
                    result["series_name"] = after_pattern
                else:
                    parent_name = path.parent.name
                    if parent_name and parent_name not in (".", ""):
                        result["series_name"] = _clean_title(parent_name)

            return result

    return result


def get_video_files(root: Path) -> list[Path]:
    """Liste tous les fichiers vidéo dans root et dans tous ses sous-dossiers (rglob = récursion complète)."""
    root = Path(str(root).strip()).expanduser().resolve()
    if not root.is_dir():
        return []
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def get_video_files_count(root: str) -> int:
    """Compte le nombre de fichiers vidéo (listing rapide, sans ffprobe). Pour vérifier que le scan est complet."""
    root_path = Path(root.strip()).expanduser().resolve()
    if not root_path.is_dir():
        return 0
    return sum(1 for p in root_path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXT)


def get_folder_diagnostic(root: str) -> dict[str, Any]:
    """Quand 0 fichier vidéo : renvoie infos pour déboguer (chemin résolu, nb fichiers total, extensions)."""
    root_path = Path(root.strip()).expanduser().resolve()
    out: dict[str, Any] = {
        "path_resolved": str(root_path),
        "path_exists": root_path.exists(),
        "is_dir": root_path.is_dir() if root_path.exists() else False,
        "total_files": 0,
        "by_extension": {},
    }
    if not root_path.exists() or not root_path.is_dir():
        return out
    try:
        all_files = list(root_path.rglob("*"))
        files_only = [p for p in all_files if p.is_file()]
        out["total_files"] = len(files_only)
        ext_count: dict[str, int] = {}
        for p in files_only:
            ext = (p.suffix or "").lower() or "(sans extension)"
            ext_count[ext] = ext_count.get(ext, 0) + 1
        out["by_extension"] = dict(sorted(ext_count.items(), key=lambda x: -x[1]))
    except Exception as e:
        out["error"] = str(e)
    return out


def run_ffprobe(path: Path) -> dict[str, Any] | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(path)],
            capture_output=True,
            text=True,
            timeout=25,
        )
        if out.returncode != 0:
            return None
        return json.loads(out.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError):
        return None


def _tag(probe: dict[str, Any], stream_or_format: dict, *keys: str) -> str:
    tags = stream_or_format.get("tags") or {}
    for k in keys:
        if k in tags and tags[k]:
            return (str(tags[k]) or "").strip()
    return ""


def _is_hdr(stream: dict) -> bool:
    """Détecte HDR via color_transfer, color_primaries ou pixel format (10-bit)."""
    ct = (stream.get("color_transfer") or "").lower()
    cp = (stream.get("color_primaries") or "").lower()
    pf = (stream.get("pix_fmt") or "").lower()
    if "smpte2084" in ct or "arib-std-b67" in ct or "bt2020" in ct:
        return True
    if "bt2020" in cp or "smpte2084" in cp:
        return True
    if "10le" in pf or "10be" in pf or "yuv420p10" in pf or "p010" in pf:
        return True
    return False


def parse_full_metadata(path: Path, probe: dict[str, Any] | None) -> dict[str, Any]:
    """
    Extrait tout ce que ffprobe donne: nom, film attribuable, dimensions, poids,
    codec, HDR, bitrates, langues, etc.
    Détecte aussi le type de média (movie, series, documentary).
    """
    path = Path(path)

    # Detect media type
    media_info = detect_media_type(path)

    row: dict[str, Any] = {
        "path": str(path),
        "name": path.name,
        "size_bytes": 0,
        "duration_sec": 0.0,
        "format_name": "",
        "format_long_name": "",
        "bit_rate": 0,
        "metadata_title": "",
        "film_attribuable": "",
        "width": 0,
        "height": 0,
        "video_codec": "",
        "video_profile": "",
        "video_bit_rate": 0,
        "pix_fmt": "",
        "color_space": "",
        "color_transfer": "",
        "color_primaries": "",
        "hdr": False,
        "video_nb_frames": 0,
        "audio_tracks": [],
        "audio_languages": "",
        "tags": {},
        # Media type fields
        "media_type": media_info["media_type"],
        "season": media_info["season"],
        "episode": media_info["episode"],
        "series_name": media_info["series_name"],
    }
    if not probe:
        row["film_attribuable"] = path.stem
        return row

    fmt = probe.get("format") or {}
    row["size_bytes"] = int(fmt.get("size") or 0)
    row["duration_sec"] = float(fmt.get("duration") or 0)
    row["format_name"] = (fmt.get("format_name") or "").strip()
    row["format_long_name"] = (fmt.get("format_long_name") or "").strip()
    if fmt.get("bit_rate"):
        row["bit_rate"] = int(fmt["bit_rate"])
    row["metadata_title"] = _tag(probe, fmt, "title", "TITLE", "Title") or ""
    row["film_attribuable"] = row["metadata_title"] or path.stem
    row["tags"] = dict(fmt.get("tags") or {})

    for s in probe.get("streams") or []:
        if s.get("codec_type") == "video":
            row["width"] = int(s.get("width") or 0)
            row["height"] = int(s.get("height") or 0)
            row["video_codec"] = (s.get("codec_name") or "").upper()
            row["video_profile"] = (s.get("profile") or "").strip()
            if s.get("bit_rate"):
                row["video_bit_rate"] = int(s["bit_rate"])
            row["pix_fmt"] = (s.get("pix_fmt") or "").strip()
            row["color_space"] = (s.get("color_space") or "").strip()
            row["color_transfer"] = (s.get("color_transfer") or "").strip()
            row["color_primaries"] = (s.get("color_primaries") or "").strip()
            row["hdr"] = _is_hdr(s)
            row["video_nb_frames"] = int(s.get("nb_frames") or 0)
        elif s.get("codec_type") == "audio":
            lang = _tag(probe, s, "language", "LANGUAGE", "lang")
            row["audio_tracks"].append({
                "codec": (s.get("codec_name") or "").upper(),
                "channels": int(s.get("channels") or 0),
                "channel_layout": (s.get("channel_layout") or "").strip(),
                "sample_rate": (s.get("sample_rate") or "").strip(),
                "bit_rate": int(s.get("bit_rate") or 0),
                "language": lang,
            })
    if row["audio_tracks"]:
        row["audio_languages"] = ", ".join(t["language"] or "?" for t in row["audio_tracks"])

    return row


def scan_and_build_inventory(root: str) -> list[dict[str, Any]]:
    """Scan complet du volume et retourne la liste de tous les fichiers avec métadonnées complètes."""
    return list(scan_and_build_inventory_stream(root))


def scan_and_build_inventory_stream(root: str):
    """
    Parcourt récursivement l'arborescence (tous les sous-dossiers du chemin d'origine)
    et yield une ligne dès qu'un fichier vidéo est analysé. rglob("*") garantit que
    tous les niveaux de sous-dossiers sont parcourus — aucun sous-dossier n'est ignoré.
    """
    root_path = Path(root.strip()).expanduser().resolve()
    if not root_path.is_dir():
        return
    for p in root_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            continue
        probe = run_ffprobe(p)
        row = parse_full_metadata(p, probe)
        yield row


def scan_raw_stream(root: str):
    """
    Liste uniquement les fichiers vidéo (path, name, size_bytes, mtime).
    Aucun ffprobe, aucune API — utilisable sans internet. Pour mode asynchrone.
    """
    root_path = Path(root.strip()).expanduser().resolve()
    if not root_path.is_dir():
        return
    for p in root_path.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in VIDEO_EXT:
            continue
        try:
            st = os.stat(p)
            yield {
                "path": str(p),
                "name": p.name,
                "size_bytes": getattr(st, "st_size", 0) or 0,
                "mtime": st.st_mtime,
            }
        except OSError:
            continue


def build_inventory_from_raw_stream(raw_files: list[dict[str, Any]]):
    """
    À partir d'une liste raw (path, name, size_bytes, mtime), exécute ffprobe
    sur chaque fichier et yield une ligne inventaire complète. Ne re-scanne pas
    le dossier — utilise la base stockée. Fichiers manquants → ligne minimale.
    """
    for raw in raw_files:
        path_str = raw.get("path") or ""
        if not path_str:
            continue
        p = Path(path_str)
        if not p.is_file():
            row = parse_full_metadata(p, None)
            row["size_bytes"] = raw.get("size_bytes") or 0
            yield row
            continue
        probe = run_ffprobe(p)
        row = parse_full_metadata(p, probe)
        yield row
