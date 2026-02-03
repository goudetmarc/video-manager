"""
File Normalizer: generates standardized filenames for media files.

This module provides:
1. Technical tag extraction from ffprobe metadata
2. Standardized filename generation
3. Preview/validation workflow with lock/unlock support
4. Safe file renaming with validation
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TechnicalTags:
    """Technical metadata extracted from video file."""
    resolution: str = ""           # e.g., "1080p", "2160p", "720p"
    video_codec: str = ""          # e.g., "H264", "HEVC", "AV1"
    audio_codec: str = ""          # e.g., "AAC", "DTS", "AC3"
    audio_channels: str = ""       # e.g., "5.1", "7.1", "2.0"
    hdr: str = ""                  # e.g., "HDR", "HDR10", "DV" (Dolby Vision)
    source: str = ""               # e.g., "BluRay", "WEB-DL", "HDTV"


@dataclass
class NormalizedFilename:
    """Components of a normalized filename."""
    title: str = ""
    year: str = ""
    season: str = ""               # For series: "S01"
    episode: str = ""              # For series: "E05"
    episode_title: str = ""        # Optional episode title
    resolution: str = ""
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: str = ""
    hdr: str = ""
    source: str = ""
    extension: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "title": self.title,
            "year": self.year,
            "season": self.season,
            "episode": self.episode,
            "episode_title": self.episode_title,
            "resolution": self.resolution,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
            "audio_channels": self.audio_channels,
            "hdr": self.hdr,
            "source": self.source,
            "extension": self.extension,
        }

    def generate_filename(self, locks: dict[str, bool] | None = None) -> str:
        """
        Generate the final filename string.

        Args:
            locks: Dictionary of component_name -> is_locked. Locked components
                   are preserved as-is in the filename.
        """
        locks = locks or {}
        parts = []

        # Title is always first
        if self.title:
            parts.append(self.title)

        # Year (for movies)
        if self.year and not self.season:
            parts.append(f"({self.year})")

        # Season/Episode (for series)
        if self.season:
            se_part = self.season
            if self.episode:
                se_part += self.episode
            parts.append(se_part)

        # Episode title (optional, for series)
        if self.episode_title:
            parts.append(self.episode_title)

        # Technical tags
        tech_parts = []
        if self.resolution:
            tech_parts.append(self.resolution)
        if self.hdr:
            tech_parts.append(self.hdr)
        if self.video_codec:
            tech_parts.append(self.video_codec)
        if self.audio_codec:
            audio = self.audio_codec
            if self.audio_channels:
                audio += f" {self.audio_channels}"
            tech_parts.append(audio)
        if self.source:
            tech_parts.append(self.source)

        if tech_parts:
            parts.append(f"[{' '.join(tech_parts)}]")

        filename = " ".join(parts)
        if self.extension:
            filename += self.extension

        return filename


class MediaAnalyzer:
    """Extracts technical information from video metadata."""

    # Resolution mappings
    RESOLUTION_MAP = {
        (3840, 2160): "2160p",
        (2560, 1440): "1440p",
        (1920, 1080): "1080p",
        (1280, 720): "720p",
        (854, 480): "480p",
        (640, 360): "360p",
    }

    # Codec display names
    VIDEO_CODEC_MAP = {
        "hevc": "HEVC",
        "h265": "HEVC",
        "h.265": "HEVC",
        "h264": "H264",
        "h.264": "H264",
        "avc": "H264",
        "av1": "AV1",
        "vp9": "VP9",
        "mpeg4": "MPEG4",
        "xvid": "XviD",
        "divx": "DivX",
    }

    AUDIO_CODEC_MAP = {
        "aac": "AAC",
        "ac3": "AC3",
        "eac3": "EAC3",
        "dts": "DTS",
        "dtshd": "DTS-HD",
        "dts-hd": "DTS-HD",
        "truehd": "TrueHD",
        "flac": "FLAC",
        "mp3": "MP3",
        "opus": "Opus",
        "vorbis": "Vorbis",
        "pcm": "PCM",
    }

    # Source detection patterns (from filename)
    SOURCE_PATTERNS = [
        (r"blu-?ray|bdrip|bd-?rip", "BluRay"),
        (r"web-?dl", "WEB-DL"),
        (r"webrip|web-?rip", "WEBRip"),
        (r"hdtv", "HDTV"),
        (r"dvdrip|dvd-?rip", "DVDRip"),
        (r"hdcam|hd-?cam", "HDCAM"),
        (r"telesync|ts(?![a-z])", "TS"),
        (r"telecine|tc(?![a-z])", "TC"),
    ]

    @classmethod
    def extract_tags(cls, file_data: dict[str, Any], filename: str) -> TechnicalTags:
        """
        Extract technical tags from file metadata and filename.

        Args:
            file_data: Dictionary with video metadata (from ffprobe/inventory)
            filename: Original filename for source detection

        Returns:
            TechnicalTags with extracted information
        """
        tags = TechnicalTags()

        # Resolution
        width = file_data.get("width", 0) or 0
        height = file_data.get("height", 0) or 0
        tags.resolution = cls._get_resolution(width, height)

        # Video codec
        video_codec = (file_data.get("video_codec") or file_data.get("codec") or "").lower()
        tags.video_codec = cls.VIDEO_CODEC_MAP.get(video_codec, video_codec.upper() if video_codec else "")

        # Audio codec and channels
        audio_codec = (file_data.get("audio_codec") or "").lower()
        tags.audio_codec = cls.AUDIO_CODEC_MAP.get(audio_codec, audio_codec.upper() if audio_codec else "")

        audio_channels = file_data.get("audio_channels") or 0
        tags.audio_channels = cls._get_channel_layout(audio_channels)

        # HDR
        if file_data.get("hdr"):
            color_transfer = (file_data.get("color_transfer") or "").lower()
            if "arib-std-b67" in color_transfer:
                tags.hdr = "HLG"
            elif "smpte2084" in color_transfer:
                # Could be HDR10 or Dolby Vision
                tags.hdr = "HDR10"
            else:
                tags.hdr = "HDR"

        # Source (detect from filename)
        tags.source = cls._detect_source(filename)

        return tags

    @classmethod
    def _get_resolution(cls, width: int, height: int) -> str:
        """Convert width/height to standard resolution string."""
        if height == 0:
            return ""

        # Check exact matches first
        for (w, h), res in cls.RESOLUTION_MAP.items():
            if height == h or (width == w and height == h):
                return res

        # Fallback: use height
        if height >= 2000:
            return "2160p"
        elif height >= 1000:
            return "1080p"
        elif height >= 700:
            return "720p"
        elif height >= 400:
            return "480p"
        else:
            return f"{height}p"

    @classmethod
    def _get_channel_layout(cls, channels: int) -> str:
        """Convert channel count to layout string."""
        channel_map = {
            1: "1.0",
            2: "2.0",
            3: "2.1",
            6: "5.1",
            7: "6.1",
            8: "7.1",
        }
        return channel_map.get(channels, f"{channels}ch" if channels else "")

    @classmethod
    def _detect_source(cls, filename: str) -> str:
        """Detect video source from filename patterns."""
        filename_lower = filename.lower()
        for pattern, source in cls.SOURCE_PATTERNS:
            if re.search(pattern, filename_lower):
                return source
        return ""


class FileNormalizer:
    """
    Main service for generating normalized filenames.

    Workflow:
    1. generate_preview(file_path) → Returns NormalizedFilename components
    2. User reviews and optionally locks components
    3. execute_rename(file_path, components, locks) → Performs actual rename
    """

    # Patterns to clean from titles
    CLEANUP_PATTERNS = [
        r"[\s._-]*(dvdrip|bdrip|bluray|blu-ray|webrip|web-dl|web-rip|hdtv|hdcam|hdrip)",
        r"[\s._-]*(1080p|720p|2160p|4k|uhd)",
        r"[\s._-]*(x264|x265|h264|h265|hevc|avc|xvid|divx|av1)",
        r"[\s._-]*(aac|ac3|dts|mp3|flac|truehd|eac3|atmos)",
        r"[\s._-]*(proper|repack|internal|limited|unrated|extended)",
        r"[\s._-]*(hdr|hdr10|dv|dolby[\s._-]*vision|hlg)",
        r"[\s._-]*(10bit|8bit|10-bit|8-bit)",
        r"[\s._-]*(multi|vff|vfq|french|english|vostfr|subbed)",
        r"[\s._-]*-[a-z0-9]{2,12}$",  # Release group
        r"\[.*?\]",  # Bracketed content
        r"\(.*?\)",  # Parenthesized content (except year)
    ]

    # Series patterns
    SERIES_PATTERNS = [
        re.compile(r"[Ss](\d{1,2})[Ee](\d{1,3})"),
        re.compile(r"(\d{1,2})x(\d{2,3})"),
        re.compile(r"[Ss](?:eason|aison)[.\s_]*(\d{1,2})[.\s_]*[Ee](?:pisode)?[.\s_]*(\d{1,3})", re.IGNORECASE),
    ]

    # Year pattern
    YEAR_PATTERN = re.compile(r"[\(\[]?((?:19|20)\d{2})[\)\]]?")

    def __init__(self):
        self.analyzer = MediaAnalyzer()

    def generate_preview(
        self,
        file_path: str,
        file_data: dict[str, Any],
        tmdb_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Generate a preview of the normalized filename.

        Args:
            file_path: Full path to the video file
            file_data: Dictionary with video metadata (from inventory)
            tmdb_data: Optional TMDB metadata for title/year

        Returns:
            Dictionary with:
            - original_filename: The current filename
            - proposed_filename: The suggested new filename
            - components: NormalizedFilename components as dict
            - can_rename: Whether the file can be renamed
        """
        path = Path(file_path)
        original_filename = path.name
        extension = path.suffix.lower()

        # Extract technical tags
        tags = self.analyzer.extract_tags(file_data, original_filename)

        # Build normalized filename components
        components = NormalizedFilename(
            extension=extension,
            resolution=tags.resolution,
            video_codec=tags.video_codec,
            audio_codec=tags.audio_codec,
            audio_channels=tags.audio_channels,
            hdr=tags.hdr,
            source=tags.source,
        )

        # Get title and year
        media_type = file_data.get("media_type", "movie")

        if tmdb_data:
            # Use TMDB data
            components.title = tmdb_data.get("original_title") or tmdb_data.get("title") or ""
            components.year = tmdb_data.get("year") or ""
        else:
            # Extract from filename
            components.title = self._extract_title(original_filename, extension)
            components.year = self._extract_year(original_filename)

        # Handle series
        if media_type == "series":
            season = file_data.get("season")
            episode = file_data.get("episode")
            if season is not None:
                components.season = f"S{season:02d}"
            if episode is not None:
                components.episode = f"E{episode:02d}"
            # Clear year for series (use season/episode instead)
            components.year = ""

        # Check if file can be renamed
        can_rename = path.exists() and os.access(path, os.W_OK)

        proposed = components.generate_filename()

        return {
            "original_path": file_path,
            "original_filename": original_filename,
            "proposed_filename": proposed,
            "components": components.to_dict(),
            "can_rename": can_rename,
            "media_type": media_type,
        }

    def execute_rename(
        self,
        file_path: str,
        components: dict[str, Any],
        dry_run: bool = False,
    ) -> dict[str, Any]:
        """
        Execute the file rename operation.

        Args:
            file_path: Full path to the video file
            components: NormalizedFilename components as dict
            dry_run: If True, only simulate the rename

        Returns:
            Dictionary with:
            - success: Whether the operation succeeded
            - old_path: Original file path
            - new_path: New file path (or proposed if dry_run)
            - error: Error message if failed
        """
        path = Path(file_path)

        if not path.exists():
            return {
                "success": False,
                "old_path": file_path,
                "new_path": None,
                "error": "Le fichier n'existe pas",
            }

        if not os.access(path, os.W_OK):
            return {
                "success": False,
                "old_path": file_path,
                "new_path": None,
                "error": "Pas de permission d'écriture sur le fichier",
            }

        # Reconstruct NormalizedFilename from components
        norm = NormalizedFilename(
            title=components.get("title", ""),
            year=components.get("year", ""),
            season=components.get("season", ""),
            episode=components.get("episode", ""),
            episode_title=components.get("episode_title", ""),
            resolution=components.get("resolution", ""),
            video_codec=components.get("video_codec", ""),
            audio_codec=components.get("audio_codec", ""),
            audio_channels=components.get("audio_channels", ""),
            hdr=components.get("hdr", ""),
            source=components.get("source", ""),
            extension=components.get("extension", path.suffix),
        )

        new_filename = norm.generate_filename()
        new_path = path.parent / new_filename

        # Safety checks
        if new_path.exists() and new_path != path:
            return {
                "success": False,
                "old_path": file_path,
                "new_path": str(new_path),
                "error": "Un fichier avec ce nom existe déjà",
            }

        if not new_filename or new_filename == path.suffix:
            return {
                "success": False,
                "old_path": file_path,
                "new_path": None,
                "error": "Le nouveau nom de fichier est invalide",
            }

        # Check for invalid characters
        invalid_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in invalid_chars:
            if char in new_filename:
                return {
                    "success": False,
                    "old_path": file_path,
                    "new_path": None,
                    "error": f"Caractère invalide dans le nom: {char}",
                }

        if dry_run:
            return {
                "success": True,
                "old_path": file_path,
                "new_path": str(new_path),
                "dry_run": True,
            }

        try:
            path.rename(new_path)
            return {
                "success": True,
                "old_path": file_path,
                "new_path": str(new_path),
            }
        except OSError as e:
            return {
                "success": False,
                "old_path": file_path,
                "new_path": str(new_path),
                "error": f"Erreur lors du renommage: {e}",
            }

    def _extract_title(self, filename: str, extension: str) -> str:
        """Extract clean title from filename."""
        # Remove extension
        name = filename
        if name.lower().endswith(extension.lower()):
            name = name[: -len(extension)]

        # Remove series patterns first
        for pattern in self.SERIES_PATTERNS:
            match = pattern.search(name)
            if match:
                name = name[: match.start()]
                break

        # Remove year
        year_match = self.YEAR_PATTERN.search(name)
        if year_match:
            name = name[: year_match.start()]

        # Apply cleanup patterns
        for pattern in self.CLEANUP_PATTERNS:
            name = re.sub(pattern, "", name, flags=re.IGNORECASE)

        # Clean separators
        name = name.replace(".", " ").replace("_", " ").replace("-", " ")
        name = re.sub(r"\s+", " ", name).strip()

        # Capitalize properly
        name = self._title_case(name)

        return name

    def _extract_year(self, filename: str) -> str:
        """Extract year from filename."""
        match = self.YEAR_PATTERN.search(filename)
        if match:
            return match.group(1)
        return ""

    def _title_case(self, text: str) -> str:
        """Convert text to title case, preserving acronyms."""
        words = text.split()
        result = []
        for word in words:
            # Keep all-caps words as-is (acronyms)
            if word.isupper() and len(word) > 1:
                result.append(word)
            else:
                result.append(word.capitalize())
        return " ".join(result)


# Singleton instance
_normalizer = None


def get_normalizer() -> FileNormalizer:
    """Get the singleton FileNormalizer instance."""
    global _normalizer
    if _normalizer is None:
        _normalizer = FileNormalizer()
    return _normalizer
