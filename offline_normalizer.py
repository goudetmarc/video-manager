"""
Offline Normalizer: Normalize media files WITHOUT any API dependency.
Based purely on filename/folder analysis.

This module provides:
1. Directory analysis to detect movies and series
2. Normalization plan generation
3. Safe execution with validation
"""

import os
import re
import shutil
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)

# Video extensions
VIDEO_EXTENSIONS = {'.mkv', '.mp4', '.avi', '.mov', '.m4v', '.wmv', '.webm', '.mpg', '.mpeg'}

# Quality tags to detect
QUALITY_TAGS = [
    '2160p', '4k', 'uhd',
    '1080p', '1080i', 'fullhd', 'fhd',
    '720p', 'hd',
    '480p', 'sd',
    '360p',
]

# Source tags
SOURCE_TAGS = [
    'bluray', 'blu-ray', 'bdrip', 'brrip',
    'web-dl', 'webdl', 'webrip', 'web',
    'hdtv', 'hdtvrip',
    'dvdrip', 'dvd',
    'hdcam', 'cam', 'ts', 'telesync',
]

# Codec tags
CODEC_TAGS = [
    'hevc', 'x265', 'h265', 'h.265',
    'avc', 'x264', 'h264', 'h.264',
    'xvid', 'divx',
    'av1',
]

# Audio tags
AUDIO_TAGS = [
    'dts', 'dts-hd', 'dtshd', 'truehd', 'atmos',
    'ac3', 'eac3', 'dd5.1', 'dd7.1',
    'aac', 'flac', 'mp3',
]

# HDR tags
HDR_TAGS = [
    'hdr', 'hdr10', 'hdr10+', 'dolby vision', 'dv', 'hlg',
]


class OfflineNormalizer:
    """
    Normalisation de fichiers SANS API externe.
    Analyse uniquement les noms de fichiers et dossiers.
    """

    def __init__(self):
        self.video_extensions = VIDEO_EXTENSIONS

    def analyze_directory(self, root_path: str) -> Dict[str, Any]:
        """
        Analyser dossier et detecter oeuvres.

        Args:
            root_path: Path to analyze

        Returns:
            {
                "root_path": str,
                "movies": [...],
                "series": [...],
                "unidentified": [...],
                "stats": {...}
            }
        """
        root = Path(root_path)
        if not root.exists():
            raise ValueError(f"Path does not exist: {root_path}")
        if not root.is_dir():
            raise ValueError(f"Path is not a directory: {root_path}")

        logger.info(f"Analyzing directory: {root_path}")

        movies_dict: Dict[tuple, List[Dict]] = defaultdict(list)
        series_dict: Dict[tuple, List[Dict]] = defaultdict(list)
        unidentified: List[Dict] = []

        total_files = 0
        total_size = 0

        for filepath in root.rglob("*"):
            if not filepath.is_file():
                continue
            if filepath.suffix.lower() not in self.video_extensions:
                continue

            total_files += 1
            file_size = filepath.stat().st_size
            total_size += file_size

            file_info = self._analyze_file(filepath)

            if file_info is None:
                unidentified.append({
                    "filepath": str(filepath),
                    "filename": filepath.name,
                    "size": file_size,
                })
                continue

            if file_info["type"] == "series":
                key = self._normalize_title(file_info["title"])
                series_dict[key].append(file_info)
            else:
                key = (self._normalize_title(file_info["title"]), file_info.get("year"))
                movies_dict[key].append(file_info)

        # Format results
        movies = self._format_movies(movies_dict, root_path)
        series = self._format_series(series_dict, root_path)

        logger.info(f"Found {len(movies)} movies, {len(series)} series, {len(unidentified)} unidentified")

        return {
            "root_path": root_path,
            "total_files": total_files,  # Also at root for convenience
            "movies": movies,
            "series": series,
            "unidentified": unidentified,
            "stats": {
                "total_files": total_files,
                "total_size": total_size,
                "movies_count": len(movies),
                "series_count": len(series),
                "unidentified_count": len(unidentified),
            }
        }

    def _analyze_file(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """
        Analyze a single file to extract metadata.

        Returns file info dict or None if unidentifiable.
        """
        filename = filepath.name
        parent_folder = filepath.parent.name

        # Try to detect series first (more specific patterns)
        series_info = self._detect_series(filename, filepath)
        if series_info:
            return series_info

        # Try to detect movie
        movie_info = self._detect_movie(filename, filepath)
        if movie_info:
            return movie_info

        # Use parent folder name as fallback
        folder_info = self._detect_from_folder(parent_folder, filepath)
        if folder_info:
            return folder_info

        return None

    def _detect_series(self, filename: str, filepath: Path) -> Optional[Dict[str, Any]]:
        """Detect series episode from filename."""
        # Pattern: Show.Name.S01E01 or Show Name S01E01
        pattern1 = r'^(.+?)[\s._-]+[Ss](\d{1,2})[Ee](\d{1,3})(?:[\s._-]+(.+?))?(?:\.\w+)?$'
        match = re.match(pattern1, filename)
        if match:
            title = self._clean_title(match.group(1))
            season = int(match.group(2))
            episode = int(match.group(3))
            rest = match.group(4) or ""

            # Try to extract episode title (before quality tags)
            ep_title = self._extract_episode_title(rest)

            # Extract year from title if present
            year = self._extract_year(title)
            if year:
                title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?\s*', ' ', title).strip()

            return {
                "type": "series",
                "title": title,
                "year": year,
                "season": season,
                "episode": episode,
                "episode_title": ep_title,
                "filepath": str(filepath),
                "filename": filename,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filename),
            }

        # Pattern: Show.Name.1x01 or Show Name 1x01
        pattern2 = r'^(.+?)[\s._-]+(\d{1,2})x(\d{1,3})(?:[\s._-]+(.+?))?(?:\.\w+)?$'
        match = re.match(pattern2, filename)
        if match:
            title = self._clean_title(match.group(1))
            season = int(match.group(2))
            episode = int(match.group(3))
            rest = match.group(4) or ""
            ep_title = self._extract_episode_title(rest)

            year = self._extract_year(title)
            if year:
                title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?\s*', ' ', title).strip()

            return {
                "type": "series",
                "title": title,
                "year": year,
                "season": season,
                "episode": episode,
                "episode_title": ep_title,
                "filepath": str(filepath),
                "filename": filename,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filename),
            }

        # Pattern: Show Name - S01E01 - Episode Title
        pattern3 = r'^(.+?)\s*-\s*[Ss](\d{1,2})[Ee](\d{1,3})\s*-?\s*(.*)$'
        match = re.match(pattern3, filename.rsplit('.', 1)[0])
        if match:
            title = self._clean_title(match.group(1))
            season = int(match.group(2))
            episode = int(match.group(3))
            ep_title = self._extract_episode_title(match.group(4))

            year = self._extract_year(title)
            if year:
                title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?\s*', ' ', title).strip()

            return {
                "type": "series",
                "title": title,
                "year": year,
                "season": season,
                "episode": episode,
                "episode_title": ep_title,
                "filepath": str(filepath),
                "filename": filename,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filename),
            }

        return None

    def _detect_movie(self, filename: str, filepath: Path) -> Optional[Dict[str, Any]]:
        """Detect movie from filename."""
        name = filename.rsplit('.', 1)[0]  # Remove extension

        # Pattern: Movie Name (2020) or Movie.Name.2020
        # Look for year in parentheses first
        pattern1 = r'^(.+?)[\s._-]*\((\d{4})\)'
        match = re.search(pattern1, name)
        if match:
            title = self._clean_title(match.group(1))
            year = int(match.group(2))

            return {
                "type": "movie",
                "title": title,
                "year": year,
                "filepath": str(filepath),
                "filename": filename,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filename),
            }

        # Pattern: Movie.Name.2020.1080p
        # Find year followed by quality tag
        pattern2 = r'^(.+?)[\s._-]+(\d{4})[\s._-]+(?:' + '|'.join(QUALITY_TAGS) + ')'
        match = re.search(pattern2, name, re.IGNORECASE)
        if match:
            title = self._clean_title(match.group(1))
            year = int(match.group(2))

            return {
                "type": "movie",
                "title": title,
                "year": year,
                "filepath": str(filepath),
                "filename": filename,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filename),
            }

        # Look for any quality/source tag and extract title before it
        all_tags = QUALITY_TAGS + SOURCE_TAGS + CODEC_TAGS
        for tag in all_tags:
            pattern = r'^(.+?)[\s._-]+' + re.escape(tag)
            match = re.search(pattern, name, re.IGNORECASE)
            if match:
                title = self._clean_title(match.group(1))
                year = self._extract_year(title)
                if year:
                    title = re.sub(r'\s*[\(\[]?\d{4}[\)\]]?\s*', ' ', title).strip()

                return {
                    "type": "movie",
                    "title": title,
                    "year": year,
                    "filepath": str(filepath),
                    "filename": filename,
                    "size": filepath.stat().st_size,
                    "technical": self._extract_technical_tags(filename),
                }

        return None

    def _detect_from_folder(self, folder_name: str, filepath: Path) -> Optional[Dict[str, Any]]:
        """Try to detect media info from parent folder name."""
        # Check if folder looks like a movie folder
        pattern = r'^(.+?)[\s._-]*[\(\[]?(\d{4})[\)\]]?'
        match = re.match(pattern, folder_name)
        if match:
            title = self._clean_title(match.group(1))
            year = int(match.group(2))

            return {
                "type": "movie",
                "title": title,
                "year": year,
                "filepath": str(filepath),
                "filename": filepath.name,
                "size": filepath.stat().st_size,
                "technical": self._extract_technical_tags(filepath.name),
                "from_folder": True,
            }

        return None

    def _clean_title(self, title: str) -> str:
        """Clean up title string."""
        # Replace dots and underscores with spaces
        title = title.replace('.', ' ').replace('_', ' ')
        # Remove multiple spaces
        title = re.sub(r'\s+', ' ', title)
        # Strip and title case
        title = title.strip()
        # Capitalize first letter of each word
        return title.title()

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from text."""
        # Look for 4-digit year between 1900 and 2099
        match = re.search(r'[\(\[]?(\d{4})[\)\]]?', text)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2099:
                return year
        return None

    def _extract_episode_title(self, text: str) -> str:
        """Extract episode title, removing quality tags."""
        if not text:
            return ""

        # Remove file extension if present
        text = re.sub(r'\.\w{2,4}$', '', text)

        # Remove quality/codec/source tags
        all_tags = QUALITY_TAGS + SOURCE_TAGS + CODEC_TAGS + AUDIO_TAGS + HDR_TAGS
        pattern = r'[\s._-]*(' + '|'.join(re.escape(t) for t in all_tags) + r')[\s._-]*'
        text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)

        # Remove release group (usually at end after dash)
        text = re.sub(r'\s*-\s*[A-Za-z0-9]+$', '', text)

        # Clean up
        text = self._clean_title(text)

        return text if text and len(text) > 1 else ""

    def _extract_technical_tags(self, filename: str) -> Dict[str, Optional[str]]:
        """Extract technical information from filename."""
        filename_lower = filename.lower()

        result = {
            "resolution": None,
            "source": None,
            "codec": None,
            "audio": None,
            "hdr": None,
        }

        # Resolution
        for tag in QUALITY_TAGS:
            if tag in filename_lower:
                result["resolution"] = tag.upper()
                break

        # Source
        for tag in SOURCE_TAGS:
            if tag in filename_lower:
                result["source"] = tag.upper().replace('-', '')
                break

        # Codec
        for tag in CODEC_TAGS:
            if tag in filename_lower:
                result["codec"] = tag.upper()
                break

        # Audio
        for tag in AUDIO_TAGS:
            if tag in filename_lower:
                result["audio"] = tag.upper()
                break

        # HDR
        for tag in HDR_TAGS:
            if tag.replace(' ', '') in filename_lower.replace(' ', ''):
                result["hdr"] = tag.upper()
                break

        return result

    def _normalize_title(self, title: str) -> str:
        """Normalize title for grouping."""
        # Lowercase, remove special chars
        normalized = title.lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        return normalized

    def _format_movies(self, movies_dict: Dict, root_path: str) -> List[Dict]:
        """Format movie data for frontend."""
        result = []

        for (title_key, year), files in movies_dict.items():
            if not files:
                continue

            # Get best title (from file with most info)
            best_file = max(files, key=lambda f: len(f.get("title", "")))
            title = best_file["title"]

            # Current folders
            current_folders = list(set([
                str(Path(f["filepath"]).parent)
                for f in files
            ]))

            # Suggested folder
            if year:
                folder_name = f"{title} ({year})"
            else:
                folder_name = title
            suggested_folder = os.path.join(root_path, folder_name)

            # Total size
            total_size = sum(f.get("size", 0) for f in files)

            result.append({
                "id": f"movie_{hash(title_key) % 100000}",
                "detected_title": title,
                "detected_year": year,
                "files": files,
                "file_count": len(files),
                "total_size": total_size,
                "current_folders": current_folders,
                "suggested_folder": suggested_folder,
            })

        return sorted(result, key=lambda x: x["detected_title"].lower())

    def _format_series(self, series_dict: Dict, root_path: str) -> List[Dict]:
        """Format series data for frontend."""
        result = []

        for title_key, episodes in series_dict.items():
            if not episodes:
                continue

            # Get best title
            best_ep = max(episodes, key=lambda e: len(e.get("title", "")))
            title = best_ep["title"]
            year = best_ep.get("year")

            # Current folders
            current_folders = list(set([
                str(Path(ep["filepath"]).parent)
                for ep in episodes
            ]))

            # Suggested folder
            if year:
                folder_name = f"{title} ({year})"
            else:
                folder_name = title
            suggested_folder = os.path.join(root_path, folder_name)

            # Sort episodes
            sorted_episodes = sorted(episodes, key=lambda e: (e.get("season", 0), e.get("episode", 0)))

            # Count seasons
            seasons = set(ep.get("season", 0) for ep in episodes)

            # Total size
            total_size = sum(ep.get("size", 0) for ep in episodes)

            result.append({
                "id": f"series_{hash(title_key) % 100000}",
                "detected_title": title,
                "detected_year": year,
                "episodes": sorted_episodes,
                "episode_count": len(episodes),
                "season_count": len(seasons),
                "total_size": total_size,
                "current_folders": current_folders,
                "suggested_folder": suggested_folder,
            })

        return sorted(result, key=lambda x: x["detected_title"].lower())

    def generate_plan(self, analysis: Dict, options: Optional[Dict] = None) -> List[Dict]:
        """
        Generate normalization plan from analysis.

        Args:
            analysis: Result from analyze_directory()
            options: Optional settings for normalization

        Returns:
            List of operations to perform
        """
        options = options or {}
        operations = []
        seen_targets: Set[str] = set()

        # Process movies
        for movie in analysis.get("movies", []):
            movie_ops = self._generate_movie_operations(movie, seen_targets, options)
            operations.extend(movie_ops)

        # Process series
        for series in analysis.get("series", []):
            series_ops = self._generate_series_operations(series, seen_targets, options)
            operations.extend(series_ops)

        return operations

    def _generate_movie_operations(self, movie: Dict, seen_targets: Set[str], options: Dict) -> List[Dict]:
        """Generate operations for a movie."""
        operations = []
        target_folder = movie["suggested_folder"]

        # Create folder operation
        operations.append({
            "type": "create_folder",
            "source": None,
            "target": target_folder,
            "preview": f"CREATE: {Path(target_folder).name}/",
            "editable": True,
            "item_id": movie["id"],
            "item_type": "movie",
        })

        # Move each file
        for file_info in movie["files"]:
            source = file_info["filepath"]
            ext = Path(source).suffix

            # Generate new filename
            title = movie["detected_title"]
            year = movie["detected_year"]

            if year:
                new_name = f"{title} ({year})"
            else:
                new_name = title

            # Add technical tags if option enabled
            if options.get("keep_technical_tags", True):
                tech = file_info.get("technical", {})
                tags = []
                if tech.get("resolution"):
                    tags.append(tech["resolution"])
                if tech.get("hdr"):
                    tags.append(tech["hdr"])
                if tech.get("codec"):
                    tags.append(tech["codec"])
                if tech.get("source"):
                    tags.append(tech["source"])
                if tags:
                    new_name += f" [{' '.join(tags)}]"

            new_name += ext
            target = os.path.join(target_folder, new_name)

            # Handle duplicates
            if target in seen_targets:
                counter = 2
                while target in seen_targets:
                    base = new_name.rsplit('.', 1)[0]
                    target = os.path.join(target_folder, f"{base} (v{counter}){ext}")
                    counter += 1

            seen_targets.add(target)

            operations.append({
                "type": "move_file",
                "source": source,
                "target": target,
                "preview": f"MOVE: {Path(source).name} -> {Path(target).name}",
                "editable": True,
                "item_id": movie["id"],
                "item_type": "movie",
            })

        # Cleanup old folders
        for folder in movie["current_folders"]:
            if folder != target_folder and folder != analysis.get("root_path"):
                operations.append({
                    "type": "delete_folder_if_empty",
                    "source": folder,
                    "target": None,
                    "preview": f"DELETE (if empty): {Path(folder).name}/",
                    "editable": False,
                    "item_id": movie["id"],
                    "item_type": "movie",
                })

        return operations

    def _generate_series_operations(self, series: Dict, seen_targets: Set[str], options: Dict) -> List[Dict]:
        """Generate operations for a series."""
        operations = []
        target_folder = series["suggested_folder"]

        # Create main folder
        operations.append({
            "type": "create_folder",
            "source": None,
            "target": target_folder,
            "preview": f"CREATE: {Path(target_folder).name}/",
            "editable": True,
            "item_id": series["id"],
            "item_type": "series",
        })

        # Create season subfolders if option enabled
        create_season_folders = options.get("create_season_folders", True)

        for ep in series["episodes"]:
            source = ep["filepath"]
            ext = Path(source).suffix

            title = series["detected_title"]
            year = series["detected_year"]
            season = ep.get("season", 1)
            episode = ep.get("episode", 0)
            ep_title = ep.get("episode_title", "")

            # Determine target folder
            if create_season_folders:
                season_folder = os.path.join(target_folder, f"Season {season:02d}")
                # Add create folder operation for season
                if not any(op["target"] == season_folder for op in operations):
                    operations.append({
                        "type": "create_folder",
                        "source": None,
                        "target": season_folder,
                        "preview": f"CREATE: {Path(target_folder).name}/Season {season:02d}/",
                        "editable": False,
                        "item_id": series["id"],
                        "item_type": "series",
                    })
                ep_target_folder = season_folder
            else:
                ep_target_folder = target_folder

            # Generate new filename
            if year:
                new_name = f"{title} ({year}) - S{season:02d}E{episode:02d}"
            else:
                new_name = f"{title} - S{season:02d}E{episode:02d}"

            if ep_title:
                new_name += f" - {ep_title}"

            # Add technical tags if option enabled
            if options.get("keep_technical_tags", True):
                tech = ep.get("technical", {})
                tags = []
                if tech.get("resolution"):
                    tags.append(tech["resolution"])
                if tech.get("hdr"):
                    tags.append(tech["hdr"])
                if tech.get("codec"):
                    tags.append(tech["codec"])
                if tags:
                    new_name += f" [{' '.join(tags)}]"

            new_name += ext
            target = os.path.join(ep_target_folder, new_name)

            # Handle duplicates
            if target in seen_targets:
                counter = 2
                while target in seen_targets:
                    base = new_name.rsplit('.', 1)[0]
                    target = os.path.join(ep_target_folder, f"{base} (v{counter}){ext}")
                    counter += 1

            seen_targets.add(target)

            operations.append({
                "type": "move_file",
                "source": source,
                "target": target,
                "preview": f"MOVE: {Path(source).name} -> {Path(target).name}",
                "editable": True,
                "item_id": series["id"],
                "item_type": "series",
            })

        # Cleanup old folders
        for folder in series["current_folders"]:
            if folder != target_folder and folder != analysis.get("root_path"):
                operations.append({
                    "type": "delete_folder_if_empty",
                    "source": folder,
                    "target": None,
                    "preview": f"DELETE (if empty): {Path(folder).name}/",
                    "editable": False,
                    "item_id": series["id"],
                    "item_type": "series",
                })

        return operations

    def execute_plan(self, operations: List[Dict], dry_run: bool = False) -> Dict[str, Any]:
        """
        Execute normalization plan.

        Args:
            operations: List of operations from generate_plan()
            dry_run: If True, only simulate without making changes

        Returns:
            Execution result with success/failure counts
        """
        results = {
            "success": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "dry_run": dry_run,
            "operations": [],
        }

        for op in operations:
            op_result = self._execute_operation(op, dry_run)
            results["operations"].append(op_result)

            if op_result["status"] == "success":
                results["success"] += 1
            elif op_result["status"] == "failed":
                results["failed"] += 1
                results["errors"].append(op_result["error"])
            else:
                results["skipped"] += 1

        return results

    def _execute_operation(self, op: Dict, dry_run: bool) -> Dict:
        """Execute a single operation."""
        result = {
            "type": op["type"],
            "source": op.get("source"),
            "target": op.get("target"),
            "status": "success",
            "error": None,
        }

        if dry_run:
            result["status"] = "dry_run"
            return result

        try:
            if op["type"] == "create_folder":
                target = Path(op["target"])
                target.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created folder: {target}")

            elif op["type"] == "move_file":
                source = Path(op["source"])
                target = Path(op["target"])

                if not source.exists():
                    result["status"] = "failed"
                    result["error"] = f"Source file not found: {source}"
                    return result

                # Create parent directory if needed
                target.parent.mkdir(parents=True, exist_ok=True)

                # Move file
                shutil.move(str(source), str(target))
                logger.info(f"Moved: {source} -> {target}")

            elif op["type"] == "delete_folder_if_empty":
                folder = Path(op["source"])
                if folder.exists() and folder.is_dir():
                    # Check if empty
                    if not any(folder.iterdir()):
                        folder.rmdir()
                        logger.info(f"Deleted empty folder: {folder}")
                    else:
                        result["status"] = "skipped"
                        result["error"] = "Folder not empty"
                else:
                    result["status"] = "skipped"

        except Exception as e:
            result["status"] = "failed"
            result["error"] = str(e)
            logger.error(f"Operation failed: {op['type']} - {e}")

        return result


# Singleton instance
_offline_normalizer = None


def get_offline_normalizer() -> OfflineNormalizer:
    """Get the singleton OfflineNormalizer instance."""
    global _offline_normalizer
    if _offline_normalizer is None:
        _offline_normalizer = OfflineNormalizer()
    return _offline_normalizer
