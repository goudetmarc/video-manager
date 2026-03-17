#!/usr/bin/env python3
"""
Organize TV series: flatten folders, rename files, remove duplicates and junk.
Format: "Series Name - S01E01 - Episode Title.ext"
"""

import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

SERIES_ROOT = "/Volumes/series"
VIDEO_EXTS = {'.mkv', '.mp4', '.avi', '.m4v', '.wmv', '.mov'}
JUNK_EXTS = {'.nfo', '.txt', '.srt', '.sub', '.idx', '.jpg', '.jpeg', '.png', '.gif', '.sfv', '.md5'}
SAMPLE_PATTERNS = [r'\.sample\.', r'-sample\.', r'_sample\.', r'\bsample\b']

# Skip these folders
SKIP_FOLDERS = {'#recycle', '@eaDir', '.'}

def is_sample(filename):
    """Check if file is a sample."""
    lower = filename.lower()
    return any(re.search(p, lower) for p in SAMPLE_PATTERNS)

def parse_episode_info(filename):
    """Extract series name, season, episode, and title from filename."""
    # Common patterns: S01E01, 1x01, etc.
    patterns = [
        # S01E01 or S01E01E02
        r'^(.+?)[.\s_-]+[Ss](\d{1,2})[Ee](\d{1,2})(?:[Ee]\d{1,2})?[.\s_-]*(.*)$',
        # 1x01
        r'^(.+?)[.\s_-]+(\d{1,2})x(\d{1,2})[.\s_-]*(.*)$',
    ]
    
    base = os.path.splitext(filename)[0]
    
    for pattern in patterns:
        match = re.match(pattern, base, re.IGNORECASE)
        if match:
            series = match.group(1).replace('.', ' ').replace('_', ' ').strip()
            season = int(match.group(2))
            episode = int(match.group(3))
            title = match.group(4) if match.group(4) else ''
            
            # Clean up title - remove quality/codec info at the end
            title = re.sub(r'[.\s_-]*(1080p|720p|480p|2160p|WEB-DL|WEBRip|BluRay|HDTV|DVDRip|BDRip|REMUX|x264|x265|H\.?264|H\.?265|HEVC|AAC|DD5\.?1|DDP5\.?1|DTS|FLAC|LPCM|AVC|WEB|HDR).*$', '', title, flags=re.IGNORECASE)
            title = re.sub(r'[.\s_-]+(DEFLATE|NTb|NTG|ROVERS|SiGMA|ION10|STRiFE|KiNGS|Monkee|D3g|EPSiLON|FLUX|RARBG|BTN|DIMENSION|Lord123|MRN|AJP69|SbR|BLUTONiUM).*$', '', title, flags=re.IGNORECASE)
            title = title.replace('.', ' ').replace('_', ' ').strip(' -')
            
            # Remove year patterns like (2013) from series name
            series = re.sub(r'\s*\(\d{4}\)\s*$', '', series)
            series = re.sub(r'\s*\d{4}\s*$', '', series)
            
            return series, season, episode, title
    
    return None, None, None, None

def get_quality_score(filename):
    """Higher score = better quality."""
    lower = filename.lower()
    score = 0
    
    if '2160p' in lower or '4k' in lower:
        score += 1000
    elif '1080p' in lower:
        score += 500
    elif '720p' in lower:
        score += 200
    elif '480p' in lower:
        score += 100
    
    if 'remux' in lower:
        score += 50
    if 'bluray' in lower:
        score += 30
    elif 'web-dl' in lower:
        score += 25
    elif 'webrip' in lower:
        score += 20
    elif 'hdtv' in lower:
        score += 10
    elif 'dvdrip' in lower:
        score += 5
    
    if 'x265' in lower or 'h.265' in lower or 'hevc' in lower:
        score += 10  # More efficient codec
    
    return score

def clean_series_name(folder_name):
    """Get a clean series name from the folder name."""
    name = folder_name
    
    # Remove hash tags
    name = re.sub(r'\{\{.*\}\}', '', name)
    
    # Replace dots and underscores with spaces first
    name = name.replace('.', ' ').replace('_', ' ')
    
    # Remove everything from season/quality markers onwards
    # Match S01, Season 1, 1080p, 720p, WEB-DL, etc.
    name = re.sub(r'\s+(S\d{1,2}|Season\s*\d+)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(1080p|720p|480p|2160p|4K)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(WEB-DL|WEBRip|BluRay|HDTV|DVDRip|BDRip|REMUX|WEB)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(x264|x265|H\.?264|H\.?265|HEVC|AVC)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(INTEGRALE|COMPLETE|INTÉGRALE)\b.*$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'\s+(FRENCH|ENGLISH|VOSTFR|MULTI|MULTi|DUAL)\b.*$', '', name, flags=re.IGNORECASE)
    
    # Remove trailing years
    name = re.sub(r'\s*\(\d{4}\)\s*$', '', name)  # (2013)
    name = re.sub(r'\s+\d{4}\s*$', '', name)  # 2013
    
    # Remove trailing "265" (like "Buffy 265")
    name = re.sub(r'\s+265\s*$', '', name)
    
    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Title case for consistency (optional - comment out if you prefer original case)
    # name = name.title()
    
    return name

def normalize_series_name(name):
    """Normalize a series name for grouping similar folders."""
    n = name.lower()
    # Remove quality/codec info
    n = re.sub(r'[.\s_-]*(s\d{1,2}|season\s*\d+).*$', '', n, flags=re.IGNORECASE)
    n = re.sub(r'[.\s_-]*(1080p|720p|480p|2160p|4k|web-dl|webrip|bluray|hdtv|dvdrip|remux|x264|x265|h\.?264|h\.?265|hevc|french|english|vostfr|multi|complete|integrale|intégrale).*$', '', n, flags=re.IGNORECASE)
    # Remove years
    n = re.sub(r'\s*\(?\d{4}\)?', '', n)
    # Remove release group tags
    n = re.sub(r'\{\{.*\}\}', '', n)
    # Replace separators with spaces
    n = re.sub(r'[.\s_-]+', ' ', n).strip()
    return n

def find_best_folder_name(folders):
    """Pick the cleanest folder name from a group."""
    # Prefer shorter names without dots/underscores, without quality info
    def score(name):
        s = 0
        if '.' in name or '_' in name:
            s -= 10
        if re.search(r'(1080p|720p|WEB|BluRay|x264|x265)', name, re.IGNORECASE):
            s -= 20
        if re.search(r'S\d{2}', name):
            s -= 15  # Has season in folder name
        s -= len(name) / 10  # Prefer shorter
        return s
    
    return max(folders, key=score)

def main():
    stats = {
        'videos_found': 0,
        'samples_deleted': 0,
        'junk_deleted': 0,
        'duplicates_removed': 0,
        'files_moved': 0,
        'files_renamed': 0,
        'folders_removed': 0,
        'folders_merged': 0,
        'errors': []
    }
    
    # Phase 0: Merge folders for the same series
    print("=== Phase 0: Merging series folders ===")
    all_folders = [d for d in os.listdir(SERIES_ROOT) 
                   if os.path.isdir(os.path.join(SERIES_ROOT, d)) and d not in SKIP_FOLDERS]
    
    # Group by normalized name
    folder_groups = defaultdict(list)
    for folder in all_folders:
        normalized = normalize_series_name(folder)
        if normalized:  # Skip empty names
            folder_groups[normalized].append(folder)
    
    # Merge groups with multiple folders
    for normalized, folders in folder_groups.items():
        if len(folders) > 1:
            best_name = find_best_folder_name(folders)
            best_path = os.path.join(SERIES_ROOT, best_name)
            print(f"\n  Merging {len(folders)} folders for '{normalized}':")
            print(f"    Target: {best_name}")
            
            for folder in folders:
                if folder == best_name:
                    continue
                folder_path = os.path.join(SERIES_ROOT, folder)
                print(f"    <- {folder}")
                
                # Move all contents to the best folder
                for item in os.listdir(folder_path):
                    if item.startswith('.'):
                        continue
                    src = os.path.join(folder_path, item)
                    dest = os.path.join(best_path, item)
                    
                    # Handle name conflicts
                    if os.path.exists(dest):
                        if os.path.isdir(src) and os.path.isdir(dest):
                            # Merge directories recursively by moving contents
                            for sub_item in os.listdir(src):
                                sub_src = os.path.join(src, sub_item)
                                sub_dest = os.path.join(dest, sub_item)
                                if not os.path.exists(sub_dest):
                                    try:
                                        shutil.move(sub_src, sub_dest)
                                    except Exception as e:
                                        stats['errors'].append(f"Error moving {sub_src}: {e}")
                        elif os.path.isfile(src) and os.path.isfile(dest):
                            # Keep larger file
                            if os.path.getsize(src) > os.path.getsize(dest):
                                os.remove(dest)
                                shutil.move(src, dest)
                            # else keep dest, src will be cleaned up
                    else:
                        try:
                            shutil.move(src, dest)
                        except Exception as e:
                            stats['errors'].append(f"Error moving {src}: {e}")
                
                # Remove the now-empty folder
                try:
                    shutil.rmtree(folder_path)
                    stats['folders_merged'] += 1
                except Exception as e:
                    stats['errors'].append(f"Error removing folder {folder_path}: {e}")
    
    # First pass: delete samples and junk
    print("=== Phase 1: Deleting samples and junk files ===")
    for root, dirs, files in os.walk(SERIES_ROOT):
        # Skip special folders
        dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]
        
        for f in files:
            filepath = os.path.join(root, f)
            ext = os.path.splitext(f)[1].lower()
            
            # Delete junk files
            if ext in JUNK_EXTS:
                try:
                    os.remove(filepath)
                    stats['junk_deleted'] += 1
                    print(f"  Deleted junk: {f}")
                except Exception as e:
                    stats['errors'].append(f"Error deleting {filepath}: {e}")
                continue
            
            # Delete samples
            if ext in VIDEO_EXTS and is_sample(f):
                try:
                    os.remove(filepath)
                    stats['samples_deleted'] += 1
                    print(f"  Deleted sample: {f}")
                except Exception as e:
                    stats['errors'].append(f"Error deleting {filepath}: {e}")
    
    # Second pass: collect all video files per series folder
    print("\n=== Phase 2: Collecting video files ===")
    series_folders = [d for d in os.listdir(SERIES_ROOT) 
                      if os.path.isdir(os.path.join(SERIES_ROOT, d)) and d not in SKIP_FOLDERS]
    
    for series_folder in sorted(series_folders):
        series_path = os.path.join(SERIES_ROOT, series_folder)
        series_name = clean_series_name(series_folder)
        
        print(f"\nProcessing: {series_folder}")
        
        # Collect all video files in this series folder (recursively)
        videos = []
        for root, dirs, files in os.walk(series_path):
            dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in VIDEO_EXTS and not is_sample(f):
                    filepath = os.path.join(root, f)
                    size = os.path.getsize(filepath)
                    videos.append({
                        'path': filepath,
                        'filename': f,
                        'size': size,
                        'ext': ext,
                        'quality': get_quality_score(f)
                    })
        
        stats['videos_found'] += len(videos)
        
        # Group by episode (season, episode)
        episodes = defaultdict(list)
        for v in videos:
            _, season, episode, title = parse_episode_info(v['filename'])
            if season is not None:
                key = (season, episode)
                v['parsed_title'] = title
                v['parsed_season'] = season
                v['parsed_episode'] = episode
                episodes[key].append(v)
            else:
                # Can't parse - just move as-is
                episodes[('unparsed', v['filename'])].append(v)
        
        # For each episode, keep best version and move to series root
        for key, versions in episodes.items():
            if key[0] == 'unparsed':
                # Can't parse, just move to root if not there
                v = versions[0]
                if os.path.dirname(v['path']) != series_path:
                    dest = os.path.join(series_path, v['filename'])
                    if not os.path.exists(dest):
                        try:
                            shutil.move(v['path'], dest)
                            stats['files_moved'] += 1
                        except Exception as e:
                            stats['errors'].append(f"Error moving {v['path']}: {e}")
                continue
            
            season, episode = key
            
            # Sort by quality score desc, then size desc
            versions.sort(key=lambda x: (x['quality'], x['size']), reverse=True)
            
            # Keep the best one
            best = versions[0]
            
            # Delete duplicates
            for v in versions[1:]:
                try:
                    os.remove(v['path'])
                    stats['duplicates_removed'] += 1
                    print(f"  Removed duplicate: {v['filename']}")
                except Exception as e:
                    stats['errors'].append(f"Error removing duplicate {v['path']}: {e}")
            
            # Build new filename
            title_part = f" - {best['parsed_title']}" if best.get('parsed_title') else ""
            new_filename = f"{series_name} - S{season:02d}E{episode:02d}{title_part}{best['ext']}"
            new_filename = re.sub(r'[<>:"/\\|?*]', '', new_filename)  # Remove invalid chars
            
            dest_path = os.path.join(series_path, new_filename)
            
            # Move/rename
            if best['path'] != dest_path:
                if os.path.exists(dest_path):
                    # Destination exists - keep larger file
                    if os.path.getsize(dest_path) >= best['size']:
                        try:
                            os.remove(best['path'])
                            stats['duplicates_removed'] += 1
                        except:
                            pass
                        continue
                    else:
                        try:
                            os.remove(dest_path)
                        except:
                            pass
                
                try:
                    shutil.move(best['path'], dest_path)
                    if os.path.dirname(best['path']) != series_path:
                        stats['files_moved'] += 1
                    stats['files_renamed'] += 1
                    print(f"  Renamed: {os.path.basename(best['path'])} -> {new_filename}")
                except Exception as e:
                    stats['errors'].append(f"Error moving {best['path']}: {e}")
    
    # Third pass: remove empty directories
    print("\n=== Phase 3: Removing empty directories ===")
    for series_folder in sorted(series_folders):
        series_path = os.path.join(SERIES_ROOT, series_folder)
        
        # Walk bottom-up to remove empty dirs
        for root, dirs, files in os.walk(series_path, topdown=False):
            dirs[:] = [d for d in dirs if d not in SKIP_FOLDERS]
            
            if root == series_path:
                continue
                
            # Check if directory is empty (or only has hidden files)
            remaining = [f for f in os.listdir(root) if not f.startswith('.')]
            if not remaining:
                try:
                    shutil.rmtree(root)
                    stats['folders_removed'] += 1
                    print(f"  Removed empty folder: {root}")
                except Exception as e:
                    stats['errors'].append(f"Error removing folder {root}: {e}")
    
    # Fourth pass: rename ugly series folders
    print("\n=== Phase 4: Renaming ugly series folders ===")
    current_folders = [d for d in os.listdir(SERIES_ROOT) 
                       if os.path.isdir(os.path.join(SERIES_ROOT, d)) and d not in SKIP_FOLDERS]
    
    for folder in sorted(current_folders):
        clean = clean_series_name(folder)
        
        # Check if folder name needs cleaning
        if clean != folder and clean:
            old_path = os.path.join(SERIES_ROOT, folder)
            new_path = os.path.join(SERIES_ROOT, clean)
            
            # If clean name already exists, merge instead
            if os.path.exists(new_path):
                print(f"  Merging '{folder}' into existing '{clean}'")
                for item in os.listdir(old_path):
                    if item.startswith('.'):
                        continue
                    src = os.path.join(old_path, item)
                    dest = os.path.join(new_path, item)
                    if not os.path.exists(dest):
                        try:
                            shutil.move(src, dest)
                        except Exception as e:
                            stats['errors'].append(f"Error moving {src}: {e}")
                try:
                    shutil.rmtree(old_path)
                    stats['folders_renamed_count'] = stats.get('folders_renamed_count', 0) + 1
                except Exception as e:
                    stats['errors'].append(f"Error removing {old_path}: {e}")
            else:
                try:
                    os.rename(old_path, new_path)
                    stats['folders_renamed_count'] = stats.get('folders_renamed_count', 0) + 1
                    print(f"  Renamed: '{folder}' -> '{clean}'")
                except Exception as e:
                    stats['errors'].append(f"Error renaming {old_path}: {e}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Videos found: {stats['videos_found']}")
    print(f"Samples deleted: {stats['samples_deleted']}")
    print(f"Junk files deleted: {stats['junk_deleted']}")
    print(f"Duplicates removed: {stats['duplicates_removed']}")
    print(f"Files moved: {stats['files_moved']}")
    print(f"Files renamed: {stats['files_renamed']}")
    print(f"Empty folders removed: {stats['folders_removed']}")
    print(f"Folders merged: {stats['folders_merged']}")
    print(f"Folders renamed: {stats.get('folders_renamed_count', 0)}")
    
    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for e in stats['errors'][:20]:
            print(f"  - {e}")
        if len(stats['errors']) > 20:
            print(f"  ... and {len(stats['errors']) - 20} more")

if __name__ == '__main__':
    main()
