-- =============================================================================
-- Video Manager - Supabase Database Schema
-- =============================================================================
-- Execute this SQL in the Supabase SQL Editor to create the required tables
-- =============================================================================

-- =============================================================================
-- Table: movies
-- Stores movie file information from scans, including ffprobe metadata and TMDB data
-- =============================================================================
CREATE TABLE IF NOT EXISTS movies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- File information
  path TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  size_bytes BIGINT,
  mtime TIMESTAMP,

  -- Video metadata (from ffprobe)
  duration FLOAT,
  width INTEGER,
  height INTEGER,
  codec TEXT,
  codec_profile TEXT,
  audio_codec TEXT,
  audio_channels INTEGER,
  bitrate INTEGER,
  fps FLOAT,
  hdr BOOLEAN DEFAULT FALSE,

  -- Extended metadata
  container TEXT,
  video_profile TEXT,
  video_bitrate INTEGER,
  audio_bitrate INTEGER,
  audio_languages TEXT[], -- Array of language codes
  metadata_title TEXT,

  -- TMDB enrichment
  tmdb_id INTEGER,
  tmdb_title TEXT,
  tmdb_year TEXT,
  poster_url TEXT,

  -- Grouping (for duplicate detection)
  custom_group_key TEXT,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  scanned_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_movies_tmdb_id ON movies(tmdb_id);
CREATE INDEX IF NOT EXISTS idx_movies_group_key ON movies(custom_group_key);
CREATE INDEX IF NOT EXISTS idx_movies_name ON movies(name);
CREATE INDEX IF NOT EXISTS idx_movies_created_at ON movies(created_at);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

DROP TRIGGER IF EXISTS update_movies_updated_at ON movies;
CREATE TRIGGER update_movies_updated_at
  BEFORE UPDATE ON movies
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Table: poster_cache
-- Caches TMDB poster lookup results to avoid repeated API calls
-- =============================================================================
CREATE TABLE IF NOT EXISTS poster_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Cache key format: "title|year" (lowercase, trimmed)
  cache_key TEXT UNIQUE NOT NULL,

  -- TMDB data
  poster_path TEXT,           -- TMDB poster path (e.g., "/abc123.jpg")
  poster_url TEXT,            -- Full URL or local path
  tmdb_id INTEGER,
  tmdb_title TEXT,
  tmdb_year TEXT,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for cache lookups
CREATE INDEX IF NOT EXISTS idx_poster_cache_key ON poster_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_poster_cache_tmdb_id ON poster_cache(tmdb_id);

-- =============================================================================
-- Table: playback_state
-- Stores video playback state for resume functionality
-- =============================================================================
CREATE TABLE IF NOT EXISTS playback_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Video identifier (file path)
  video_path TEXT UNIQUE NOT NULL,

  -- Playback position
  position FLOAT DEFAULT 0,
  duration FLOAT DEFAULT 0,

  -- Player preferences
  volume FLOAT DEFAULT 1,
  playback_rate FLOAT DEFAULT 1,
  subtitle_track INTEGER,
  audio_track INTEGER,

  -- Watch status
  watched BOOLEAN DEFAULT FALSE,
  last_played TIMESTAMP WITH TIME ZONE,

  -- Timestamps
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for lookups
CREATE INDEX IF NOT EXISTS idx_playback_video_path ON playback_state(video_path);
CREATE INDEX IF NOT EXISTS idx_playback_last_played ON playback_state(last_played);

-- Trigger for updated_at
DROP TRIGGER IF EXISTS update_playback_state_updated_at ON playback_state;
CREATE TRIGGER update_playback_state_updated_at
  BEFORE UPDATE ON playback_state
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Table: watch_history
-- Stores watch history for the "continue watching" feature
-- =============================================================================
CREATE TABLE IF NOT EXISTS watch_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  -- Video identifier
  video_path TEXT NOT NULL,

  -- State at time of watching
  position FLOAT DEFAULT 0,
  duration FLOAT DEFAULT 0,
  watched BOOLEAN DEFAULT FALSE,

  -- Timestamp
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for recent history queries
CREATE INDEX IF NOT EXISTS idx_watch_history_timestamp ON watch_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_watch_history_video_path ON watch_history(video_path);

-- =============================================================================
-- Table: scan_metadata
-- Stores metadata about scans (path, timestamp)
-- =============================================================================
CREATE TABLE IF NOT EXISTS scan_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

  scanned_path TEXT NOT NULL,
  scanned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  file_count INTEGER DEFAULT 0,

  -- Type: 'full' or 'raw'
  scan_type TEXT DEFAULT 'full',

  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for latest scan lookup
CREATE INDEX IF NOT EXISTS idx_scan_metadata_scanned_at ON scan_metadata(scanned_at DESC);

-- =============================================================================
-- Row Level Security (RLS) - Optional
-- Uncomment if you need multi-user support
-- =============================================================================
-- ALTER TABLE movies ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE poster_cache ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE playback_state ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE watch_history ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE scan_metadata ENABLE ROW LEVEL SECURITY;

-- =============================================================================
-- Views - Useful queries
-- =============================================================================

-- View: Recent movies (last 30 days)
CREATE OR REPLACE VIEW recent_movies AS
SELECT * FROM movies
WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC;

-- View: Movies with TMDB data
CREATE OR REPLACE VIEW enriched_movies AS
SELECT * FROM movies
WHERE tmdb_id IS NOT NULL
ORDER BY tmdb_title, tmdb_year;

-- View: Continue watching (in progress videos)
CREATE OR REPLACE VIEW continue_watching AS
SELECT
  ps.video_path,
  ps.position,
  ps.duration,
  ps.last_played,
  m.name,
  m.tmdb_title,
  m.poster_url,
  ROUND((ps.position / NULLIF(ps.duration, 0)) * 100) as progress_percent
FROM playback_state ps
LEFT JOIN movies m ON m.path = ps.video_path
WHERE ps.watched = FALSE
  AND ps.position > 0
  AND ps.duration > 0
  AND (ps.position / ps.duration) < 0.95
ORDER BY ps.last_played DESC
LIMIT 20;

-- =============================================================================
-- Functions
-- =============================================================================

-- Function to clean up old watch history (keep last 100 entries per video)
CREATE OR REPLACE FUNCTION cleanup_watch_history()
RETURNS void AS $$
BEGIN
  DELETE FROM watch_history
  WHERE id NOT IN (
    SELECT id FROM watch_history
    ORDER BY timestamp DESC
    LIMIT 1000
  );
END;
$$ LANGUAGE plpgsql;

-- Function to get inventory stats
CREATE OR REPLACE FUNCTION get_inventory_stats()
RETURNS TABLE(
  total_movies BIGINT,
  total_size_bytes BIGINT,
  enriched_count BIGINT,
  avg_duration FLOAT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    COUNT(*)::BIGINT as total_movies,
    COALESCE(SUM(size_bytes), 0)::BIGINT as total_size_bytes,
    COUNT(CASE WHEN tmdb_id IS NOT NULL THEN 1 END)::BIGINT as enriched_count,
    AVG(duration)::FLOAT as avg_duration
  FROM movies;
END;
$$ LANGUAGE plpgsql;
