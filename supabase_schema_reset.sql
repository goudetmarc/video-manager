-- =============================================================================
-- Video Manager - RESET & CREATE Schema
-- =============================================================================
-- This script DROPS all existing objects and recreates them from scratch
-- =============================================================================

-- Drop views first (they depend on tables)
DROP VIEW IF EXISTS continue_watching CASCADE;
DROP VIEW IF EXISTS enriched_movies CASCADE;
DROP VIEW IF EXISTS recent_movies CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS cleanup_watch_history() CASCADE;
DROP FUNCTION IF EXISTS get_inventory_stats() CASCADE;
DROP FUNCTION IF EXISTS update_updated_at_column() CASCADE;

-- Drop tables (CASCADE drops dependent triggers and indexes)
DROP TABLE IF EXISTS watch_history CASCADE;
DROP TABLE IF EXISTS playback_state CASCADE;
DROP TABLE IF EXISTS poster_cache CASCADE;
DROP TABLE IF EXISTS scan_metadata CASCADE;
DROP TABLE IF EXISTS movies CASCADE;

-- =============================================================================
-- Table: movies
-- =============================================================================
CREATE TABLE movies (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  path TEXT UNIQUE NOT NULL,
  name TEXT NOT NULL,
  size_bytes BIGINT,
  mtime TIMESTAMP,
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
  container TEXT,
  video_profile TEXT,
  video_bitrate INTEGER,
  audio_bitrate INTEGER,
  audio_languages TEXT[],
  metadata_title TEXT,
  tmdb_id INTEGER,
  tmdb_title TEXT,
  tmdb_year TEXT,
  poster_url TEXT,
  custom_group_key TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  scanned_at TIMESTAMP WITH TIME ZONE
);

CREATE INDEX idx_movies_tmdb_id ON movies(tmdb_id);
CREATE INDEX idx_movies_group_key ON movies(custom_group_key);
CREATE INDEX idx_movies_name ON movies(name);
CREATE INDEX idx_movies_created_at ON movies(created_at);

-- =============================================================================
-- Table: poster_cache
-- =============================================================================
CREATE TABLE poster_cache (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cache_key TEXT UNIQUE NOT NULL,
  poster_path TEXT,
  poster_url TEXT,
  tmdb_id INTEGER,
  tmdb_title TEXT,
  tmdb_year TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_poster_cache_key ON poster_cache(cache_key);
CREATE INDEX idx_poster_cache_tmdb_id ON poster_cache(tmdb_id);

-- =============================================================================
-- Table: playback_state
-- =============================================================================
CREATE TABLE playback_state (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_path TEXT UNIQUE NOT NULL,
  position FLOAT DEFAULT 0,
  duration FLOAT DEFAULT 0,
  volume FLOAT DEFAULT 1,
  playback_rate FLOAT DEFAULT 1,
  subtitle_track INTEGER,
  audio_track INTEGER,
  watched BOOLEAN DEFAULT FALSE,
  last_played TIMESTAMP WITH TIME ZONE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_playback_video_path ON playback_state(video_path);
CREATE INDEX idx_playback_last_played ON playback_state(last_played);

-- =============================================================================
-- Table: watch_history
-- =============================================================================
CREATE TABLE watch_history (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  video_path TEXT NOT NULL,
  position FLOAT DEFAULT 0,
  duration FLOAT DEFAULT 0,
  watched BOOLEAN DEFAULT FALSE,
  timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_watch_history_timestamp ON watch_history(timestamp DESC);
CREATE INDEX idx_watch_history_video_path ON watch_history(video_path);

-- =============================================================================
-- Table: scan_metadata
-- =============================================================================
CREATE TABLE scan_metadata (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  scanned_path TEXT NOT NULL,
  scanned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  file_count INTEGER DEFAULT 0,
  scan_type TEXT DEFAULT 'full',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_scan_metadata_scanned_at ON scan_metadata(scanned_at DESC);

-- =============================================================================
-- Trigger function for updated_at
-- =============================================================================
CREATE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_movies_updated_at
  BEFORE UPDATE ON movies
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_playback_state_updated_at
  BEFORE UPDATE ON playback_state
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Views
-- =============================================================================
CREATE VIEW recent_movies AS
SELECT * FROM movies
WHERE created_at > NOW() - INTERVAL '30 days'
ORDER BY created_at DESC;

CREATE VIEW enriched_movies AS
SELECT * FROM movies
WHERE tmdb_id IS NOT NULL
ORDER BY tmdb_title, tmdb_year;

CREATE VIEW continue_watching AS
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
CREATE FUNCTION cleanup_watch_history()
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

CREATE FUNCTION get_inventory_stats()
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
