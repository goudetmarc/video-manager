-- Streaming and wishlist tables for Video Manager (TMDB-based)
-- Run this in Supabase SQL Editor if migrations are not automated

-- streaming_cache: cache optionnel TMDB (TTL 6h)
CREATE TABLE IF NOT EXISTS streaming_cache (
    platform TEXT NOT NULL,
    content_type TEXT NOT NULL,
    page INTEGER NOT NULL,
    data JSONB NOT NULL,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (platform, content_type, page)
);

CREATE INDEX IF NOT EXISTS idx_streaming_cache_time ON streaming_cache(cached_at);

-- wishlist: liste de souhaits utilisateur
CREATE TABLE IF NOT EXISTS wishlist (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT,
    tmdb_id INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    platform TEXT NOT NULL,
    title TEXT NOT NULL,
    poster_path TEXT,
    added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, tmdb_id, platform)
);

CREATE INDEX IF NOT EXISTS idx_wishlist_user ON wishlist(user_id);
