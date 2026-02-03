-- Migration: Ajouter TOUS les champs TMDB à la table movies
-- À exécuter dans la console Supabase > SQL Editor

-- Champs TMDB manquants
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

-- Index pour les requêtes fréquentes
CREATE INDEX IF NOT EXISTS idx_movies_media_type ON movies(media_type);
CREATE INDEX IF NOT EXISTS idx_movies_genre_ids ON movies USING GIN(genre_ids);
CREATE INDEX IF NOT EXISTS idx_movies_tmdb_original_title ON movies(tmdb_original_title);

-- Vérification
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'movies'
ORDER BY ordinal_position;
