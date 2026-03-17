-- Ajoute les colonnes manquantes à wishlist (si table créée avec ancien schéma)
-- Exécuter dans Supabase > SQL Editor

ALTER TABLE wishlist ADD COLUMN IF NOT EXISTS media_type TEXT NOT NULL DEFAULT 'movie';
ALTER TABLE wishlist ADD COLUMN IF NOT EXISTS poster_path TEXT;
