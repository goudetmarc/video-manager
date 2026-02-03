# Intégration TMDB (comme Plex)

Plex identifie les films via des bases de métadonnées (The Movie Database, TheTVDB, etc.). On peut faire pareil avec l’**API TMDB** (gratuite) pour avoir un **identifiant de film** par fichier et regrouper correctement les versions.

## 1. Principe

- **Aujourd’hui** : regroupement heuristique (dossier parent, titre normalisé) → beaucoup de groupes, pas de vraie « identité film ».
- **Avec TMDB** : pour chaque fichier, on envoie le titre (et optionnellement l’année) à l’API TMDB → on récupère un **`tmdb_id`** (ex. 603 pour « The Matrix »). On regroupe ensuite **par `tmdb_id`** → une carte = un film TMDB, toutes les versions (720p, 1080p, 4K, etc.) dans la même carte.

## 2. Obtenir une clé API TMDB (gratuit)

1. Créer un compte sur [themoviedb.org](https://www.themoviedb.org/).
2. Aller dans [Paramètres → API](https://www.themoviedb.org/settings/api).
3. Demander une « API Key » (type « Developer »). Accepter les conditions.
4. Copier la clé **API Key (v3)** (pas le token Bearer v4 pour l’instant).

Tu peux ensuite la renseigner dans **Paramètres** de Video Manager (champ « Clé API TMDB »). Elle est stockée localement dans `settings.json` et n’est utilisée que pour appeler TMDB depuis ton serveur.

## 3. Ce qui est déjà en place

- **Paramètres** : le backend peut stocker une clé `tmdb_api_key` (optionnelle) dans `settings.json`.
- **Endpoint** : `GET /api/tmdb/search?query=Matrix&year=1999`  
  → appelle TMDB, retourne le meilleur résultat : `{ "tmdb_id": 603, "title": "The Matrix", "year": "1999" }` (ou vide si pas de clé / pas de résultat).

Cela permet de tester l’API et, plus tard, d’enrichir l’inventaire avec un `tmdb_id` par fichier.

## 4. Suite possible (à développer)

1. **Enrichissement après scan**  
   - Un bouton « Enrichir avec TMDB » (ou une étape automatique après le scan).  
   - Pour chaque fichier (ou chaque **titre normalisé unique** pour limiter les appels), appel à `GET /api/tmdb/search?query=...&year=...`.  
   - Stocker le `tmdb_id` (et éventuellement titre + année) dans l’inventaire (ex. champ `tmdb_id` / `tmdb_title` dans chaque ligne).

2. **Regroupement par TMDB**  
   - Dans l’UI, si des fichiers ont un `tmdb_id`, regrouper **par `tmdb_id`** (une carte = un film TMDB).  
   - Si pas de `tmdb_id`, garder le comportement actuel (dossier parent, titre normalisé).

3. **Limites TMDB**  
   - Quotas / rate limits : ne pas appeler l’API des milliers de fois d’un coup ; regrouper par titre unique, mettre en cache, et éventuellement faire des pauses entre les requêtes.

En résumé : **oui, on a besoin d’une API (type TMDB) pour identifier les films comme Plex**. La base est en place (paramètre + recherche) ; la prochaine étape serait d’enrichir l’inventaire avec `tmdb_id` et de regrouper par cet ID.
