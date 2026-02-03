# Video Manager — Description du projet et prompt pour Claude AI

## 1. Description du projet à ce jour

### Objectif
**Video Manager** est une application **locale** (web) qui permet de :
- **Scanner** un dossier vidéo (ex. NAS) une seule fois ;
- **Enregistrer** toutes les métadonnées dans un inventaire persistant (`inventory.json`) ;
- **Regrouper** les fichiers par « même film » (titre normalisé + durée proche) pour repérer les doublons ;
- **Afficher** les groupes sous forme de **grandes cartes** : un rectangle par film, avec le **fichier le plus lourd en premier** et les autres versions en dessous ;
- Aider l’utilisateur à décider quelles versions garder (l’app **ne supprime rien**).

### Stack technique
- **Backend** : Python 3.11/3.12/3.13, FastAPI, `uvicorn` (port 8000). CORS autorise `localhost:3000` et `127.0.0.1:3000`.
- **Frontend** : Next.js (App Router, TypeScript, Tailwind), port 3000. Appels API vers `http://127.0.0.1:8000` par défaut.
- **Métadonnées** : **ffprobe** uniquement (pas d’analyse d’image). Extensions : mkv, mp4, avi, mov, wmv, m4v, webm, mpg, mpeg.
- **Données** : `settings.json` (chemin vidéo), `inventory.json` (inventaire complet après scan).

### Lancement (utilisateur final)
- **Première fois** : double-clic sur **`Install.command`** (venv + pip + optionnellement `npm install` dans `frontend/` au premier lancement).
- **Ensuite** : double-clic sur **`Lancer Video Manager.command`** → démarre l’API (8000) puis le frontend Next.js (3000), ouvre le navigateur sur http://localhost:3000. Fermer la fenêtre Terminal arrête tout.

### Backend (racine du projet)
- **`main.py`** : FastAPI, CORS, routes `/api/settings`, `/api/inventory`, `/api/scan`, `/api/scan-stream` (NDJSON : `started` → `file` × N → `done` avec `total_expected`), `/api/ffmpeg`, `/api/health`. Servage de `static/` (ancienne UI HTML).
- **`scanner.py`** : `get_video_files`, `get_video_files_count`, `get_folder_diagnostic`, `run_ffprobe`, `parse_full_metadata`, `scan_and_build_inventory`, `scan_and_build_inventory_stream` (yield fichier par fichier pour affichage en direct).
- **`config.py`** : réglages (ex. `video_root`).

### Frontend (`frontend/`)
- **Pages** : `/` → redirection vers `/inventaire` ; **`/inventaire`** (scan, filtre, cartes par film) ; **`/parametres`** (chemin vidéo, statut FFmpeg).
- **Layout** : sidebar (Inventaire, Paramètres), thème sombre (variables CSS `--bg`, `--surface`, `--accent`, etc.).
- **Inventaire** : zone de scan (chemin, bouton « Scanner et enregistrer l’inventaire »), statut X/Y pendant le flux ; **cartes** = un grand rectangle par film, avec titre + liste de fichiers **triés par poids décroissant** (plus lourd en premier) ; badges (codec, résolution, HDR, conteneur), durées en h/min/s.
- **Logique doublons** : `lib/duplicates.ts` — `normalizeTitleForGroup`, `durationMatch`, `buildDuplicateGroups` (attribue `filmGroupKey` et `duplicateGroup`), `getFilmGroups` (retourne des groupes triés par titre, fichiers dans chaque groupe triés par `size_bytes` décroissant). Aucun fichier ne doit être omis (safeguard pour fichiers « orphelins »).
- **API client** : `lib/api.ts` — `getSettings`, `saveSettings`, `getInventory`, `getFfmpegStatus`, `scanStream(path, onMessage)` (lecture NDJSON en flux).
- **Types** : `types/inventory.ts` — `FileRow`, `AudioTrack`, `InventoryResponse`, etc., avec `duplicateGroup` et `filmGroupKey` optionnels.
- **Formatters** : `lib/formatters.ts` — durées, tailles, bitrate, résolution (badges 720p–8K), codec, HDR/Dolby Vision/SDR.

### Fichiers importants
- **`Lancer Video Manager.command`** : lance l’API en arrière-plan, puis `npm run dev` dans `frontend/`, ouvre le navigateur après ~10 s ; installe `node_modules` si besoin.
- **`static/index.html`** : ancienne UI (tableau complet) ; référence pour la logique si besoin.
- **`inventory.json`** : généré après un scan ; contient `scanned_path`, `scanned_at`, `files` (tableau de `FileRow`).

### Comportement clé
- Un **nouveau scan remplace entièrement** l’inventaire précédent.
- Le **scan en flux** envoie d’abord `total_expected`, puis chaque fichier au fur et à mesure (NDJSON), puis `done` ; le frontend met à jour les cartes en direct.
- **Regroupement « même film »** : clé = titre normalisé (sans année, 720p/1080p, etc.) + regroupement par durée proche (±2,5 %) ; chaque fichier a un `filmGroupKey` ; les groupes sont affichés en cartes (un rectangle par film).
- Affichage des stats : **« X fichiers · Y groupes (films) »** pour éviter la confusion entre nombre de fichiers et nombre de cartes.

---

## 2. Prompt pour expliquer le projet à Claude AI

Tu peux copier-coller le bloc ci-dessous dans une nouvelle conversation avec Claude pour lui donner le contexte du projet.

```
Je travaille sur **Video Manager**, une app locale (web) pour gérer un inventaire vidéo et repérer les doublons de films.

**Contexte technique**
- **Backend** : FastAPI (Python 3.11–3.13) sur le port 8000. Fichiers principaux : `main.py`, `scanner.py`, `config.py`. Données : `settings.json`, `inventory.json`.
- **Frontend** : Next.js (App Router, TypeScript, Tailwind) dans le dossier `frontend/`, port 3000. API appelée en `http://127.0.0.1:8000`.
- **Lancement** : sur macOS, double-clic sur `Lancer Video Manager.command` lance l’API puis le frontend et ouvre le navigateur. Pas besoin de taper de commandes.

**Fonctionnement**
- Scan d’un dossier vidéo (ex. NAS) **une fois** ; les métadonnées sont extraites avec **ffprobe** (pas d’analyse d’image) et sauvegardées dans `inventory.json`.
- Les fichiers sont regroupés par « même film » (titre normalisé + durée proche). L’UI affiche des **grandes cartes** : un rectangle par film, avec le **fichier le plus lourd en premier** et les autres versions en dessous.
- Endpoint de scan en flux : `POST /api/scan-stream` (body : `{ "path": "..." }`), réponse NDJSON avec `type: started | file | done`, `total_expected`, et pour chaque `file` un objet `data` (une ligne d’inventaire).

**Structure frontend**
- Pages : `/inventaire` (scan, filtre, cartes par film), `/parametres` (chemin vidéo, FFmpeg).
- `frontend/lib/duplicates.ts` : `buildDuplicateGroups`, `getFilmGroups` (groupes triés par titre, fichiers dans chaque groupe triés par taille décroissante).
- `frontend/lib/api.ts` : `getSettings`, `saveSettings`, `getInventory`, `getFfmpegStatus`, `scanStream(path, onMessage)`.
- `frontend/types/inventory.ts` : types `FileRow`, `InventoryResponse`, etc. (`filmGroupKey`, `duplicateGroup` optionnels).

Quand tu proposes des changements, garde la cohérence avec ce qui existe (regroupement par film, affichage en cartes, pas de suppression automatique de fichiers). Le chemin du projet sur ma machine est : `Documents/PROJECTS/video-manager/`.
```

---

*Document généré pour faciliter la reprise du projet et l’onboarding d’un assistant IA (Claude).*
