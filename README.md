# Video Manager

Application **locale** (page web) qui scanne un dossier vidéo (ex. NAS), identifie les **doublons de films** (même film, plusieurs fichiers) via les métadonnées, et recommande la **meilleure version** (qualité). Vous décidez ensuite de supprimer les autres.

- **Pas de terminal** : lancement par double-clic.
- **UI** : Paramètres (chemin NAS, statut FFmpeg) et Scan (doublons + version recommandée).
- **Métadonnées uniquement** : ffprobe (fourni avec FFmpeg).

---

## Lancer l’app

1. **Première fois** : double-cliquez sur **`Install.command`** (dans le Finder).  
   Une fenêtre s’ouvre ; à la fin, appuyez sur Entrée pour fermer.

2. **Ensuite** : double-cliquez sur **`Lancer Video Manager.command`** (dans le Finder).  
   Une fenêtre s’ouvre, le navigateur s’ouvre sur l’app. Pour arrêter : fermez cette fenêtre.

---

## Prérequis

- **macOS** (scripts `.command` prévus pour Mac).
- **Python 3.11, 3.12 ou 3.13** (Python 3.14 n’est pas encore supporté par pydantic). Si besoin : `brew install python@3.12`.
- **FFmpeg** pour le scan : l’app affiche dans Paramètres si FFmpeg est installé et comment l’installer (ex. `brew install ffmpeg`).

---

## Utilisation

1. **Paramètres** (menu latéral) : saisissez le **chemin du dossier vidéo** (NAS, ex. `/Volumes/MonNAS/Videos`) et cliquez sur **Enregistrer**. Vérifiez le statut **FFmpeg** (installé / non installé + instructions).
2. **Scan & doublons** : cliquez sur **Lancer le scan**. Le chemin enregistré est utilisé ; vous pouvez en saisir un autre dans le champ si besoin. Les groupes de doublons s’affichent avec la version **Recommandée** ; les autres sont marquées **Doublon**. Vous supprimez manuellement les fichiers que vous ne voulez pas garder (l’app ne supprime rien).

---

## Comportement technique

- **Scan** : liste récursive des fichiers vidéo (mkv, mp4, avi, etc.).
- **Métadonnées** : `ffprobe` donne résolution, durée, bitrate, codec.
- **Regroupement** : même film = titre normalisé (nom sans 720p/1080p, année, etc.) + durées proches (±2 %).
- **Qualité** : score (résolution, bitrate, bonus HEVC) pour désigner la version recommandée.
- **Paramètres** : chemin NAS enregistré dans `settings.json` à la racine du projet.

---

## Lancement manuel (terminal)

Si vous préférez lancer depuis le terminal :

```bash
cd video-manager
# Première fois :
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
# Puis :
source .venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 8000
```

Ouvrez **http://127.0.0.1:8000**.
