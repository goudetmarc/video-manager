#!/bin/bash
# Double-cliquez pour lancer Video Manager. Le navigateur s'ouvrira sur l'app.

cd "$(dirname "$0")"
APP_DIR="$PWD"
export PATH="/opt/local/bin:$PATH"

if [ ! -d ".venv" ]; then
    echo "Double-cliquez d'abord sur « Install.command » pour installer."
    echo ""
    read -r -p "Appuyez sur Entrée pour fermer."
    exit 1
fi

source .venv/bin/activate

# Vérifier que les dépendances sont installées (ex. httpx)
if ! python -c "import httpx" 2>/dev/null; then
  echo "Installation des dépendances (première fois ou mise à jour)..."
  pip install -r requirements.txt -q
  if [ $? -ne 0 ]; then
    echo "Erreur lors de l'installation. Vérifiez votre connexion internet et réessayez."
    read -r -p "Appuyez sur Entrée pour fermer."
    exit 1
  fi
  echo "Dépendances installées."
fi

# Démarrer l'API en arrière-plan
echo "Démarrage de l'API (port 8000)..."
python -m uvicorn main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

# Attendre que l'API réponde (max 15 s)
API_OK=0
for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/api/health 2>/dev/null | grep -q 200; then
    API_OK=1
    break
  fi
  sleep 1
done

if [ "$API_OK" -ne 1 ]; then
  kill $UVICORN_PID 2>/dev/null
  echo ""
  echo "Erreur : l'API n'a pas démarré (port 8000)."
  echo "Vérifiez les messages ci-dessus (module manquant ?)."
  echo "Relancez « Install.command » si besoin, puis réessayez."
  echo ""
  read -r -p "Appuyez sur Entrée pour fermer."
  exit 1
fi
echo "API démarrée."

# Ouvrir le navigateur après 10 secondes (le temps que Next démarre)
( sleep 10 && open "http://localhost:3000" ) &

echo ""
echo "=============================================="
echo "  Video Manager"
echo "=============================================="
echo "  API : http://127.0.0.1:8000"
echo "  App  : http://localhost:3000 (le navigateur va s'ouvrir)"
echo "  Pour tout arrêter : fermez cette fenêtre."
echo "=============================================="
echo ""

# Installer le frontend si besoin (première fois)
if [ ! -d "$APP_DIR/frontend/node_modules" ]; then
    echo "Première fois : installation du frontend (npm install)..."
    (cd "$APP_DIR/frontend" && npm install)
    echo ""
fi

# Démarrer le frontend (reste au premier plan)
cd "$APP_DIR/frontend" && npm run dev

# À la fermeture : arrêter l'API
kill $UVICORN_PID 2>/dev/null
