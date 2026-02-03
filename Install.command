#!/bin/bash
# Double-cliquez pour installer les dépendances (venv + pip). Une seule fois.

cd "$(dirname "$0")"
APP_DIR="$PWD"

echo "=============================================="
echo "  Video Manager — Installation"
echo "=============================================="
echo ""

# Choisir un Python 3.12 ou 3.13 (pydantic ne supporte pas encore Python 3.14)
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
        if [ -n "$ver" ] && [ "$ver" -le 13 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Aucun Python 3.11, 3.12 ou 3.13 trouvé."
    echo "Python 3.14 n'est pas encore supporté par les dépendances (pydantic)."
    echo ""
    echo "Installez Python 3.12 ou 3.13 :"
    echo "  brew install python@3.12"
    echo "  (ou python@3.13)"
    echo ""
    echo "Si vous avez déjà Python 3.12/3.13, supprimez le dossier .venv puis relancez Install.command"
    echo ""
    echo "Appuyez sur Entrée pour fermer."
    read -r
    exit 1
fi

echo "Utilisation de : $PYTHON"
$PYTHON --version
echo ""

# Supprimer un ancien venv créé avec Python 3.14
if [ -d ".venv" ]; then
    VENV_PY=$(.venv/bin/python -c "import sys; print(sys.version_info.minor)" 2>/dev/null)
    if [ "$VENV_PY" = "14" ]; then
        echo "L'environnement .venv actuel utilise Python 3.14 (non supporté)."
        echo "Suppression de .venv pour recréer avec $PYTHON..."
        rm -rf .venv
    else
        echo "Environnement .venv déjà présent."
    fi
fi

# Créer le venv
if [ ! -d ".venv" ]; then
    echo "Création de l'environnement virtuel (.venv)..."
    $PYTHON -m venv .venv
fi

echo "Activation et installation des paquets..."
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

if [ $? -ne 0 ]; then
    echo ""
    echo "Erreur lors de l'installation. Vérifiez votre connexion et réessayez."
    echo "Appuyez sur Entrée pour fermer."
    read -r
    exit 1
fi

echo ""
echo "Installation terminée."
echo ""
echo "Pour lancer l'app : double-cliquez sur « Lancer Video Manager.command »"
echo ""
echo "FFmpeg est nécessaire pour le scan. Si besoin : brew install ffmpeg"
echo ""
echo "Appuyez sur Entrée pour fermer."
read -r
