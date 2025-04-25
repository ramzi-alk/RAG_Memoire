FROM python:3.9-slim

WORKDIR /app

# Installation des dépendances système nécessaires pour PyMuPDF et Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-fra \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copie des fichiers de dépendances
COPY requirements.txt .

# Installation des dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copie du code source
COPY src/ ./src/
COPY gui.py main.py .
COPY config.json ./config.json

# Création des répertoires nécessaires
RUN mkdir -p data chroma_db documents

# Variable d'environnement pour la clé API Gemini
# À remplacer lors du déploiement ou via un .env monté en volume
ENV GEMINI_API_KEY="votre_clé_api"

# Commande par défaut - Lancement de l'application en mode CLI
# Pour l'interface graphique, il faudra utiliser un environnement avec support GUI
CMD ["python", "main.py", "init", "--data-dir", "./data", "--db-dir", "./chroma_db", "--collection", "docss"]

# Pour utiliser l'interface graphique, on peut lancer le container avec:
# docker run -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix votre-image-rag python gui.py 