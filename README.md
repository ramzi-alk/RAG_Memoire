# Système RAG en Python

Ce projet implémente un système RAG (Retrieval-Augmented Generation) complet en Python, utilisant des outils modernes pour l'extraction de texte, la génération d'embeddings, le stockage vectoriel et la génération de réponses basées sur une collection de documents PDF.

## Architecture du système

Le système est composé de plusieurs modules interconnectés :

1. **Extraction de texte** : Utilise PyMuPDF pour extraire le texte des fichiers PDF avec support OCR pour les documents scannés
2. **Prétraitement et embeddings** : Utilise Sentence Transformers pour générer des représentations vectorielles du texte
3. **Base de données vectorielle** : Utilise ChromaDB en mode persistant pour stocker et rechercher les embeddings
4. **Modèle de langage** : Utilise l'API Gemini-2.0-Flash-001 pour générer des réponses
5. **Pipeline RAG** : Intègre tous ces composants dans un système cohérent
6. **Interface graphique** : Fournit une interface utilisateur simple et intuitive avec un historique des requêtes

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd RAG

# Installer les dépendances
pip install -r requirements.txt

# Configurer la clé API Gemini
# Option 1: Créer un fichier .env avec GEMINI_API_KEY=votre_clé_api
# Option 2: La spécifier dans l'interface graphique
```

## Lancement de l'application

### Interface graphique

Pour lancer l'interface graphique de l'application :

```bash
python gui.py
```

L'interface vous permet de :
- Configurer l'API Gemini et les paramètres du système
- Traiter des documents PDF (répertoire entier ou fichiers individuels)
- Effectuer des recherches dans les documents
- Consulter et réutiliser l'historique des requêtes

### Interface en ligne de commande

Le système peut également être utilisé via une interface en ligne de commande :

```bash
# Initialisation
python main.py init --api-key votre_clé_api --data-dir ./data --db-dir ./chroma_db --collection docss

# Traitement de PDF
python main.py process --pdf-dir ./docs --recursive

# Traitement d'un seul fichier
python main.py process --pdf-file ./docs/document.pdf

# Exécution d'une requête
python main.py query --query "Quelle est la définition de la qualité des données?"
```

## Création d'un exécutable

Pour créer un exécutable Windows :

```bash
# Utiliser le script Python
python build_exe.py
```

L'exécutable sera créé dans le dossier 'dist', avec les dossiers nécessaires pour l'application (data, chroma_db, documents).

## Déploiement avec Docker

Le projet peut être déployé dans un conteneur Docker pour faciliter le partage et la distribution.

### Prérequis

- Docker et Docker Compose installés sur votre machine
- Une clé API Gemini valide

### Construction et lancement du conteneur

1. Clonez le dépôt et placez-vous dans le répertoire du projet
2. Créez un fichier `.env` à la racine du projet avec votre clé API:
   ```
   GEMINI_API_KEY=votre_clé_api_gemini
   ```
3. Construisez et lancez le conteneur:
   ```bash
   docker-compose build
   docker-compose up rag-cli
   ```

### Utilisation en mode CLI

Pour exécuter différentes commandes dans le conteneur:

```bash
# Initialiser le système
docker-compose run rag-cli python main.py init --data-dir ./data --db-dir ./chroma_db --collection docss

# Traiter un répertoire de documents
docker-compose run rag-cli python main.py process --pdf-dir ./documents --recursive

# Exécuter une requête
docker-compose run rag-cli python main.py query --query "Votre question ici"
```

### Interface graphique avec Docker

L'interface graphique peut être utilisée depuis Docker, mais nécessite un accès au serveur X11:

#### Sous Linux:
```bash
xhost +local:docker
docker-compose up rag-gui
```

#### Sous Windows avec WSL2:
Vous devez avoir un serveur X configuré (VcXsrv, X410, etc.) et définir la variable DISPLAY correctement dans WSL.

### Volumes et données persistantes

Le docker-compose.yml configure les volumes suivants pour la persistance des données:
- `./documents`: Répertoire contenant vos PDF
- `./data`: Stockage des données et du cache
- `./chroma_db`: Base de données vectorielle

### Partage du code source

Pour partager le code source du projet:

1. Le dépôt inclut un fichier `.gitignore` qui exclut automatiquement:
   - Les dossiers de données et de cache
   - Les fichiers de configuration sensibles (.env)
   - Les fichiers temporaires et compilés
   
2. Avant de partager, assurez-vous de:
   - Ne pas inclure votre clé API Gemini personnelle
   - Supprimer les documents confidentiels du dossier `docs/`
   - Vérifier qu'aucune information sensible n'est présente dans les fichiers de configuration

3. Pour cloner et utiliser le projet partagé:
   ```bash
   git clone <url-du-repo>
   cd RAG
   pip install -r requirements.txt
   # Créer un fichier .env avec votre clé API
   echo "GEMINI_API_KEY=votre_clé_api" > .env
   # Lancer l'application
   python gui.py
   ```

## Structure du projet

```
RAG/
├── data/                  # Répertoire pour les données et le cache
│   ├── pdf_cache/         # Cache des extractions PDF
│   └── query_history.json # Historique des requêtes
├── src/                   # Code source
│   ├── extractors/        # Extraction de texte
│   │   └── pdf_extractor.py
│   ├── embeddings/        # Génération d'embeddings
│   │   └── sentence_embeddings.py
│   ├── database/          # Stockage vectoriel
│   │   └── chroma_db.py
│   ├── llm/               # Intégration du modèle de langage
│   │   └── gemini_api.py
│   └── pipeline/          # Pipeline RAG
│       └── rag_pipeline.py
├── docs/                  # Documents PDF source
├── chroma_db/             # Base de données vectorielle
├── docss/                 # Collection ChromaDB
├── tests/                 # Tests et optimisation
├── gui.py                 # Interface graphique
├── main.py                # Interface en ligne de commande
├── requirements.txt       # Dépendances
└── README.md              # Documentation
```

## Fonctionnalités principales

### Traitement des documents

- Support pour les documents PDF
- Extraction de texte avec suppression des en-têtes et pieds de page
- Support OCR pour les documents scannés
- Chunking intelligent avec chevauchement configurable
- Extraction des métadonnées (titre, auteur, date)
- Cache d'extraction pour améliorer les performances

### Génération d'embeddings

- Utilisation de modèles Sentence Transformers (par défaut: paraphrase-multilingual-MiniLM-L12-v2)
- Support multilingue
- Gestion des dimensions d'embeddings cohérentes
- Traitement par lots pour les grands volumes de documents

### Recherche et génération

- Recherche sémantique avec ChromaDB
- Génération de réponses structurées via l'API Gemini
- Citation automatique des sources
- Historique des requêtes avec possibilité de filtrage et réutilisation
- Affichage des scores de pertinence des sources

### Interface utilisateur

- Interface graphique simple avec tkinter
- Organisation en onglets (Recherche, Historique)
- Configuration des paramètres du système
- Visualisation des sources avec leurs scores
- Affichage détaillé des résultats des requêtes

## Modèles d'embeddings disponibles

Le système supporte plusieurs modèles d'embeddings :

- **all-MiniLM-L6-v2** : Plus petit et plus rapide (384 dimensions)
- **paraphrase-multilingual-MiniLM-L12-v2** : Bon compromis taille/performance, multilingue
- **distiluse-base-multilingual-cased-v1** : Multilingue
- **all-mpnet-base-v2** : Meilleure performance mais plus grand

## Limitations et améliorations possibles

1. **OCR** : La reconnaissance de texte dans les images pourrait être améliorée avec des modèles plus avancés.
2. **Interface web** : Une interface web pourrait être développée pour faciliter l'accès distant.
3. **Performances** : L'optimisation des paramètres de chunks et de recherche est cruciale pour de grands corpus.
4. **Support de formats** : L'ajout du support pour d'autres formats de documents (DOCX, HTML, etc.).
5. **Authentification** : Implémentation d'un système d'authentification pour un usage multi-utilisateur.

## Conclusion

Ce système RAG offre une solution complète pour l'extraction d'informations à partir de documents PDF et la génération de réponses basées sur ces informations. L'interface graphique le rend accessible aux utilisateurs non techniques, tandis que l'interface en ligne de commande permet une utilisation plus avancée et scriptable.
