"""
Module d'embeddings basé sur Sentence Transformers.

Ce module fournit des classes pour générer des embeddings de haute qualité 
à partir de textes en utilisant des modèles pré-entraînés de Sentence Transformers.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Union, Optional, Tuple, Any
import pickle
from tqdm import tqdm
import time
import warnings

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filtrer certains avertissements
warnings.filterwarnings("ignore", message=".*huggingface_hub.*cache-system uses symlinks.*")

# Variable globale pour le stockage du modèle (singleton)
_model_cache = {}

class SentenceEmbeddings:
    """Classe pour créer et manipuler des embeddings via Sentence Transformers."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",  # Modèle plus petit par défaut
                 device: str = None,
                 batch_size: int = 32,
                 show_progress_bar: bool = True,
                 normalize_embeddings: bool = True,
                 use_cache: bool = True):
        """
        Initialise le générateur d'embeddings avec un modèle Sentence Transformers.
        
        Args:
            model_name: Nom du modèle Sentence Transformers à utiliser
            device: Dispositif à utiliser (None pour auto-détection, 'cpu', 'cuda', 'cuda:0', etc.)
            batch_size: Taille des lots pour la génération d'embeddings
            show_progress_bar: Afficher une barre de progression lors du traitement par lots
            normalize_embeddings: Normaliser les embeddings (recommandé pour la similarité cosinus)
            use_cache: Utiliser le cache global des modèles pour éviter de recharger un modèle déjà chargé
        """
        global _model_cache
        
        try:
            # Importer ici pour éviter d'avoir une dépendance obligatoire
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("SentenceTransformers n'est pas installé. Installez-le avec: pip install sentence-transformers")
            raise ImportError("SentenceTransformers n'est pas installé. Installez-le avec: pip install sentence-transformers")
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        
        # Chargement du modèle (ou récupération depuis le cache)
        start_time = time.time()
        try:
            # Vérifier si le modèle est déjà dans le cache
            if use_cache and model_name in _model_cache:
                logger.info(f"Utilisation du modèle {model_name} depuis le cache")
                self.model = _model_cache[model_name]
            else:
                # Définir une variable d'environnement pour désactiver l'avertissement des symlinks
                os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
                
                logger.info(f"Chargement du modèle {model_name}...")
                self.model = SentenceTransformer(model_name, device=device)
                
                # Mettre le modèle dans le cache
                if use_cache:
                    _model_cache[model_name] = self.model
                    
            logger.info(f"Modèle {model_name} prêt en {time.time() - start_time:.2f} secondes")
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle {model_name}: {str(e)}")
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                raise ValueError(f"Problème de connexion lors du téléchargement du modèle. "
                                "Vérifiez votre connexion Internet ou utilisez un modèle déjà téléchargé.")
            else:
                raise
        
        self.is_fitted = True  # Les modèles pré-entraînés sont déjà entraînés
        self.document_embeddings = None
        self.document_ids = None
        
        # Affichage des informations sur le modèle
        logger.info(f"Dimensions des embeddings: {self.model.get_sentence_embedding_dimension()}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Prétraite un texte avant de générer son embedding.
        Cette méthode est fournie pour la compatibilité avec l'API existante.
        
        Args:
            text: Texte à prétraiter
            
        Returns:
            Texte prétraité
        """
        # Le prétraitement est géré par SentenceTransformers, mais on peut
        # ajouter ici des étapes supplémentaires si nécessaire
        return text

    def fit(self, documents: List[str], document_ids: Optional[List[Any]] = None) -> 'SentenceEmbeddings':
        """
        Génère des embeddings pour un corpus de documents et les stocke.
        Cette méthode est présente pour la compatibilité avec l'API TF-IDF.
        
        Args:
            documents: Liste de documents textuels
            document_ids: Identifiants optionnels pour les documents
            
        Returns:
            Self pour le chaînage de méthodes
        """
        if not documents:
            raise ValueError("La liste de documents ne peut pas être vide")
        
        # Génération des embeddings pour les documents
        self.document_embeddings = self.encode(documents)
        
        # Stockage des identifiants de documents
        if document_ids is None:
            self.document_ids = list(range(len(documents)))
        else:
            if len(document_ids) != len(documents):
                raise ValueError("Le nombre d'identifiants doit correspondre au nombre de documents")
            self.document_ids = document_ids
        
        logger.info(f"Embeddings générés pour {len(documents)} documents")
        
        return self
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Transforme des textes en embeddings.
        
        Args:
            texts: Liste de textes à transformer
            
        Returns:
            Tableau NumPy d'embeddings
        """
        start_time = time.time()
        
        # Utilisation du modèle pour encoder les textes
        embeddings = self.model.encode(
            texts, 
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"{len(texts)} textes encodés en {elapsed_time:.2f} secondes")
        
        return embeddings
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforme des documents en embeddings.
        Méthode de compatibilité avec l'API TF-IDF.
        
        Args:
            documents: Liste de documents à transformer
            
        Returns:
            Matrice d'embeddings
        """
        return self.encode(documents)
    
    def fit_transform(self, documents: List[str], document_ids: Optional[List[Any]] = None) -> np.ndarray:
        """
        Génère des embeddings et les stocke en une seule étape.
        
        Args:
            documents: Liste de documents textuels
            document_ids: Identifiants optionnels pour les documents
            
        Returns:
            Matrice d'embeddings
        """
        self.fit(documents, document_ids)
        return self.document_embeddings
    
    def get_document_embedding(self, document_id: Any) -> Optional[np.ndarray]:
        """
        Récupère l'embedding d'un document par son identifiant.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            Embedding du document ou None si non trouvé
        """
        if self.document_embeddings is None:
            raise ValueError("Aucun embedding de document n'a été généré")
        
        try:
            idx = self.document_ids.index(document_id)
            return self.document_embeddings[idx]
        except ValueError:
            logger.warning(f"Document avec l'identifiant {document_id} non trouvé")
            return None
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Recherche les documents les plus similaires à une requête.
        
        Args:
            query: Texte de la requête
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de tuples (document_id, score de similarité)
        """
        if self.document_embeddings is None:
            raise ValueError("Aucun embedding de document n'a été généré")
        
        # Génération de l'embedding pour la requête
        query_embedding = self.encode([query])[0]
        
        # Calcul des similarités avec tous les documents
        # Utilise directement la similarité cosinus puisque les vecteurs sont normalisés
        similarities = np.dot(self.document_embeddings, query_embedding)
        
        # Tri des résultats par similarité décroissante
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Création de la liste de résultats
        results = [(self.document_ids[idx], similarities[idx]) for idx in top_indices]
        
        logger.info(f"Recherche effectuée pour la requête: '{query}', {len(results)} résultats trouvés")
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde les embeddings et leurs identifiants.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        model_data = {
            'model_name': self.model_name,
            'document_embeddings': self.document_embeddings,
            'document_ids': self.document_ids,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Embeddings sauvegardés dans {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SentenceEmbeddings':
        """
        Charge des embeddings sauvegardés.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            Instance de SentenceEmbeddings chargée
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Création d'une nouvelle instance avec le même modèle
        instance = cls(model_name=model_data['model_name'])
        
        # Restauration des données
        instance.document_embeddings = model_data['document_embeddings']
        instance.document_ids = model_data['document_ids']
        instance.is_fitted = model_data['is_fitted']
        
        logger.info(f"Embeddings chargés depuis {filepath}")
        
        return instance
    
    def get_embedding_dimension(self) -> int:
        """
        Retourne la dimension des embeddings générés par le modèle.
        
        Returns:
            Dimension des embeddings
        """
        return self.model.get_sentence_embedding_dimension()


class SentenceEmbeddingFunction:
    """Adaptateur pour utiliser SentenceEmbeddings avec ChromaDB."""
    
    def __init__(self, model):
        """
        Initialise l'adaptateur avec un modèle SentenceEmbeddings ou un nom de modèle.
        
        Args:
            model: Instance de SentenceEmbeddings ou nom du modèle à utiliser
        """
        if isinstance(model, str):
            self.model = SentenceEmbeddings(model_name=model)
        else:
            self.model = model
    
    def __call__(self, input):
        """
        Transforme des textes en embeddings.
        Cette méthode est appelée automatiquement par ChromaDB.
        
        Args:
            input: Texte ou liste de textes à transformer en embeddings
            
        Returns:
            Liste de vecteurs d'embedding
        """
        if not isinstance(input, list):
            input = [input]
        
        # Transformer les textes en embeddings
        embeddings = self.model.encode(input)
        
        # Convertir en liste pour ChromaDB
        return embeddings.tolist()


# Exemple d'utilisation
if __name__ == "__main__":
    # Cet exemple montre comment utiliser la classe SentenceEmbeddings
    
    # Création d'une instance avec un modèle multilingue
    embeddings_generator = SentenceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")
    
    # Exemple de documents
    documents = [
        "Le système RAG (Retrieval-Augmented Generation) combine la recherche d'information et la génération de texte.",
        "Les embeddings neuronaux sont une méthode efficace pour représenter des documents textuels.",
        "PyMuPDF est une bibliothèque Python pour extraire du texte à partir de fichiers PDF.",
        "ChromaDB est une base de données vectorielle open-source pour stocker et rechercher des embeddings."
    ]
    
    # Génération d'embeddings
    document_embeddings = embeddings_generator.fit_transform(documents)
    print(f"Dimensions des embeddings: {document_embeddings.shape}")
    
    # Recherche
    query = "Comment fonctionne la recherche d'information?"
    results = embeddings_generator.search(query, top_k=2)
    print(f"\nRésultats pour la requête '{query}':")
    for doc_id, score in results:
        print(f"- Document {doc_id}: {documents[doc_id]} (score: {score:.4f})")
    
    # Test de l'adaptateur pour ChromaDB
    embedding_function = SentenceEmbeddingFunction(embeddings_generator)
    test_embeddings = embedding_function(["Ceci est un test de l'adaptateur ChromaDB"])
    print(f"\nDimension de l'embedding de test: {len(test_embeddings[0])}")
