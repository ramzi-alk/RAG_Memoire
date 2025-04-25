"""
Module de prétraitement de texte et de création d'embeddings TF-IDF.

Ce module fournit des fonctions pour nettoyer et prétraiter le texte,
ainsi que pour créer des embeddings vectoriels en utilisant TF-IDF.
"""

import re
import string
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
from tqdm import tqdm

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Classe pour le prétraitement du texte avant la vectorisation."""
    
    def __init__(self, 
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 remove_stopwords: bool = True,
                 language: str = 'french'):
        """
        Initialise le préprocesseur de texte avec des options de configuration.
        
        Args:
            remove_punctuation: Si True, supprime la ponctuation
            lowercase: Si True, convertit le texte en minuscules
            remove_numbers: Si True, supprime les chiffres
            remove_stopwords: Si True, supprime les mots vides
            language: Langue pour les mots vides ('french', 'english', etc.)
        """
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.remove_stopwords = remove_stopwords
        self.language = language
        
        # Chargement des mots vides si nécessaire
        self.stopwords = set()
        if remove_stopwords:
            try:
                from nltk.corpus import stopwords
                import nltk
                try:
                    self.stopwords = set(stopwords.words(language))
                except LookupError:
                    nltk.download('stopwords')
                    self.stopwords = set(stopwords.words(language))
            except ImportError:
                logger.warning("NLTK n'est pas installé. Les mots vides ne seront pas supprimés.")
                logger.warning("Installez NLTK avec: pip install nltk")
        
        logger.info(f"Préprocesseur de texte initialisé avec: remove_punctuation={remove_punctuation}, "
                   f"lowercase={lowercase}, remove_numbers={remove_numbers}, "
                   f"remove_stopwords={remove_stopwords}, language={language}")
    
    def preprocess_text(self, text: str) -> str:
        """
        Prétraite un texte selon les options configurées.
        
        Args:
            text: Texte à prétraiter
            
        Returns:
            Texte prétraité
        """
        if not text:
            return ""
        
        # Conversion en minuscules
        if self.lowercase:
            text = text.lower()
        
        # Suppression des nombres
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Suppression de la ponctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Suppression des mots vides
        if self.remove_stopwords and self.stopwords:
            words = text.split()
            words = [word for word in words if word not in self.stopwords]
            text = ' '.join(words)
        
        # Suppression des espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_documents(self, documents: List[str]) -> List[str]:
        """
        Prétraite une liste de documents.
        
        Args:
            documents: Liste de textes à prétraiter
            
        Returns:
            Liste de textes prétraités
        """
        preprocessed_docs = []
        for doc in tqdm(documents, desc="Prétraitement des documents"):
            preprocessed_docs.append(self.preprocess_text(doc))
        
        logger.info(f"{len(documents)} documents prétraités")
        return preprocessed_docs


class TFIDFEmbeddings:
    """Classe pour créer et manipuler des embeddings TF-IDF."""
    
    def __init__(self, 
                 preprocessor: Optional[TextPreprocessor] = None,
                 max_features: Optional[int] = None,
                 ngram_range: Tuple[int, int] = (1, 1),
                 min_df: Union[int, float] = 1,
                 max_df: Union[int, float] = 1.0):
        """
        Initialise le générateur d'embeddings TF-IDF.
        
        Args:
            preprocessor: Instance de TextPreprocessor pour le prétraitement
            max_features: Nombre maximum de features à conserver
            ngram_range: Plage de n-grammes à extraire (par défaut: unigrammes)
            min_df: Fréquence minimale des documents pour inclure un terme
            max_df: Fréquence maximale des documents pour inclure un terme
        """
        self.preprocessor = preprocessor or TextPreprocessor()
        
        # Initialisation du vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        
        self.is_fitted = False
        self.document_embeddings = None
        self.document_ids = None
        
        logger.info(f"TFIDFEmbeddings initialisé avec max_features={max_features}, "
                   f"ngram_range={ngram_range}, min_df={min_df}, max_df={max_df}")
    
    def fit(self, documents: List[str], document_ids: Optional[List[Any]] = None) -> 'TFIDFEmbeddings':
        """
        Entraîne le vectoriseur TF-IDF sur un corpus de documents.
        
        Args:
            documents: Liste de documents textuels
            document_ids: Identifiants optionnels pour les documents
            
        Returns:
            Self pour le chaînage de méthodes
        """
        if not documents:
            raise ValueError("La liste de documents ne peut pas être vide")
        
        # Prétraitement des documents
        preprocessed_docs = self.preprocessor.preprocess_documents(documents)
        
        # Entraînement du vectoriseur
        self.vectorizer.fit(preprocessed_docs)
        
        # Création des embeddings pour les documents d'entraînement
        self.document_embeddings = self.vectorizer.transform(preprocessed_docs)
        
        # Stockage des identifiants de documents
        if document_ids is None:
            self.document_ids = list(range(len(documents)))
        else:
            if len(document_ids) != len(documents):
                raise ValueError("Le nombre d'identifiants doit correspondre au nombre de documents")
            self.document_ids = document_ids
        
        self.is_fitted = True
        
        logger.info(f"Vectoriseur TF-IDF entraîné sur {len(documents)} documents, "
                   f"vocabulaire de taille {len(self.vectorizer.get_feature_names_out())}")
        
        return self
    
    def transform(self, documents: List[str]) -> np.ndarray:
        """
        Transforme des documents en embeddings TF-IDF.
        
        Args:
            documents: Liste de documents à transformer
            
        Returns:
            Matrice d'embeddings TF-IDF
            
        Raises:
            ValueError: Si le vectoriseur n'a pas été entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le vectoriseur doit être entraîné avant de transformer des documents")
        
        # Prétraitement des documents
        preprocessed_docs = self.preprocessor.preprocess_documents(documents)
        
        # Transformation en embeddings
        embeddings = self.vectorizer.transform(preprocessed_docs)
        
        logger.info(f"{len(documents)} documents transformés en embeddings TF-IDF")
        
        return embeddings
    
    def fit_transform(self, documents: List[str], document_ids: Optional[List[Any]] = None) -> np.ndarray:
        """
        Entraîne le vectoriseur et transforme les documents en une seule étape.
        
        Args:
            documents: Liste de documents textuels
            document_ids: Identifiants optionnels pour les documents
            
        Returns:
            Matrice d'embeddings TF-IDF
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
        if not self.is_fitted:
            raise ValueError("Le vectoriseur doit être entraîné avant de récupérer des embeddings")
        
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
            
        Raises:
            ValueError: Si le vectoriseur n'a pas été entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le vectoriseur doit être entraîné avant de faire une recherche")
        
        # Prétraitement de la requête
        preprocessed_query = self.preprocessor.preprocess_text(query)
        
        # Transformation de la requête en embedding
        query_embedding = self.vectorizer.transform([preprocessed_query])
        
        # Calcul des similarités avec tous les documents
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()
        
        # Tri des résultats par similarité décroissante
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Création de la liste de résultats
        results = [(self.document_ids[idx], similarities[idx]) for idx in top_indices]
        
        logger.info(f"Recherche effectuée pour la requête: '{query}', {len(results)} résultats trouvés")
        
        return results
    
    def save(self, filepath: str) -> None:
        """
        Sauvegarde le modèle TF-IDF et ses embeddings.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Raises:
            ValueError: Si le vectoriseur n'a pas été entraîné
        """
        if not self.is_fitted:
            raise ValueError("Le vectoriseur doit être entraîné avant d'être sauvegardé")
        
        model_data = {
            'vectorizer': self.vectorizer,
            'document_embeddings': self.document_embeddings,
            'document_ids': self.document_ids,
            'is_fitted': self.is_fitted,
            'preprocessor': self.preprocessor
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Modèle TF-IDF sauvegardé dans {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TFIDFEmbeddings':
        """
        Charge un modèle TF-IDF sauvegardé.
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            
        Returns:
            Instance de TFIDFEmbeddings chargée
            
        Raises:
            FileNotFoundError: Si le fichier n'existe pas
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Le fichier {filepath} n'existe pas")
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls()
        instance.vectorizer = model_data['vectorizer']
        instance.document_embeddings = model_data['document_embeddings']
        instance.document_ids = model_data['document_ids']
        instance.is_fitted = model_data['is_fitted']
        instance.preprocessor = model_data['preprocessor']
        
        logger.info(f"Modèle TF-IDF chargé depuis {filepath}")
        
        return instance


class TFIDFEmbeddingFunction:
    """Adaptateur pour utiliser TFIDFEmbeddings avec ChromaDB."""
    
    def __init__(self, tfidf_model):
        """
        Initialise l'adaptateur avec un modèle TFIDFEmbeddings.
        
        Args:
            tfidf_model: Instance de TFIDFEmbeddings pré-entraînée
        """
        self.model = tfidf_model
    
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
        
        # S'assurer que le modèle est entraîné
        if not self.model.is_fitted:
            raise ValueError("Le modèle TF-IDF doit être entraîné avant d'être utilisé pour les embeddings")
        
        # Transformer les textes en embeddings
        embeddings = self.model.transform(input)
        
        # Convertir en liste pour ChromaDB
        return embeddings.toarray().tolist()



# Exemple d'utilisation
if __name__ == "__main__":
    # Cet exemple montre comment utiliser les classes TextPreprocessor et TFIDFEmbeddings
    
    # Exemple de documents
    documents = [
        "Le système RAG (Retrieval-Augmented Generation) combine la recherche d'information et la génération de texte.",
        "Les embeddings TF-IDF sont une méthode simple mais efficace pour représenter des documents textuels.",
        "PyMuPDF est une bibliothèque Python pour extraire du texte à partir de fichiers PDF.",
        "ChromaDB est une base de données vectorielle open-source pour stocker et rechercher des embeddings."
    ]
    
    # Prétraitement
    preprocessor = TextPreprocessor(language='french')
    preprocessed_docs = preprocessor.preprocess_documents(documents)
    print("Documents prétraités:")
    for doc in preprocessed_docs:
        print(f"- {doc}")
    
    # Création d'embeddings TF-IDF
    tfidf = TFIDFEmbeddings(preprocessor=preprocessor)
    embeddings = tfidf.fit_transform(documents)
    print(f"\nDimensions des embeddings: {embeddings.shape}")
    
    # Recherche
    query = "Comment fonctionne la recherche d'information?"
    results = tfidf.search(query, top_k=2)
    print(f"\nRésultats pour la requête '{query}':")
    for doc_id, score in results:
        print(f"- Document {doc_id}: {documents[doc_id]} (score: {score:.4f})")
