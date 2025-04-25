"""
Module pour la gestion de la base de données vectorielle ChromaDB.

Ce module fournit des fonctions pour configurer ChromaDB en mode persistant,
ajouter des documents et effectuer des recherches par similarité.
"""

import os
import logging
import time
import functools
import concurrent.futures
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid
from tqdm import tqdm
import json

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache simple pour les requêtes fréquentes
class _SimpleCache:
    """Implémentation d'un cache simple pour stocker les résultats de requêtes."""
    
    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            self.access_times[key] = time.time()
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any) -> None:
        if len(self.cache) >= self.max_size and key not in self.cache:
            # Supprime l'entrée la moins récemment utilisée
            oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        self.cache.clear()
        self.access_times.clear()

def _timeit(func):
    """Décorateur pour mesurer le temps d'exécution d'une fonction."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.debug(f"{func.__name__} exécuté en {execution_time:.4f} secondes")
        return result
    return wrapper

class ChromaDBManager:
    """Classe pour gérer la base de données vectorielle ChromaDB."""
    
    def __init__(self, 
                 persist_directory: str,
                 collection_name: str = "docss",
                 embedding_function: Optional[Any] = None,
                 create_if_not_exists: bool = True):
        """
        Initialise le gestionnaire ChromaDB.
        
        Args:
            persist_directory: Répertoire pour la persistance des données
            collection_name: Nom de la collection de documents
            embedding_function: Fonction d'embedding à utiliser (None pour utiliser celui par défaut)
            create_if_not_exists: Si True, crée la collection si elle n'existe pas
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self._cache = _SimpleCache(max_size=100)  # Cache interne
        
        # Création du répertoire de persistance s'il n'existe pas
        if not os.path.exists(persist_directory):
            os.makedirs(persist_directory)
            logger.info(f"Répertoire de persistance créé: {persist_directory}")
        
        # Initialisation du client ChromaDB avec la nouvelle API
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Récupération ou création de la collection
        if create_if_not_exists:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=embedding_function,
                metadata={"description": "Collection pour le système RAG"}
            )
            logger.info(f"Collection '{collection_name}' récupérée ou créée")
        else:
            try:
                self.collection = self.client.get_collection(
                    name=collection_name,
                    embedding_function=embedding_function
                )
                logger.info(f"Collection existante '{collection_name}' récupérée")
            except Exception as e:
                logger.error(f"Erreur lors de la récupération de la collection: {str(e)}")
                raise ValueError(f"La collection '{collection_name}' n'existe pas et create_if_not_exists est False")
    
    def _generate_cache_key(self, query: str, n_results: int, where: Optional[Dict[str, Any]] = None) -> str:
        """Génère une clé de cache basée sur les paramètres de recherche."""
        # Création d'une clé unique basée sur les paramètres de recherche
        key_parts = [query, str(n_results)]
        if where:
            # Conversion du dictionnaire where en une représentation string stable
            # au lieu de trier directement les items qui peut causer une erreur
            # si des valeurs sont des dictionnaires imbriqués
            # Transformation en JSON trié pour assurer la cohérence des clés
            where_str = json.dumps(where, sort_keys=True)
            key_parts.append(where_str)
        return str(hash(tuple(key_parts)))
    
    def _process_batch(self, batch_data):
        """Traite un lot de documents pour l'ajout à la collection."""
        texts, ids, metadatas, embeddings, start_idx, batch_size = batch_data
        
        end_idx = min(start_idx + batch_size, len(texts))
        batch_texts = texts[start_idx:end_idx]
        batch_ids = ids[start_idx:end_idx]
        
        batch_metadatas = None
        if metadatas is not None:
            batch_metadatas = metadatas[start_idx:end_idx]
        
        batch_embeddings = None
        if embeddings is not None:
            batch_embeddings = embeddings[start_idx:end_idx]
        
        success_ids = []
        try:
            # Ajout du lot à la collection
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
            success_ids = batch_ids
            return success_ids, 0  # 0 échecs
        except Exception as e:
            logger.error(f"Erreur lors du traitement du lot {start_idx//batch_size + 1}: {str(e)}")
            return success_ids, len(batch_texts)  # Tous les documents du lot ont échoué

    def add_documents(self, 
                     texts: List[str], 
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None,
                     embeddings: Optional[List[List[float]]] = None,
                     batch_size: int = 50) -> List[str]:
        """
        Ajoute des documents à la collection ChromaDB.
        
        Args:
            texts: Liste des textes des documents
            metadatas: Liste des métadonnées associées aux documents
            ids: Liste des identifiants des documents (générés automatiquement si None)
            embeddings: Liste des embeddings pré-calculés (calculés par ChromaDB si None)
            batch_size: Taille des lots pour l'ajout par lots
            
        Returns:
            Liste des identifiants des documents ajoutés
        """
        if not texts:
            logger.warning("Aucun document à ajouter")
            return []
        
        start_time = time.time()
        
        # Génération des identifiants si non fournis
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        # Vérification que les listes ont la même longueur
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError(f"Les listes texts ({len(texts)}) et metadatas ({len(metadatas)}) doivent avoir la même longueur")
        
        if embeddings is not None:
            if len(embeddings) != len(texts):
                raise ValueError(f"Les listes texts ({len(texts)}) et embeddings ({len(embeddings)}) doivent avoir la même longueur")
            
            # Vérification de la cohérence des dimensions des embeddings
            embedding_dims = [len(emb) for emb in embeddings]
            if len(set(embedding_dims)) > 1:
                logger.warning(f"Dimensions incohérentes dans les embeddings: min={min(embedding_dims)}, max={max(embedding_dims)}")
                logger.warning("Tentative de correction des dimensions des embeddings...")
                
                # Trouver la dimension la plus courante
                from collections import Counter
                dim_counter = Counter(embedding_dims)
                most_common_dim = dim_counter.most_common(1)[0][0]
                
                # Filtrer et corriger les embeddings
                filtered_embeddings = []
                filtered_texts = []
                filtered_ids = []
                filtered_metadatas = []
                
                for i, (text, emb_id, emb) in enumerate(zip(texts, ids, embeddings)):
                    if len(emb) == most_common_dim:
                        filtered_embeddings.append(emb)
                        filtered_texts.append(text)
                        filtered_ids.append(emb_id)
                        if metadatas:
                            filtered_metadatas.append(metadatas[i])
                    else:
                        logger.warning(f"Exclusion du document {emb_id}: dimension de l'embedding ({len(emb)}) différente de la plus courante ({most_common_dim})")
                
                # Remplacer par les listes filtrées
                texts = filtered_texts
                ids = filtered_ids
                embeddings = filtered_embeddings
                metadatas = filtered_metadatas if metadatas else None
                
                logger.info(f"Après correction: {len(texts)} documents sur {len(embedding_dims)} conservés")
        
        # Si après filtrage il n'y a plus de documents, retourner une liste vide
        if not texts:
            logger.warning("Après filtrage des embeddings, aucun document à ajouter")
            return []
        
        # Utilisation d'une taille de batch plus petite pour éviter les problèmes de mémoire
        batch_size = min(batch_size, 50)  # Limiter la taille des lots à 50 maximum
        
        # Détermine si le traitement parallèle est avantageux (plus de 1000 documents ou lots > 10)
        use_parallel = len(texts) > 1000 or (len(texts) // batch_size) > 10
        
        # Ajout par lots
        success_ids = []
        failed_count = 0
        
        try:
            if use_parallel:
                # Préparation des tâches pour un traitement parallèle
                batch_tasks = []
                for i in range(0, len(texts), batch_size):
                    batch_tasks.append((texts, ids, metadatas, embeddings, i, batch_size))
                    
                # Traitement parallèle des lots
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(batch_tasks))) as executor:
                    # Utiliser tqdm pour afficher la progression
                    results = list(tqdm(
                        executor.map(self._process_batch, batch_tasks),
                        total=len(batch_tasks),
                        desc="Ajout des documents à ChromaDB"
                    ))
                    
                    # Récupérer les résultats
                    for batch_result in results:
                        if isinstance(batch_result, tuple):
                            batch_success_ids, batch_failed = batch_result
                            success_ids.extend(batch_success_ids)
                            failed_count += batch_failed
            else:
                # Ajout séquentiel par lots pour les petits ensembles de données
                for i in tqdm(range(0, len(texts), batch_size), desc="Ajout des documents à ChromaDB"):
                    end_idx = min(i + batch_size, len(texts))
                    batch_texts = texts[i:end_idx]
                    batch_ids = ids[i:end_idx]
                    
                    batch_metadatas = None
                    if metadatas is not None:
                        batch_metadatas = metadatas[i:end_idx]
                    
                    batch_embeddings = None
                    if embeddings is not None:
                        batch_embeddings = embeddings[i:end_idx]
                    
                    try:
                        # Ajout du lot à la collection
                        self.collection.add(
                            documents=batch_texts,
                            metadatas=batch_metadatas,
                            ids=batch_ids,
                            embeddings=batch_embeddings
                        )
                        success_ids.extend(batch_ids)
                    except Exception as e:
                        logger.error(f"Erreur lors de l'ajout du lot {i//batch_size + 1}/{len(texts)//batch_size + 1}: {str(e)}")
                        failed_count += len(batch_texts)
                        # Continuer avec le lot suivant malgré l'erreur
            
            # Invalidation du cache, car les données ont changé
            self._cache.clear()
            
            elapsed_time = time.time() - start_time
            logger.info(f"{len(success_ids)} documents ajoutés avec succès à la collection '{self.collection_name}' en {elapsed_time:.2f} secondes")
            if failed_count > 0:
                logger.warning(f"{failed_count} documents n'ont pas pu être ajoutés")
            
            return success_ids
        
        except Exception as e:
            logger.error(f"Erreur grave lors de l'ajout des documents: {str(e)}")
            return success_ids  # Retourner les IDs des documents qui ont été ajoutés avec succès avant l'erreur
    
    def search(self, 
            query_texts: Union[str, List[str]] = None, 
            query_embeddings: List[List[float]] = None,
            n_results: int = 5,
            where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recherche dans la collection ChromaDB.
        
        Args:
            query_texts: Textes de requête (si embedding_function est fourni)
            query_embeddings: Embeddings de requête pré-calculés
            n_results: Nombre de résultats à retourner
            where: Filtres sur les métadonnées
        
        Returns:
            Résultats de la recherche
        """
        start_time = time.time()
        
        # Gestion du cas où query_texts est une chaîne simple
        if isinstance(query_texts, str):
            # Vérification du cache pour les requêtes textuelles simples
            cache_key = self._generate_cache_key(query_texts, n_results, where)
            cached_result = self._cache.get(cache_key)
            if cached_result:
                logger.info("Résultats récupérés depuis le cache")
                return cached_result
                
            query_texts = [query_texts]
        
        include_params = ["documents", "metadatas", "distances", "embeddings"]
            
        if query_embeddings is not None:
            # Si des embeddings sont fournis, utilisez-les directement
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=include_params
            )
        elif query_texts is not None:
            # Sinon, utilisez les textes de requête (ChromaDB utilise embedding_function)
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                include=include_params
            )
        else:
            raise ValueError("Vous devez fournir soit query_texts, soit query_embeddings")
        
        # Mise en cache des résultats pour les requêtes textuelles simples
        if isinstance(query_texts, list) and len(query_texts) == 1:
            cache_key = self._generate_cache_key(query_texts[0], n_results, where)
            self._cache.set(cache_key, results)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Recherche effectuée en {elapsed_time:.4f} secondes")
        
        return results
    
    def search_by_embedding(self, 
                           embedding: List[float], 
                           n_results: int = 5,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Recherche les documents les plus similaires à un embedding.
        
        Args:
            embedding: Embedding vectoriel de la requête
            n_results: Nombre de résultats à retourner
            where: Filtre sur les métadonnées
            
        Returns:
            Résultats de la recherche avec les documents, scores et métadonnées
        """
        start_time = time.time()
        
        include_params = ["documents", "metadatas", "distances", "embeddings"]
        
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where,
            include=include_params
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Recherche par embedding effectuée en {elapsed_time:.4f} secondes, {len(results['ids'][0])} résultats trouvés")
        
        return results
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Récupère un document par son identifiant.
        
        Args:
            document_id: Identifiant du document
            
        Returns:
            Document avec ses métadonnées et son embedding
            
        Raises:
            ValueError: Si le document n'est pas trouvé
        """
        try:
            result = self.collection.get(ids=[document_id], include=["documents", "metadatas", "embeddings"])
            if not result['ids']:
                raise ValueError(f"Document avec l'identifiant {document_id} non trouvé")
            
            return {
                'id': result['ids'][0],
                'document': result['documents'][0],
                'metadata': result['metadatas'][0] if result['metadatas'] else None,
                'embedding': result['embeddings'][0] if 'embeddings' in result and result['embeddings'] else None
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération du document: {str(e)}")
            raise ValueError(f"Erreur lors de la récupération du document: {str(e)}")
    
    def update_document(self, 
                       document_id: str, 
                       text: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       embedding: Optional[List[float]] = None) -> None:
        """
        Met à jour un document existant.
        
        Args:
            document_id: Identifiant du document à mettre à jour
            text: Nouveau texte du document (None pour ne pas modifier)
            metadata: Nouvelles métadonnées (None pour ne pas modifier)
            embedding: Nouvel embedding (None pour ne pas modifier)
            
        Raises:
            ValueError: Si le document n'est pas trouvé
        """
        try:
            # Vérification que le document existe
            self.get_document(document_id)
            
            # Mise à jour du document
            self.collection.update(
                ids=[document_id],
                documents=[text] if text is not None else None,
                metadatas=[metadata] if metadata is not None else None,
                embeddings=[embedding] if embedding is not None else None
            )
            
            # Invalidation du cache car les données ont changé
            self._cache.clear()
            
            logger.info(f"Document {document_id} mis à jour")
        except ValueError as e:
            logger.error(f"Erreur lors de la mise à jour du document: {str(e)}")
            raise
    
    def delete_document(self, document_id: str) -> None:
        """
        Supprime un document de la collection.
        
        Args:
            document_id: Identifiant du document à supprimer
        """
        try:
            self.collection.delete(ids=[document_id])
            
            # Invalidation du cache car les données ont changé
            self._cache.clear()
            
            logger.info(f"Document {document_id} supprimé")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression du document: {str(e)}")
            raise ValueError(f"Erreur lors de la suppression du document: {str(e)}")
    
    def delete_collection(self) -> None:
        """
        Supprime la collection entière.
        """
        try:
            self.client.delete_collection(self.collection_name)
            
            # Invalidation du cache car les données ont changé
            self._cache.clear()
            
            logger.info(f"Collection '{self.collection_name}' supprimée")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la collection: {str(e)}")
            raise ValueError(f"Erreur lors de la suppression de la collection: {str(e)}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Récupère des statistiques sur la collection.
        
        Returns:
            Dictionnaire contenant des statistiques sur la collection
        """
        try:
            count = self.collection.count()
            return {
                'name': self.collection_name,
                'document_count': count,
                'persist_directory': self.persist_directory
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques: {str(e)}")
            raise ValueError(f"Erreur lors de la récupération des statistiques: {str(e)}")
    
    def create_tfidf_embedding_function(self):
        """
        Crée une fonction d'embedding TF-IDF personnalisée pour ChromaDB.
        
        Cette méthode est utile si vous souhaitez utiliser vos propres embeddings TF-IDF
        au lieu de ceux fournis par défaut par ChromaDB.
        
        Returns:
            Fonction d'embedding compatible avec ChromaDB
        """
        # Import ici pour éviter les dépendances circulaires
        from src.embeddings.tfidf_embeddings import TFIDFEmbeddings, TextPreprocessor
        
        # Création d'une instance de TFIDFEmbeddings
        preprocessor = TextPreprocessor(language='french')
        tfidf = TFIDFEmbeddings(preprocessor=preprocessor)
        
        # Fonction d'embedding qui utilise TFIDFEmbeddings
        def tfidf_embedding_function(texts: List[str]) -> List[List[float]]:
            if not tfidf.is_fitted:
                # Entraînement initial si nécessaire
                tfidf.fit(texts)
                return tfidf.document_embeddings.toarray().tolist()
            else:
                # Transformation des textes en embeddings
                embeddings = tfidf.transform(texts)
                return embeddings.toarray().tolist()
        
        return tfidf_embedding_function
    
    def create_sentence_transformer_embedding_function(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Crée une fonction d'embedding basée sur Sentence Transformers pour ChromaDB.
        
        Args:
            model_name: Nom du modèle Sentence Transformers à utiliser
            
        Returns:
            Fonction d'embedding compatible avec ChromaDB
        """
        # Import ici pour éviter les dépendances circulaires
        try:
            from src.embeddings.sentence_embeddings import SentenceEmbeddings
            
            # Création d'une instance de SentenceEmbeddings
            sentence_embeddings = SentenceEmbeddings(model_name=model_name)
            
            # Fonction d'embedding qui utilise SentenceEmbeddings
            def sentence_transformer_function(texts: List[str]) -> List[List[float]]:
                embeddings = sentence_embeddings.encode(texts)
                return embeddings.tolist()
            
            return sentence_transformer_function
        except ImportError:
            logger.error("Le module sentence_embeddings n'est pas disponible")
            try:
                # Essayer d'utiliser directement sentence_transformers si disponible
                from sentence_transformers import SentenceTransformer
                
                model = SentenceTransformer(model_name)
                
                def direct_sentence_transformer_function(texts: List[str]) -> List[List[float]]:
                    embeddings = model.encode(texts, convert_to_tensor=False)
                    return embeddings.tolist()
                
                return direct_sentence_transformer_function
            except ImportError:
                logger.error("SentenceTransformers n'est pas installé. Utilisez 'pip install sentence-transformers'")
                raise ImportError("SentenceTransformers n'est pas installé. Utilisez 'pip install sentence-transformers'")
    
    def create_embedding_function(self, embedding_type: str = "sentence_transformers", model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Crée une fonction d'embedding selon le type spécifié.
        
        Args:
            embedding_type: Type d'embedding ("tfidf" ou "sentence_transformers")
            model_name: Nom du modèle pour sentence_transformers
            
        Returns:
            Fonction d'embedding compatible avec ChromaDB
        """
        if embedding_type.lower() == "tfidf":
            return self.create_tfidf_embedding_function()
        elif embedding_type.lower() == "sentence_transformers":
            return self.create_sentence_transformer_embedding_function(model_name)
        else:
            raise ValueError(f"Type d'embedding non supporté: {embedding_type}")
    
    def get_collection(self):
        """
        Récupère la collection ChromaDB.
        
        Returns:
            Collection ChromaDB
        """
        return self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )    


# Exemple d'utilisation
if __name__ == "__main__":
    # Cet exemple montre comment utiliser la classe ChromaDBManager
    
    # Création d'une instance de ChromaDBManager
    db_manager = ChromaDBManager(
        persist_directory="./chroma_db",
        collection_name="docss"
    )
    
    # Exemple de documents
    documents = [
        "Le système RAG (Retrieval-Augmented Generation) combine la recherche d'information et la génération de texte.",
        "Les embeddings TF-IDF sont une méthode simple mais efficace pour représenter des documents textuels.",
        "PyMuPDF est une bibliothèque Python pour extraire du texte à partir de fichiers PDF.",
        "ChromaDB est une base de données vectorielle open-source pour stocker et rechercher des embeddings."
    ]
    
    # Métadonnées associées aux documents
    metadatas = [
        {"source": "article_rag", "author": "John Doe", "date": "2023-01-15"},
        {"source": "article_embeddings", "author": "Jane Smith", "date": "2023-02-20"},
        {"source": "documentation_pymupdf", "author": "PyMuPDF Team", "date": "2023-03-10"},
        {"source": "documentation_chromadb", "author": "ChromaDB Team", "date": "2023-04-05"}
    ]
    
    # Ajout des documents à la collection
    doc_ids = db_manager.add_documents(documents, metadatas=metadatas)
    print(f"Documents ajoutés avec les IDs: {doc_ids}")
    
    # Recherche de documents similaires
    query = "Comment fonctionne la recherche d'information?"
    results = db_manager.search(query_texts=query, n_results=2)
    
    print(f"\nRésultats pour la requête '{query}':")
    for i, (doc_id, doc, metadata, distance) in enumerate(zip(
        results['ids'][0], 
        results['documents'][0], 
        results['metadatas'][0], 
        results['distances'][0]
    )):
        print(f"{i+1}. Document ID: {doc_id}")
        print(f"   Texte: {doc}")
        print(f"   Métadonnées: {metadata}")
        print(f"   Score de similarité: {1 - distance:.4f}")  # Conversion de distance en similarité
        print()
    
    # Statistiques sur la collection
    stats = db_manager.get_collection_stats()
    print(f"Statistiques de la collection: {stats}")
