"""
Module principal du pipeline RAG combinant tous les composants.

Ce module intègre l'extraction de texte, les embeddings avec Sentence Transformers,
le stockage vectoriel ChromaDB et l'API Gemini pour former un
système RAG complet.
"""

import os
import logging
import json
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
import time
import uuid
import re

# Import des composants du système RAG
from src.extractors.pdf_extractor import PDFExtractor
from src.embeddings.sentence_embeddings import SentenceEmbeddings, SentenceEmbeddingFunction
from src.database.chroma_db import ChromaDBManager
from src.llm.gemini_api import GeminiAPI

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """Classe principale pour le pipeline RAG."""
    
    def __init__(self, 
                data_dir: str = "./data",
                db_dir: str = "./chroma_db",
                collection_name: str = "docss",
                gemini_api_key: Optional[str] = None,
                gemini_model: str = "gemini-2.0-flash-001",
                remove_headers_footers: bool = True,
                min_line_length: int = 10,
                language: str = "french",
                embedding_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                reset_collection: bool = False,
                enable_cache: bool = True,
                enable_ocr: bool = False,
                max_workers: int = None,
                enable_query_history: bool = True,
                history_file: str = None,
                max_history_entries: int = 100):
        """
        Initialise le pipeline RAG avec tous ses composants.
        
        Args:
            data_dir: Répertoire pour les données et fichiers temporaires
            db_dir: Répertoire pour la base de données ChromaDB
            collection_name: Nom de la collection ChromaDB
            gemini_api_key: Clé API Gemini (si None, cherche dans les variables d'environnement)
            gemini_model: Modèle Gemini à utiliser
            remove_headers_footers: Si True, tente de supprimer les en-têtes et pieds de page des PDF
            min_line_length: Longueur minimale des lignes à conserver dans les PDF
            language: Langue pour le prétraitement du texte
            embedding_model_name: Nom du modèle Sentence Transformers à utiliser
            reset_collection: Si True, réinitialise la collection existante (utile lors du changement de modèle d'embedding)
                             (Note: cette fonctionnalité est désactivée pour éviter la perte de données)
            enable_cache: Si True, active la mise en cache des extractions PDF
            enable_ocr: Si True, active la détection automatique et l'OCR pour les PDF scannés
            max_workers: Nombre maximum de workers pour le traitement parallèle
            enable_query_history: Si True, active l'historique des requêtes
            history_file: Chemin vers le fichier de sauvegarde de l'historique (par défaut: data_dir/query_history.json)
            max_history_entries: Nombre maximum d'entrées d'historique à conserver
        """
        # Création des répertoires si nécessaire
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(db_dir, exist_ok=True)
        
        # Répertoire de cache pour les extractions PDF
        cache_dir = os.path.join(data_dir, "pdf_cache") if enable_cache else None
        
        # Initialisation des composants
        self.pdf_extractor = PDFExtractor(
            remove_headers_footers=remove_headers_footers,
            min_line_length=min_line_length,
            cache_dir=cache_dir
        )
        
        # Initialisation du modèle d'embeddings Sentence Transformers
        logger.info(f"Initialisation du modèle d'embeddings avec {embedding_model_name}")
        self.embedding_model = SentenceEmbeddings(
            model_name=embedding_model_name,
            normalize_embeddings=True
        )
        
        # Création de la fonction d'embedding pour ChromaDB
        embedding_function = SentenceEmbeddingFunction(self.embedding_model)

        # Ignorer la demande de réinitialisation pour éviter la perte de données
        if reset_collection:
            logger.warning("L'option reset_collection est désactivée pour éviter la perte accidentelle de données.")
            logger.info("Les documents existants dans la collection seront conservés.")
            # Ne pas appeler self._reset_collection()

        self.db_manager = ChromaDBManager(
            persist_directory=db_dir,
            collection_name=collection_name,
            embedding_function=embedding_function,
            create_if_not_exists=True
        )

        self.gemini_api = GeminiAPI(
            api_key=gemini_api_key,
            model_name=gemini_model,
            temperature=0.2,
            max_output_tokens=5048
        )

        # Initialisation de l'historique des requêtes
        self.enable_query_history = enable_query_history
        if enable_query_history:
            if history_file is None:
                history_file = os.path.join(data_dir, "query_history.json")
            self.query_history = QueryHistory(
                history_file=history_file,
                max_entries=max_history_entries
            )
            logger.info(f"Historique des requêtes activé, sauvegardé dans {history_file}")
        else:
            self.query_history = None
            logger.info("Historique des requêtes désactivé")

        # Stockage des chemins et paramètres
        self.data_dir = data_dir
        self.db_dir = db_dir
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.enable_ocr = enable_ocr
        self.ocr_language = language[:3]  # Conversion du code de langue pour Tesseract
        self.max_workers = max_workers
        
        logger.info("Pipeline RAG initialisé avec tous ses composants")

    
    def process_pdf_directory(self, 
                             pdf_dir: str, 
                             recursive: bool = False,
                             chunk_size: int = 1000,
                             chunk_overlap: int = 200,
                             file_extension: str = ".pdf",
                             exclude_patterns: List[str] = None,
                             extract_metadata: bool = True,
                             semantic_chunking: bool = False,
                             min_chunk_size: int = 200,
                             max_chunk_size: int = 1500) -> Dict[str, Any]:
        """
        Traite tous les fichiers PDF d'un répertoire et les ajoute à la base de données.
        
        Args:
            pdf_dir: Chemin vers le répertoire contenant les fichiers PDF
            recursive: Si True, parcourt également les sous-répertoires
            chunk_size: Taille des chunks de texte (en caractères) pour le chunking standard
            chunk_overlap: Chevauchement entre les chunks (en caractères) pour le chunking standard
            file_extension: Extension des fichiers à traiter (par défaut .pdf)
            exclude_patterns: Liste de motifs à exclure (sous-chaînes dans les noms de fichiers)
            extract_metadata: Si True, extrait et stocke les métadonnées des PDF
            semantic_chunking: Si True, utilise l'algorithme de chunking sémantique
            min_chunk_size: Taille minimale des chunks pour le chunking sémantique
            max_chunk_size: Taille maximale des chunks pour le chunking sémantique
            
        Returns:
            Statistiques sur le traitement
        """
        if not os.path.exists(pdf_dir):
            raise FileNotFoundError(f"Le répertoire {pdf_dir} n'existe pas")
        
        # Initialisation des statistiques
        stats = {
            "pdf_count": 0,
            "chunk_count": 0,
            "processed_files": [],
            "failed_files": [],
            "skipped_files": [],
            "chunking_method": "semantic" if semantic_chunking else "standard"
        }
        
        # Si exclude_patterns n'est pas fourni, initialiser avec une liste vide
        if exclude_patterns is None:
            exclude_patterns = []
        
        # Extraction du texte de tous les PDF du répertoire en parallèle
        logger.info(f"Extraction du texte des PDF dans {pdf_dir}")
        extracted_texts = self.pdf_extractor.extract_text_from_directory(
            directory_path=pdf_dir, 
            recursive=recursive,
            file_extension=file_extension,
            max_workers=self.max_workers
        )
        
        # Filtrage des fichiers extraits
        filtered_texts = {}
        for pdf_path, page_texts in extracted_texts.items():
            # Vérifier si le fichier correspond à un motif à exclure
            file_name = os.path.basename(pdf_path)
            should_exclude = False
            
            for pattern in exclude_patterns:
                if pattern in file_name:
                    should_exclude = True
                    stats["skipped_files"].append({
                        "file": pdf_path,
                        "reason": f"Correspond au motif d'exclusion: {pattern}"
                    })
                    logger.info(f"Fichier ignoré (motif d'exclusion): {pdf_path}")
                    break
            
            if not should_exclude:
                filtered_texts[pdf_path] = page_texts
        
        # Mise à jour du nombre de PDF
        stats["pdf_count"] = len(filtered_texts)
        logger.info(f"{stats['pdf_count']} fichiers PDF trouvés après filtrage")
        
        # Traitement de chaque PDF
        all_chunks = []
        all_metadatas = []
        
        for pdf_path, page_texts in tqdm(filtered_texts.items(), desc="Traitement des PDF"):
            try:
                # Vérification des erreurs
                if "error" in page_texts:
                    stats["failed_files"].append({
                        "file": pdf_path,
                        "error": page_texts["error"]
                    })
                    continue
                
                # Vérification si le fichier est réellement un PDF valide
                if not page_texts or len(page_texts) == 0:
                    stats["failed_files"].append({
                        "file": pdf_path,
                        "error": "Aucune page extraite - fichier PDF potentiellement invalide"
                    })
                    logger.warning(f"Aucune page extraite de {pdf_path} - ignoré")
                    continue
                
                # Concaténation du texte de toutes les pages
                full_text = ""
                for page_num in sorted(page_texts.keys()):
                    page_content = page_texts[page_num]
                    # Vérifier si le contenu de la page est significatif
                    if len(page_content.strip()) < 50:  # Ignorer les pages avec très peu de contenu
                        continue
                    full_text += page_content + "\n\n"
                
                # Vérifier si le texte obtenu est suffisamment significatif
                if len(full_text.strip()) < 200:  # Ignorer les documents avec très peu de contenu
                    stats["skipped_files"].append({
                        "file": pdf_path,
                        "reason": "Contenu insuffisant (moins de 200 caractères)"
                    })
                    logger.warning(f"Contenu insuffisant dans {pdf_path} - ignoré")
                    continue
                
                # Extraction des métadonnées si demandé
                pdf_metadata = {}
                if extract_metadata:
                    try:
                        pdf_metadata = self.pdf_extractor.extract_metadata(pdf_path)
                        logger.info(f"Métadonnées extraites pour {pdf_path}")
                    except Exception as e:
                        logger.warning(f"Échec de l'extraction des métadonnées pour {pdf_path}: {str(e)}")
                
                # Découpage en chunks selon la méthode choisie
                if semantic_chunking:
                    chunks = self._chunk_text_semantic(full_text, max_chunk_size, min_chunk_size)
                    logger.info(f"Chunking sémantique: {len(chunks)} chunks générés pour {pdf_path}")
                else:
                    chunks = self._chunk_text(full_text, chunk_size, chunk_overlap)
                    logger.info(f"Chunking standard: {len(chunks)} chunks générés pour {pdf_path}")
                
                # Création des métadonnées
                file_name = os.path.basename(pdf_path)
                metadatas = []
                
                for i in range(len(chunks)):
                    # Métadonnées de base
                    chunk_metadata = {
                        "source": file_name,
                        "path": pdf_path,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunking_method": "semantic" if semantic_chunking else "standard"
                    }
                    
                    # Ajouter les métadonnées du PDF si disponibles
                    if pdf_metadata:
                        # Ajouter les champs principaux des métadonnées
                        for key in ["title", "author", "subject", "keywords", "creation_date"]:
                            if key in pdf_metadata and pdf_metadata[key]:
                                chunk_metadata[key] = pdf_metadata[key]
                    
                    metadatas.append(chunk_metadata)
                
                # Ajout aux listes
                all_chunks.extend(chunks)
                all_metadatas.extend(metadatas)
                
                # Mise à jour des statistiques
                stats["chunk_count"] += len(chunks)
                stats["processed_files"].append({
                    "file": pdf_path,
                    "chunks": len(chunks)
                })
                logger.info(f"Traité: {file_name} - {len(chunks)} chunks générés")
                
            except Exception as e:
                logger.error(f"Erreur lors du traitement de {pdf_path}: {str(e)}")
                stats["failed_files"].append({
                    "file": pdf_path,
                    "error": str(e)
                })
        
        # Entraînement du modèle Sentence Transformers
        if all_chunks:
            try:
                logger.info(f"Création des embeddings pour {len(all_chunks)} chunks")
                
                # Générer les embeddings en plus petits lots pour éviter les problèmes de mémoire
                batch_size = 50
                all_embeddings = []
                
                for i in tqdm(range(0, len(all_chunks), batch_size), desc="Génération des embeddings"):
                    end_idx = min(i + batch_size, len(all_chunks))
                    batch_chunks = all_chunks[i:end_idx]
                    
                    try:
                        # Transformation en embeddings
                        batch_embeddings = self.embedding_model.transform(batch_chunks).tolist()
                        all_embeddings.extend(batch_embeddings)
                    except Exception as e:
                        logger.error(f"Erreur lors de la génération des embeddings pour le lot {i//batch_size + 1}: {str(e)}")
                        # On ajoute des embeddings vides pour maintenir l'alignement
                        failed_batch_size = len(batch_chunks)
                        failed_embeddings = [[0.0] * 384] * failed_batch_size  # Dimension par défaut
                        all_embeddings.extend(failed_embeddings)
                
                # Vérifier la cohérence des dimensions
                if all_embeddings:
                    embedding_dims = set(len(emb) for emb in all_embeddings)
                    if len(embedding_dims) > 1:
                        logger.warning(f"Dimensions incohérentes dans les embeddings: {embedding_dims}")
                    else:
                        dim = next(iter(embedding_dims))
                        logger.info(f"Dimension des embeddings: {dim}")
                
                # Générer des IDs uniques
                ids = [f"chunk_{i}_{uuid.uuid4()}" for i in range(len(all_chunks))]
                
                # Ajout à ChromaDB avec gestion d'erreur
                logger.info(f"Ajout des chunks à ChromaDB")
                try:
                    # Utilisation d'une taille de lot réduite pour l'ajout
                    added_ids = self.db_manager.add_documents(
                        texts=all_chunks,
                        metadatas=all_metadatas,
                        ids=ids,
                        embeddings=all_embeddings,
                        batch_size=25  # Taille de lot réduite
                    )
                    
                    logger.info(f"{len(added_ids)} chunks ajoutés avec succès à ChromaDB sur {len(all_chunks)} chunks")
                    if len(added_ids) < len(all_chunks):
                        logger.warning(f"{len(all_chunks) - len(added_ids)} chunks n'ont pas pu être ajoutés")
                except Exception as e:
                    logger.error(f"Erreur lors de l'ajout des chunks à ChromaDB: {str(e)}")
                    stats["error"] = str(e)
                
                # Sauvegarde du modèle Sentence Transformers
                embedding_model_path = os.path.join(self.data_dir, "embedding_model.pkl")
                try:
                    self.embedding_model.save(embedding_model_path)
                    logger.info(f"Modèle Sentence Transformers sauvegardé dans {embedding_model_path}")
                except Exception as e:
                    logger.error(f"Erreur lors de la sauvegarde du modèle: {str(e)}")
            except Exception as e:
                logger.error(f"Erreur globale lors du traitement des embeddings: {str(e)}")
                stats["error"] = str(e)
        else:
            logger.warning("Aucun chunk généré - vérifiez le contenu des fichiers PDF")
        
        logger.info(f"Traitement terminé: {stats['pdf_count']} PDF, {stats['chunk_count']} chunks")
        logger.info(f"Fichiers ignorés: {len(stats['skipped_files'])}, Fichiers avec erreurs: {len(stats['failed_files'])}")
        return stats
    
    def process_single_pdf(self, 
                          pdf_path: str,
                          chunk_size: int = 1000,
                          chunk_overlap: int = 200,
                          extract_metadata: bool = True,
                          extract_images: bool = False,
                          images_dir: str = None,
                          semantic_chunking: bool = False,
                          min_chunk_size: int = 200,
                          max_chunk_size: int = 1500) -> Dict[str, Any]:
        """
        Traite un seul fichier PDF et l'ajoute à la base de données.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            chunk_size: Taille des chunks de texte (en caractères) pour le chunking standard
            chunk_overlap: Chevauchement entre les chunks (en caractères) pour le chunking standard
            extract_metadata: Si True, extrait et stocke les métadonnées du PDF
            extract_images: Si True, extrait les images du PDF
            images_dir: Répertoire pour sauvegarder les images extraites
            semantic_chunking: Si True, utilise l'algorithme de chunking sémantique
            min_chunk_size: Taille minimale des chunks pour le chunking sémantique
            max_chunk_size: Taille maximale des chunks pour le chunking sémantique
            
        Returns:
            Statistiques sur le traitement
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier {pdf_path} n'existe pas")
        
        # Extraction du texte du PDF avec détection OCR si activé
        logger.info(f"Extraction du texte du PDF {pdf_path}")
        
        if self.enable_ocr:
            page_texts = self.pdf_extractor.extract_text_auto(pdf_path, self.ocr_language)
        else:
            page_texts = self.pdf_extractor.extract_text_from_file(pdf_path)
        
        # Concaténation du texte de toutes les pages
        full_text = ""
        for page_num in sorted(page_texts.keys()):
            full_text += page_texts[page_num] + "\n\n"
        
        # Extraction des métadonnées si demandé
        pdf_metadata = {}
        if extract_metadata:
            try:
                pdf_metadata = self.pdf_extractor.extract_metadata(pdf_path)
                logger.info(f"Métadonnées extraites pour {pdf_path}")
            except Exception as e:
                logger.warning(f"Échec de l'extraction des métadonnées pour {pdf_path}: {str(e)}")
        
        # Extraction des images si demandé
        images_info = []
        if extract_images:
            try:
                images_info = self.pdf_extractor.extract_images(pdf_path, images_dir)
                logger.info(f"{len(images_info)} images extraites de {pdf_path}")
            except Exception as e:
                logger.warning(f"Échec de l'extraction des images pour {pdf_path}: {str(e)}")
        
        # Découpage en chunks selon la méthode choisie
        if semantic_chunking:
            chunks = self._chunk_text_semantic(full_text, max_chunk_size, min_chunk_size)
            logger.info(f"Chunking sémantique: {len(chunks)} chunks générés pour {pdf_path}")
        else:
            chunks = self._chunk_text(full_text, chunk_size, chunk_overlap)
            logger.info(f"Chunking standard: {len(chunks)} chunks générés pour {pdf_path}")
        
        # Création des métadonnées
        file_name = os.path.basename(pdf_path)
        metadatas = []
        
        for i in range(len(chunks)):
            # Métadonnées de base
            chunk_metadata = {
                "source": file_name,
                "path": pdf_path,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunking_method": "semantic" if semantic_chunking else "standard"
            }
            
            # Ajouter les métadonnées du PDF si disponibles
            if pdf_metadata:
                # Ajouter les champs principaux des métadonnées
                for key in ["title", "author", "subject", "keywords", "creation_date"]:
                    if key in pdf_metadata and pdf_metadata[key]:
                        chunk_metadata[key] = pdf_metadata[key]
            
            metadatas.append(chunk_metadata)
        
        # Vérification si le modèle Sentence Transformers est déjà entraîné
        if not hasattr(self.embedding_model, 'is_fitted') or not self.embedding_model.is_fitted:
            # Entraînement du modèle Sentence Transformers
            logger.info(f"Entraînement du modèle Sentence Transformers sur {len(chunks)} chunks")
            self.embedding_model.fit(chunks)
        else:
            # Transformation en embeddings
            logger.info(f"Transformation des chunks en embeddings")
        
        # Obtention des embeddings
        embeddings = self.embedding_model.transform(chunks).tolist()
        
        # Ajout à ChromaDB
        logger.info(f"Ajout des chunks à ChromaDB")
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]
        self.db_manager.add_documents(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        
        # Statistiques
        stats = {
            "file": pdf_path,
            "chunk_count": len(chunks),
            "has_metadata": bool(pdf_metadata),
            "images_count": len(images_info),
            "chunking_method": "semantic" if semantic_chunking else "standard"
        }
        
        logger.info(f"Traitement terminé: {len(chunks)} chunks créés pour {pdf_path}")
        return stats
    
    def query(self, 
            query_text: str, 
            n_results: int = 5,
            where: Optional[Dict[str, Any]] = None,
            save_to_history: bool = True,
            suggest_similar_queries: bool = True) -> Dict[str, Any]:
        """
        Exécute une requête RAG complète.
        
        Args:
            query_text: Texte de la requête
            n_results: Nombre de résultats à récupérer
            where: Filtre sur les métadonnées
            save_to_history: Si True, sauvegarde la requête dans l'historique
            suggest_similar_queries: Si True, suggère des requêtes similaires depuis l'historique
            
        Returns:
            Résultat complet de la requête RAG
        """
        # Vérifier les requêtes similaires dans l'historique si activé
        similar_queries = []
        if self.enable_query_history and suggest_similar_queries:
            similar_queries = self.query_history.search_similar_queries(query_text, n_results=3)
            if similar_queries:
                logger.info(f"Trouvé {len(similar_queries)} requêtes similaires dans l'historique")
        
        # Prétraitement de la requête
        preprocessed_query = self.embedding_model.preprocess_text(query_text)
        
        # Recherche dans ChromaDB
        logger.info(f"Recherche pour la requête: '{query_text}'")
        
        # Utilisation de la méthode search avec query_texts
        results = self.db_manager.search(
            query_texts=[preprocessed_query],  # ChromaDB attend une liste
            n_results=n_results,
            where=where
        )
        
        # Extraction des documents et métadonnées
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        system_instruction = """
            Tu es un expert en recherche académique qui répond aux questions en te basant uniquement sur les documents fournis.
            Respecte scrupuleusement ces instructions:
            1. Utilise uniquement les informations présentes dans les documents fournis pour répondre.
            2. Si les documents ne contiennent pas l'information nécessaire pour répondre à la question, indique clairement que tu ne peux pas répondre avec les informations disponibles.
            3. Citations des sources:
               - Si le document contient un champ "author" dans ses métadonnées, cite ce document selon le format : (Auteur, Document N).
               - Si le document ne contient pas de champ "author" dans ses métadonnées, vérifie dans le titre du document si il y a un auteur : (Auteur, Document N).
               - Si aucun auteur n'est disponible, cite clairement le numéro du document sous la forme : (Document N).
               - Tu dois obligatoirement citer les sources pour chaque information importante, même lorsque tu reformules.
            4. À la fin de ta réponse, présente systématiquement la section "Sources" qui liste les documents utilisés sous la forme suivante:
               Sources:
               - Document 1: [Titre ou premiers mots]
               - Document 2: [Titre ou premiers mots]
            5. Priorise les sources ayant un score de similarité plus élevé, elles sont généralement plus pertinentes.
            6. Reste factuel et objectif, en te limitant strictement au contenu des documents fournis.
            """
        
        # Conversion des distances en scores de similarité (1 - distance)
        similarity_scores = [1 - dist for dist in distances]
        
        # Génération de la réponse avec Gemini
        logger.info(f"Génération de la réponse avec Gemini")
        rag_response = self.gemini_api.generate_rag_response(
            query=query_text,
            retrieved_documents=documents,
            document_scores=similarity_scores,
            document_metadatas=metadatas, 
            system_instruction=system_instruction
        )
        
        # Ajout des informations de recherche au résultat
        rag_response['search_info'] = {
            'query': query_text,
            'preprocessed_query': preprocessed_query,
            'n_results': n_results,
            'where_filter': where
        }
        
        # Ajout des requêtes similaires de l'historique si disponibles
        if similar_queries:
            rag_response['similar_queries'] = [
                {
                    "query": q["query"],
                    "timestamp": q["timestamp"],
                    "sources_count": q["sources_count"]
                }
                for q in similar_queries
            ]
        
        # Enregistrement dans l'historique si activé
        if self.enable_query_history and save_to_history:
            self.query_history.add_query(query_text, rag_response)
            logger.debug(f"Requête ajoutée à l'historique: '{query_text}'")
        
        logger.info(f"Requête RAG complétée")
        return rag_response

    def get_query_history(self, limit: int = None, filter_query: str = None) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des requêtes.
        
        Args:
            limit: Nombre maximum d'entrées à récupérer
            filter_query: Filtre textuel pour rechercher dans les requêtes
            
        Returns:
            Liste des entrées d'historique ou None si l'historique est désactivé
        """
        if not self.enable_query_history or self.query_history is None:
            return None
        
        return self.query_history.get_history(limit, filter_query)
    
    def clear_query_history(self) -> bool:
        """
        Efface tout l'historique des requêtes.
        
        Returns:
            True si l'opération a réussi, False si l'historique est désactivé
        """
        if not self.enable_query_history or self.query_history is None:
            return False
        
        self.query_history.clear_history()
        logger.info("Historique des requêtes effacé")
        return True

    def save_config(self, config_path: str) -> None:
        """
        Sauvegarde la configuration du pipeline dans un fichier JSON.
        
        Args:
            config_path: Chemin vers le fichier de configuration à créer
        """
        config = {
            "data_dir": self.data_dir,
            "db_dir": self.db_dir,
            "collection_name": self.collection_name,
            "gemini_model": self.gemini_api.model_name,
            "embedding_model": self.embedding_model_name,
            "pdf_extractor": {
                "remove_headers_footers": self.pdf_extractor.remove_headers_footers,
                "min_line_length": self.pdf_extractor.min_line_length,
                "cache_dir": self.pdf_extractor.cache_dir
            },
            "ocr": {
                "enabled": self.enable_ocr,
                "language": self.ocr_language
            },
            "max_workers": self.max_workers,
            "query_history": {
                "enabled": self.enable_query_history,
                "history_file": self.query_history.history_file if self.enable_query_history else None,
                "max_entries": self.query_history.max_entries if self.enable_query_history else 100
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Configuration du pipeline sauvegardée dans {config_path}")

    @classmethod
    def load_from_config(cls, config_path: str, gemini_api_key: Optional[str] = None) -> 'RAGPipeline':
        """
        Charge un pipeline RAG à partir d'un fichier de configuration.
        
        Args:
            config_path: Chemin vers le fichier de configuration
            gemini_api_key: Clé API Gemini (si None, utilise celle du fichier .env)
            
        Returns:
            Instance de RAGPipeline initialisée avec les paramètres du fichier de configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Extraction des paramètres de base
        data_dir = config.get("data_dir", "./data")
        db_dir = config.get("db_dir", "./chroma_db")
        collection_name = config.get("collection_name", "docss")
        gemini_model = config.get("gemini_model", "gemini-2.0-flash-001")
        embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
        
        # Extraction des paramètres de l'extracteur PDF
        pdf_config = config.get("pdf_extractor", {})
        remove_headers_footers = pdf_config.get("remove_headers_footers", True)
        min_line_length = pdf_config.get("min_line_length", 10)
        cache_dir = pdf_config.get("cache_dir")
        
        # Extraction des paramètres OCR
        ocr_config = config.get("ocr", {})
        enable_ocr = ocr_config.get("enabled", False)
        ocr_language = ocr_config.get("language", "fra")
        
        # Extraction des paramètres d'historique
        history_config = config.get("query_history", {})
        enable_query_history = history_config.get("enabled", True)
        history_file = history_config.get("history_file")
        max_history_entries = history_config.get("max_entries", 100)
        
        # Autres paramètres
        max_workers = config.get("max_workers")
        
        # Création du pipeline
        pipeline = cls(
            data_dir=data_dir,
            db_dir=db_dir,
            collection_name=collection_name,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            remove_headers_footers=remove_headers_footers,
            min_line_length=min_line_length,
            embedding_model_name=embedding_model,
            enable_cache=bool(cache_dir),
            enable_ocr=enable_ocr,
            max_workers=max_workers,
            enable_query_history=enable_query_history,
            history_file=history_file,
            max_history_entries=max_history_entries
        )
        
        logger.info(f"Pipeline RAG chargé depuis {config_path}")
        return pipeline

    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Récupère les statistiques de la collection ChromaDB.
        
        Returns:
            Statistiques de la collection
        """
        count = self.db_manager.get_collection_count()
        
        # Récupération des sources uniques
        sources = set()
        if count > 0:
            all_metadata = self.db_manager.get_collection().get(include=["metadatas"])["metadatas"]
            for metadata in all_metadata:
                if "source" in metadata:
                    sources.add(metadata["source"])
        
        return {
            "document_count": count,
            "unique_sources": len(sources),
            "sources": list(sources),
            "collection_name": self.collection_name,
            "db_directory": self.db_dir
        }
    
    def clear_collection(self) -> None:
        """
        Méthode désactivée pour éviter la suppression involontaire des documents.
        """
        logger.warning("La fonction clear_collection() a été désactivée pour éviter la perte accidentelle de données.")
        logger.info("Les documents existants dans la collection ont été conservés.")
        
        # Ancienne implémentation (commentée pour référence):
        # logger.warning(f"Suppression de tous les documents de la collection {self.collection_name}")
        # self.db_manager.clear_collection()
        # logger.info(f"Collection {self.collection_name} vidée avec succès")
    
    def batch_query(self, 
                   queries: List[str], 
                   n_results: int = 5,
                   where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Exécute un lot de requêtes RAG.
        
        Args:
            queries: Liste des requêtes
            n_results: Nombre de résultats à récupérer par requête
            where: Filtre sur les métadonnées
            
        Returns:
            Liste des résultats pour chaque requête
        """
        results = []
        
        for query in tqdm(queries, desc="Traitement des requêtes"):
            try:
                result = self.query(query, n_results=n_results, where=where)
                results.append(result)
                # Pause pour éviter de surcharger l'API
                time.sleep(1)
            except Exception as e:
                logger.error(f"Erreur lors du traitement de la requête '{query}': {str(e)}")
                results.append({
                    "query": query,
                    "error": str(e),
                    "success": False
                })
        
        return results
    
    def export_documents(self, output_path: str) -> Dict[str, Any]:
        """
        Exporte tous les documents et métadonnées de la collection ChromaDB.
        
        Args:
            output_path: Chemin du fichier de sortie (JSON)
            
        Returns:
            Statistiques sur l'exportation
        """
        if self.db_manager.get_collection_count() == 0:
            logger.warning("La collection est vide, rien à exporter")
            return {"success": False, "message": "Collection vide"}
        
        # Récupération de tous les documents
        collection_data = self.db_manager.get_collection().get(
            include=["documents", "metadatas", "embeddings"]
        )
        
        export_data = {
            "documents": collection_data["documents"],
            "metadatas": collection_data["metadatas"],
            "ids": collection_data["ids"],
            "export_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "collection_name": self.collection_name
        }
        
        # Sauvegarde des données
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Collection exportée avec succès vers {output_path}")
        return {
            "success": True,
            "document_count": len(export_data["documents"]),
            "file_path": output_path
        }

    def _reset_collection(self, db_dir: str, collection_name: str) -> None:
        """
        Méthode modifiée pour ne plus supprimer la collection ChromaDB.
        Cette méthode est désactivée pour éviter la perte de données.
        
        Args:
            db_dir: Chemin du répertoire de la base de données
            collection_name: Nom de la collection à réinitialiser
        """
        logger.warning(f"La suppression de la collection ChromaDB a été désactivée pour éviter la perte de données.")
        logger.info(f"Les documents existants dans la collection '{collection_name}' sont conservés.")
        
        # Ancienne implémentation (commentée pour référence):
        # try:
        #     import shutil
        #     import chromadb
        #     
        #     # Tentative de suppression via l'API ChromaDB
        #     try:
        #         client = chromadb.PersistentClient(path=db_dir)
        #         client.delete_collection(collection_name)
        #         logger.info(f"Collection '{collection_name}' supprimée via l'API ChromaDB")
        #     except Exception as e:
        #         logger.warning(f"Impossible de supprimer la collection via l'API: {str(e)}")
        #         
        #         # Suppression manuelle des fichiers de la collection
        #         collection_path = os.path.join(db_dir, collection_name)
        #         if os.path.exists(collection_path):
        #             shutil.rmtree(collection_path)
        #             logger.info(f"Répertoire de collection '{collection_path}' supprimé manuellement")
        #         
        #         # Supprimer également le fichier d'index si présent
        #         index_path = os.path.join(db_dir, "chroma.sqlite3")
        #         if os.path.exists(index_path):
        #             os.remove(index_path)
        #             logger.info(f"Fichier d'index '{index_path}' supprimé")
        #     
        #     logger.info(f"Collection '{collection_name}' réinitialisée avec succès")
        # except Exception as e:
        #     logger.error(f"Erreur lors de la réinitialisation de la collection: {str(e)}")
        #     raise

    def _chunk_text(self, 
                   text: str, 
                   chunk_size: int = 1000, 
                   chunk_overlap: int = 200) -> List[str]:
        """
        Découpe un texte en chunks de taille spécifiée avec chevauchement.
        
        Args:
            text: Texte à découper
            chunk_size: Taille des chunks (en caractères)
            chunk_overlap: Chevauchement entre les chunks (en caractères)
            
        Returns:
            Liste des chunks de texte
        """
        if not text:
            return []
        
        # Découpage en paragraphes
        paragraphs = text.split('\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # Si le paragraphe est trop long, le découper
            if len(paragraph) > chunk_size:
                # Ajouter le chunk actuel s'il n'est pas vide
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Découper le paragraphe en chunks
                for i in range(0, len(paragraph), chunk_size - chunk_overlap):
                    chunk = paragraph[i:i + chunk_size]
                    if len(chunk) < chunk_size / 2 and i > 0:
                        # Si le dernier morceau est trop petit, l'ajouter au chunk précédent
                        chunks[-1] += " " + chunk
                    else:
                        chunks.append(chunk)
            
            # Si l'ajout du paragraphe dépasse la taille du chunk
            elif len(current_chunk) + len(paragraph) > chunk_size:
                # Ajouter le chunk actuel et commencer un nouveau
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # Ajouter le paragraphe au chunk actuel
                if current_chunk:
                    current_chunk += "\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Ajouter le dernier chunk s'il n'est pas vide
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _chunk_text_semantic(self, 
                        text: str,
                        max_chunk_size: int = 1500,
                        min_chunk_size: int = 200) -> List[str]:
        """
        Découpe un texte selon sa structure logique (titres, paragraphes).
        
        Cette méthode tente de préserver la structure sémantique du document en:
        1. Identifiant les titres et sous-titres potentiels
        2. Regroupant les paragraphes qui appartiennent à la même section
        3. Divisant les sections trop grandes en respectant les limites de paragraphes
        
        Args:
            text: Texte à découper
            max_chunk_size: Taille maximale des chunks (en caractères)
            min_chunk_size: Taille minimale des chunks (en caractères)
            
        Returns:
            Liste des chunks de texte avec préservation de la structure logique
        """
        if not text:
            return []
        
        # Nettoyer le texte et le diviser en lignes non vides
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if not lines:
            return []
        
        # Identifier les lignes qui ressemblent à des titres
        title_patterns = [
            r'^#+\s+.+$',  # Format Markdown (# Titre)
            r'^[A-Z0-9][A-Z0-9\s\.]{2,70}$',  # TITRE EN MAJUSCULES
            r'^[IVX]+\.\s+.+$',  # Format romain (I. Titre)
            r'^[0-9]+\.[0-9]*\s+.+$',  # Format numérique (1.2 Titre)
            r'^(Chapter|Section|Chapitre|Section|Partie)\s+[0-9IVX]+',  # Mots-clés de chapitres
            r'^[A-Z][a-z].{5,70}$'  # Phrase courte commençant par une majuscule (possible titre)
        ]
        
        # Fonction pour détecter si une ligne est un titre
        def is_title(line):
            # Lignes très courtes sont probablement des titres
            if len(line) < 60 and line.endswith((':', '.')):
                return True
                
            # Lignes correspondant aux patterns de titres
            for pattern in title_patterns:
                if re.match(pattern, line):
                    return True
            
            # Lignes avec formatage particulier (toutes majuscules, etc.)
            if line.isupper() and 3 < len(line) < 100:
                return True
                
            # Autres heuristiques
            if len(line) < 60 and not line.endswith(('.', ',', ';', ':', '?', '!')):
                return True
                
            return False
        
        # Structure provisoire: tableau de sections [titre, contenu]
        sections = []
        current_title = None
        current_content = []
        
        # Parcourir toutes les lignes
        for line in lines:
            if is_title(line):
                # Si on avait déjà un titre et du contenu, on sauvegarde
                if current_title and current_content:
                    sections.append((current_title, '\n'.join(current_content)))
                
                # Commencer une nouvelle section
                current_title = line
                current_content = []
            else:
                # Ajouter au contenu de la section actuelle
                current_content.append(line)
        
        # Ajouter la dernière section
        if current_title and current_content:
            sections.append((current_title, '\n'.join(current_content)))
        elif not sections and current_content:
            # Si aucun titre n'a été identifié, utiliser le contenu comme une seule section
            sections.append((None, '\n'.join(current_content)))
        
        # Diviser les sections qui dépassent la taille maximale
        chunks = []
        for title, content in sections:
            section_text = f"{title}\n\n{content}" if title else content
            
            if len(section_text) <= max_chunk_size:
                # La section est assez petite, l'ajouter comme chunk
                if len(section_text) >= min_chunk_size:
                    chunks.append(section_text)
                elif chunks:  # Si trop petite, l'ajouter au chunk précédent si possible
                    chunks[-1] += f"\n\n{section_text}"
                else:  # Sinon, la garder comme chunk malgré sa petite taille
                    chunks.append(section_text)
            else:
                # La section est trop grande, la diviser en paragraphes
                paragraphs = content.split('\n\n')
                current_chunk = title + "\n\n" if title else ""
                
                for paragraph in paragraphs:
                    if len(paragraph) > max_chunk_size:
                        # Le paragraphe lui-même est trop long, l'ajouter séparément 
                        if current_chunk and len(current_chunk) >= min_chunk_size:
                            chunks.append(current_chunk)
                            
                        # Diviser le paragraphe en phrases
                        sentences = re.split(r'(?<=[.!?])\s+', paragraph)
                        current_chunk = ""
                        
                        for sentence in sentences:
                            if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                                current_chunk += " " + sentence if current_chunk else sentence
                            else:
                                if current_chunk and len(current_chunk) >= min_chunk_size:
                                    chunks.append(current_chunk)
                                current_chunk = sentence
                        
                        if current_chunk and len(current_chunk) >= min_chunk_size:
                            chunks.append(current_chunk)
                        current_chunk = ""
                    
                    elif len(current_chunk) + len(paragraph) + 2 <= max_chunk_size:
                        # Ajouter le paragraphe au chunk actuel
                        if current_chunk:
                            current_chunk += f"\n\n{paragraph}"
                        else:
                            current_chunk = paragraph
                    else:
                        # Le chunk actuel est plein, en commencer un nouveau
                        if current_chunk and len(current_chunk) >= min_chunk_size:
                            chunks.append(current_chunk)
                        current_chunk = paragraph
                
                # Ajouter le dernier chunk de cette section
                if current_chunk and len(current_chunk) >= min_chunk_size:
                    chunks.append(current_chunk)
        
        # S'assurer qu'aucun chunk n'est trop petit
        if chunks and len(chunks) > 1:
            # Combiner les chunks trop petits avec les précédents ou suivants
            i = 0
            while i < len(chunks) - 1:
                if len(chunks[i]) < min_chunk_size:
                    chunks[i+1] = chunks[i] + "\n\n" + chunks[i+1]
                    chunks.pop(i)
                else:
                    i += 1
                    
            # Vérifier le dernier chunk
            if len(chunks[-1]) < min_chunk_size and len(chunks) > 1:
                chunks[-2] += "\n\n" + chunks[-1]
                chunks.pop(-1)
        
        return chunks

class QueryHistory:
    """Classe pour gérer l'historique des requêtes."""
    
    def __init__(self, history_file: str = None, max_entries: int = 100):
        """
        Initialise l'historique des requêtes.
        
        Args:
            history_file: Chemin vers le fichier de sauvegarde de l'historique
            max_entries: Nombre maximum d'entrées à conserver
        """
        self.history_file = history_file
        self.max_entries = max_entries
        self.history = []
        
        # Charger l'historique existant si le fichier existe
        if history_file and os.path.exists(history_file):
            try:
                with open(history_file, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
                logger.info(f"Historique chargé depuis {history_file}: {len(self.history)} entrées")
            except Exception as e:
                logger.error(f"Erreur lors du chargement de l'historique depuis {history_file}: {str(e)}")
                self.history = []
    
    def add_query(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ajoute une requête à l'historique.
        
        Args:
            query: Texte de la requête
            result: Résultat de la requête
            
        Returns:
            Entrée ajoutée à l'historique
        """
        # Créer une entrée d'historique
        entry = {
            "query": query,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "response": result.get("response", ""),
            "sources_count": len(result.get("sources", [])),
            "sources": [
                {
                    "source": s["metadata"].get("source", ""),
                    "score": s["score"],
                    "title": s["metadata"].get("title", ""),
                    "author": s["metadata"].get("author", "")
                }
                for s in result.get("sources", [])
            ]
        }
        
        # Ajouter à l'historique
        self.history.insert(0, entry)
        
        # Limiter la taille de l'historique
        if len(self.history) > self.max_entries:
            self.history = self.history[:self.max_entries]
        
        # Sauvegarder l'historique si un fichier est spécifié
        if self.history_file:
            try:
                with open(self.history_file, 'w', encoding='utf-8') as f:
                    json.dump(self.history, f, ensure_ascii=False, indent=2)
                logger.debug(f"Historique sauvegardé dans {self.history_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de l'historique dans {self.history_file}: {str(e)}")
        
        return entry
    
    def get_history(self, limit: int = None, filter_query: str = None) -> List[Dict[str, Any]]:
        """
        Récupère l'historique des requêtes.
        
        Args:
            limit: Nombre maximum d'entrées à récupérer
            filter_query: Filtre textuel pour rechercher dans les requêtes
            
        Returns:
            Liste des entrées d'historique
        """
        if not self.history:
            return []
        
        # Appliquer le filtre si spécifié
        if filter_query:
            filtered_history = [
                entry for entry in self.history
                if filter_query.lower() in entry["query"].lower()
            ]
        else:
            filtered_history = self.history
        
        # Limiter le nombre d'entrées si spécifié
        if limit is not None and limit > 0:
            return filtered_history[:limit]
        
        return filtered_history
    
    def clear_history(self) -> None:
        """Efface tout l'historique des requêtes."""
        self.history = []
        
        # Supprimer le fichier d'historique s'il existe
        if self.history_file and os.path.exists(self.history_file):
            try:
                os.remove(self.history_file)
                logger.info(f"Fichier d'historique supprimé: {self.history_file}")
            except Exception as e:
                logger.error(f"Erreur lors de la suppression du fichier d'historique {self.history_file}: {str(e)}")
    
    def search_similar_queries(self, query: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Recherche des requêtes similaires dans l'historique.
        
        Args:
            query: Texte de la requête
            n_results: Nombre de résultats à retourner
            
        Returns:
            Liste des entrées d'historique les plus similaires
        """
        if not self.history:
            return []
        
        # Méthode simple: recherche de sous-chaînes
        # On pourrait améliorer avec des embeddings pour une véritable recherche sémantique
        query_lower = query.lower()
        scored_entries = []
        
        for entry in self.history:
            entry_query = entry["query"].lower()
            
            # Score basique: nombre de mots en commun
            query_words = set(query_lower.split())
            entry_words = set(entry_query.split())
            common_words = query_words.intersection(entry_words)
            
            if common_words:
                # Score = proportion de mots en commun
                score = len(common_words) / max(len(query_words), len(entry_words))
                scored_entries.append((score, entry))
        
        # Trier par score et retourner les n_results premiers
        scored_entries.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored_entries[:n_results]]
