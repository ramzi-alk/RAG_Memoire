"""
Point d'entrée principal pour le système RAG.

Ce script permet d'utiliser le système RAG via une interface en ligne de commande.
"""

import os
import logging
import argparse
from src.pipeline.rag_pipeline import RAGPipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Fonction principale pour l'interface en ligne de commande."""
    parser = argparse.ArgumentParser(description="Système RAG pour l'extraction et la recherche d'informations")
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")
    
    # Commande d'initialisation
    init_parser = subparsers.add_parser("init", help="Initialiser le pipeline RAG")
    init_parser.add_argument("--data-dir", default="./data", help="Répertoire pour les données")
    init_parser.add_argument("--db-dir", default="./chroma_db", help="Répertoire pour la base de données")
    init_parser.add_argument("--collection", default="docss", help="Nom de la collection")
    init_parser.add_argument("--api-key", required=True, help="Clé API Gemini")
    init_parser.add_argument("--model", default="gemini-2.0-flash-001", help="Modèle Gemini à utiliser")
    init_parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", 
                            help="Modèle Sentence Transformers à utiliser pour les embeddings")
    init_parser.add_argument("--language", default="french", help="Langue pour le prétraitement")
    init_parser.add_argument("--config", default="./config.json", help="Fichier de configuration à créer")
    init_parser.add_argument("--reset", action="store_true", 
                            help="Réinitialiser la collection existante (si vous changez de modèle d'embedding)")
    init_parser.add_argument("--enable-cache", action="store_true", 
                            help="Activer la mise en cache des extractions PDF")
    init_parser.add_argument("--enable-ocr", action="store_true", 
                            help="Activer la détection et l'OCR pour les PDFs scannés")
    init_parser.add_argument("--max-workers", type=int, default=4,
                            help="Nombre maximum de workers pour le traitement parallèle (0 ou 1 pour séquentiel)")
    init_parser.add_argument("--no-history", action="store_true",
                            help="Désactiver l'historique des requêtes")
    init_parser.add_argument("--history-file", 
                            help="Chemin vers le fichier de sauvegarde de l'historique")
    init_parser.add_argument("--max-history", type=int, default=100,
                            help="Nombre maximum d'entrées d'historique à conserver")
    
    # Commande de traitement de PDF
    process_parser = subparsers.add_parser("process", help="Traiter des fichiers PDF")
    process_parser.add_argument("--pdf-dir", help="Répertoire contenant les fichiers PDF")
    process_parser.add_argument("--pdf-file", help="Fichier PDF individuel à traiter")
    process_parser.add_argument("--recursive", action="store_true", help="Parcourir les sous-répertoires")
    
    # Options de chunking
    chunk_group = process_parser.add_argument_group("Options de chunking")
    chunk_group.add_argument("--semantic-chunking", action="store_true", 
                            help="Utiliser l'algorithme de chunking sémantique basé sur la structure du document")
    chunk_group.add_argument("--chunk-size", type=int, default=1000, 
                            help="Taille des chunks de texte pour le chunking standard")
    chunk_group.add_argument("--chunk-overlap", type=int, default=200, 
                            help="Chevauchement entre les chunks pour le chunking standard")
    chunk_group.add_argument("--min-chunk-size", type=int, default=200, 
                            help="Taille minimale des chunks pour le chunking sémantique")
    chunk_group.add_argument("--max-chunk-size", type=int, default=1500, 
                            help="Taille maximale des chunks pour le chunking sémantique")
    
    process_parser.add_argument("--config", default="./config.json", help="Fichier de configuration")
    process_parser.add_argument("--api-key", help="Clé API Gemini (remplace celle du fichier .env)")
    process_parser.add_argument("--exclude", help="Motifs à exclure (séparés par des virgules)")
    process_parser.add_argument("--min-content", type=int, default=200, 
                               help="Taille minimale de contenu pour considérer un document (caractères)")
    process_parser.add_argument("--extension", default=".pdf", help="Extension des fichiers à traiter")
    process_parser.add_argument("--reset", action="store_true", 
                              help="Réinitialiser la collection existante avant le traitement")
    process_parser.add_argument("--extract-metadata", action="store_true", 
                              help="Extraire et stocker les métadonnées des PDF")
    process_parser.add_argument("--extract-images", action="store_true",
                              help="Extraire les images des PDF (uniquement avec --pdf-file)")
    process_parser.add_argument("--images-dir", 
                              help="Répertoire où sauvegarder les images extraites (créé si n'existe pas)")
    process_parser.add_argument("--max-workers", type=int, default=4,
                              help="Nombre maximum de workers pour le traitement parallèle (0 ou 1 pour séquentiel)")
    process_parser.add_argument("--sequential", action="store_true",
                              help="Forcer le traitement séquentiel (équivalent à --max-workers=1)")
    
    # Commande de requête
    query_parser = subparsers.add_parser("query", help="Exécuter une requête RAG")
    query_parser.add_argument("--query", required=True, help="Texte de la requête")
    query_parser.add_argument("--n-results", type=int, default=5, help="Nombre de résultats à récupérer")
    query_parser.add_argument("--source", help="Filtrer par source (nom de fichier)")
    query_parser.add_argument("--config", default="./config.json", help="Fichier de configuration")
    query_parser.add_argument("--api-key", help="Clé API Gemini (remplace celle du fichier .env)")
    
    # Commande pour réinitialiser la collection (en cas de changement de modèle)
    reset_parser = subparsers.add_parser("reset", help="Réinitialiser la collection ChromaDB")
    reset_parser.add_argument("--config", default="./config.json", help="Fichier de configuration")
    reset_parser.add_argument("--api-key", help="Clé API Gemini (remplace celle du fichier .env)")
    
    # Commande pour gérer l'historique des requêtes
    history_parser = subparsers.add_parser("history", help="Gérer l'historique des requêtes")
    history_parser.add_argument("--list", action="store_true", help="Lister les requêtes de l'historique")
    history_parser.add_argument("--limit", type=int, default=10, help="Nombre maximum de requêtes à afficher")
    history_parser.add_argument("--filter", help="Filtre textuel pour rechercher dans les requêtes")
    history_parser.add_argument("--clear", action="store_true", help="Effacer tout l'historique des requêtes")
    history_parser.add_argument("--config", default="./config.json", help="Fichier de configuration")
    history_parser.add_argument("--api-key", help="Clé API Gemini (remplace celle du fichier .env)")
    
    args = parser.parse_args()
    
    # Exécution des commandes
    if args.command == "init":
        # Création des répertoires si nécessaire
        os.makedirs(args.data_dir, exist_ok=True)
        os.makedirs(args.db_dir, exist_ok=True)
        
        # Initialisation du pipeline
        pipeline = RAGPipeline(
            data_dir=args.data_dir,
            db_dir=args.db_dir,
            collection_name=args.collection,
            gemini_api_key=args.api_key,
            gemini_model=args.model,
            language=args.language,
            embedding_model_name=args.embedding_model,
            reset_collection=args.reset,
            enable_cache=args.enable_cache,
            enable_ocr=args.enable_ocr,
            max_workers=args.max_workers,
            enable_query_history=not args.no_history,
            history_file=args.history_file,
            max_history_entries=args.max_history
        )
        
        # Sauvegarde de la configuration
        pipeline.save_config(args.config)
        print(f"Pipeline RAG initialisé et configuration sauvegardée dans {args.config}")
        print(f"Modèle d'embedding utilisé: {args.embedding_model}")
        
        if args.enable_cache:
            print(f"Cache d'extraction PDF activé dans {os.path.join(args.data_dir, 'pdf_cache')}")
        
        if args.enable_ocr:
            print(f"Support OCR activé pour les PDFs scannés (langue: {args.language[:3]})")
        
        if not args.no_history:
            history_file = args.history_file or os.path.join(args.data_dir, "query_history.json")
            print(f"Historique des requêtes activé dans {history_file}")
        else:
            print("Historique des requêtes désactivé")
        
        if args.reset:
            print("La collection a été réinitialisée. Vous devez traiter à nouveau vos documents.")
        
    elif args.command == "process":
        # Vérification du fichier de configuration
        if not os.path.exists(args.config):
            print(f"Erreur: Le fichier de configuration {args.config} n'existe pas")
            print("Exécutez d'abord la commande 'init' pour créer une configuration")
            return
        
        # Chargement du pipeline depuis la configuration
        pipeline = RAGPipeline.load_from_config(args.config, gemini_api_key=args.api_key)
        
        # Si --sequential est spécifié, forcer le traitement séquentiel
        if hasattr(args, 'sequential') and args.sequential:
            args.max_workers = 1
            print("Mode séquentiel forcé: --max-workers=1")
        
        # Mettre à jour le nombre de workers
        if hasattr(args, 'max_workers') and args.max_workers is not None:
            pipeline.max_workers = args.max_workers
            print(f"Nombre de workers: {args.max_workers}")
        
        # Si --reset est spécifié, réinitialiser la collection
        if args.reset:
            pipeline._reset_collection(pipeline.db_dir, pipeline.collection_name)
            print("Collection réinitialisée.")
        
        # Traitement des PDF
        if args.pdf_dir:
            if not os.path.exists(args.pdf_dir):
                print(f"Erreur: Le répertoire {args.pdf_dir} n'existe pas")
                return
            
            # Préparation des motifs d'exclusion
            exclude_patterns = None
            if args.exclude:
                exclude_patterns = [p.strip() for p in args.exclude.split(',') if p.strip()]
            
            stats = pipeline.process_pdf_directory(
                pdf_dir=args.pdf_dir,
                recursive=args.recursive,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                exclude_patterns=exclude_patterns,
                file_extension=args.extension,
                extract_metadata=args.extract_metadata,
                semantic_chunking=args.semantic_chunking,
                min_chunk_size=args.min_chunk_size,
                max_chunk_size=args.max_chunk_size
            )
            
            print(f"Traitement terminé: {stats['pdf_count']} PDF, {stats['chunk_count']} chunks")
            print(f"Méthode de chunking: {stats['chunking_method']}")
            print(f"Fichiers ignorés: {len(stats.get('skipped_files', []))}, Fichiers en erreur: {len(stats.get('failed_files', []))}")
            
        elif args.pdf_file:
            if not os.path.exists(args.pdf_file):
                print(f"Erreur: Le fichier {args.pdf_file} n'existe pas")
                return
            
            # Création du répertoire pour les images si nécessaire
            if args.extract_images and args.images_dir:
                os.makedirs(args.images_dir, exist_ok=True)
            
            stats = pipeline.process_single_pdf(
                pdf_path=args.pdf_file,
                chunk_size=args.chunk_size,
                chunk_overlap=args.chunk_overlap,
                extract_metadata=args.extract_metadata,
                extract_images=args.extract_images,
                images_dir=args.images_dir,
                semantic_chunking=args.semantic_chunking,
                min_chunk_size=args.min_chunk_size,
                max_chunk_size=args.max_chunk_size
            )
            
            print(f"Traitement terminé: {stats['chunk_count']} chunks créés pour {args.pdf_file}")
            print(f"Méthode de chunking: {stats['chunking_method']}")
            
            if stats.get('has_metadata', False):
                print("Métadonnées extraites et stockées avec les chunks")
                
            if stats.get('images_count', 0) > 0:
                print(f"{stats['images_count']} images extraites et sauvegardées dans {args.images_dir}")
            
        else:
            print("Erreur: Vous devez spécifier --pdf-dir ou --pdf-file")
            
    elif args.command == "query":
        # Vérification du fichier de configuration
        if not os.path.exists(args.config):
            print(f"Erreur: Le fichier de configuration {args.config} n'existe pas")
            print("Exécutez d'abord la commande 'init' pour créer une configuration")
            return
        
        # Chargement du pipeline depuis la configuration
        pipeline = RAGPipeline.load_from_config(args.config, gemini_api_key=args.api_key)
        
        # Préparation du filtre
        where_filter = None
        if args.source:
            where_filter = {"source": args.source}
        
        # Exécution de la requête
        import time
        start_time = time.time()
        result = pipeline.query(
            query_text=args.query,
            n_results=args.n_results,
            where=where_filter
        )
        end_time = time.time()
        
        # Affichage des résultats
        print("\n" + "="*80)
        print(f"Requête: {args.query}")
        print(f"Temps d'exécution: {end_time - start_time:.2f} secondes")
        print("="*80 + "\n")
        
        print("Réponse:")
        print(result['response'])
        print("\n" + "-"*80 + "\n")
        
        print("Sources utilisées:")
        for i, source in enumerate(result['sources']):
            # Afficher les métadonnées supplémentaires si disponibles
            author = source['metadata'].get('author', 'Auteur inconnu')
            title = source['metadata'].get('title', source['metadata']['source'])
            
            print(f"{i+1}. {title} (Auteur: {author}, Score: {source['score']:.4f})")
            print(f"   Fichier: {source['metadata']['source']}")
            print(f"   Extrait: {source['document'][:150]}..." if len(source['document']) > 150 else f"   Extrait: {source['document']}")
            print()
    
    elif args.command == "reset":
        # Vérification du fichier de configuration
        if not os.path.exists(args.config):
            print(f"Erreur: Le fichier de configuration {args.config} n'existe pas")
            print("Exécutez d'abord la commande 'init' pour créer une configuration")
            return
        
        # Chargement du pipeline depuis la configuration
        pipeline = RAGPipeline.load_from_config(args.config, gemini_api_key=args.api_key)
        
        # Réinitialisation de la collection
        pipeline._reset_collection(pipeline.db_dir, pipeline.collection_name)
        print(f"Collection '{pipeline.collection_name}' réinitialisée avec succès.")
        print("Vous devez maintenant traiter à nouveau vos documents avec la commande 'process'.")
        
    elif args.command == "history":
        # Vérification du fichier de configuration
        if not os.path.exists(args.config):
            print(f"Erreur: Le fichier de configuration {args.config} n'existe pas")
            print("Exécutez d'abord la commande 'init' pour créer une configuration")
            return
        
        # Chargement du pipeline depuis la configuration
        pipeline = RAGPipeline.load_from_config(args.config, gemini_api_key=args.api_key)
        
        # Vérifier si l'historique est activé
        if not pipeline.enable_query_history:
            print("L'historique des requêtes est désactivé dans la configuration.")
            return
        
        # Gestion de l'historique des requêtes
        if args.clear:
            result = pipeline.clear_query_history()
            if result:
                print("Historique des requêtes effacé avec succès.")
            else:
                print("Erreur lors de l'effacement de l'historique des requêtes.")
        
        elif args.list or args.filter:
            history = pipeline.get_query_history(limit=args.limit, filter_query=args.filter)
            
            if not history:
                print("Aucune entrée dans l'historique des requêtes.")
                return
                
            print("\n" + "="*80)
            print(f"Historique des requêtes ({len(history)} entrées):")
            print("="*80)
            
            for i, entry in enumerate(history):
                query = entry.get("query", "")
                timestamp = entry.get("timestamp", "")
                sources_count = entry.get("sources_count", 0)
                
                print(f"{i+1}. [{timestamp}] \"{query}\" ({sources_count} sources)")
                
                # Afficher un extrait de la réponse
                response = entry.get("response", "")
                if response:
                    # Tronquer la réponse si elle est trop longue
                    if len(response) > 200:
                        response = response[:197] + "..."
                    print(f"   Réponse: {response}")
                
                # Afficher les sources si disponibles
                sources = entry.get("sources", [])
                if sources and len(sources) > 0:
                    print("   Sources principales:")
                    for j, source in enumerate(sources[:2]):  # Afficher seulement les 2 premières sources
                        source_name = source.get("source", "")
                        author = source.get("author", "")
                        title = source.get("title", "")
                        score = source.get("score", 0)
                        
                        if title:
                            print(f"      - {title} ({author}, score: {score:.2f})")
                        else:
                            print(f"      - {source_name} (score: {score:.2f})")
                
                print()  # Ligne vide entre les entrées
        
        else:
            print("Aucune action spécifiée pour la gestion de l'historique.")
            print("Utilisez --list pour afficher l'historique ou --clear pour l'effacer.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
