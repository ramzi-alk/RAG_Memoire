"""
Script de test pour le système RAG.

Ce script permet de tester le système RAG avec des exemples
et d'optimiser ses performances.
"""

import os
import logging
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Optional

# Import du pipeline RAG
from src.pipeline.rag_pipeline import RAGPipeline

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_environment(base_dir: str = "./test_env") -> Dict[str, str]:
    """
    Crée un environnement de test pour le système RAG.
    
    Args:
        base_dir: Répertoire de base pour l'environnement de test
        
    Returns:
        Dictionnaire avec les chemins des répertoires créés
    """
    # Création des répertoires
    os.makedirs(base_dir, exist_ok=True)
    
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    pdf_dir = os.path.join(data_dir, "docs")
    os.makedirs(pdf_dir, exist_ok=True)
    
    db_dir = os.path.join(base_dir, "./chroma_db")
    os.makedirs(db_dir, exist_ok=True)
    
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    paths = {
        "base_dir": base_dir,
        "data_dir": data_dir,
        "pdf_dir": pdf_dir,
        "db_dir": db_dir,
        "results_dir": results_dir
    }
    
    logger.info(f"Environnement de test créé dans {base_dir}")
    return paths

def initialize_test_pipeline(
    paths: Dict[str, str],
    gemini_api_key: str,
    collection_name: str = "test_documents"
) -> RAGPipeline:
    """
    Initialise un pipeline RAG pour les tests.
    
    Args:
        paths: Dictionnaire avec les chemins des répertoires
        gemini_api_key: Clé API Gemini
        collection_name: Nom de la collection ChromaDB
        
    Returns:
        Instance de RAGPipeline configurée pour les tests
    """
    # Initialisation du pipeline
    pipeline = RAGPipeline(
        data_dir=paths["data_dir"],
        db_dir=paths["db_dir"],
        collection_name=collection_name,
        gemini_api_key=gemini_api_key,
        gemini_model="gemini-2.0-flash-001",
        remove_headers_footers=True,
        min_line_length=10,
        language="french"
    )
    
    # Sauvegarde de la configuration
    config_path = os.path.join(paths["base_dir"], "test_config.json")
    pipeline.save_config(config_path)
    
    logger.info(f"Pipeline de test initialisé et configuration sauvegardée dans {config_path}")
    return pipeline

def run_performance_tests(
    pipeline: RAGPipeline,
    test_queries: List[str],
    n_results_options: List[int] = [3, 5, 10],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Exécute des tests de performance sur le pipeline RAG.
    
    Args:
        pipeline: Instance de RAGPipeline à tester
        test_queries: Liste de requêtes de test
        n_results_options: Liste des nombres de résultats à tester
        output_file: Fichier de sortie pour les résultats (optionnel)
        
    Returns:
        Dictionnaire avec les résultats des tests
    """
    results = {
        "queries": test_queries,
        "n_results_options": n_results_options,
        "query_times": {},
        "response_lengths": {},
        "average_times": {}
    }
    
    # Test avec différents nombres de résultats
    for n_results in n_results_options:
        query_times = []
        response_lengths = []
        
        logger.info(f"Test avec n_results={n_results}")
        for query in tqdm(test_queries, desc=f"Requêtes (n_results={n_results})"):
            # Mesure du temps d'exécution
            start_time = time.time()
            result = pipeline.query(query_text=query, n_results=n_results)
            end_time = time.time()
            
            # Calcul du temps d'exécution
            query_time = end_time - start_time
            query_times.append(query_time)
            
            # Calcul de la longueur de la réponse
            response_length = len(result['response'])
            response_lengths.append(response_length)
            
            logger.info(f"Requête: '{query}', Temps: {query_time:.2f}s, Longueur: {response_length}")
        
        # Calcul des moyennes
        avg_time = sum(query_times) / len(query_times)
        avg_length = sum(response_lengths) / len(response_lengths)
        
        # Stockage des résultats
        results["query_times"][n_results] = query_times
        results["response_lengths"][n_results] = response_lengths
        results["average_times"][n_results] = avg_time
        
        logger.info(f"Temps moyen pour n_results={n_results}: {avg_time:.2f}s")
        logger.info(f"Longueur moyenne des réponses pour n_results={n_results}: {avg_length:.2f} caractères")
    
    # Sauvegarde des résultats si un fichier est spécifié
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Résultats des tests sauvegardés dans {output_file}")
    
    return results

def plot_performance_results(
    results: Dict[str, Any],
    output_file: Optional[str] = None
) -> None:
    """
    Génère des graphiques à partir des résultats des tests de performance.
    
    Args:
        results: Dictionnaire avec les résultats des tests
        output_file: Fichier de sortie pour le graphique (optionnel)
    """
    n_results_options = results["n_results_options"]
    avg_times = [results["average_times"][n] for n in n_results_options]
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    
    # Graphique des temps moyens
    plt.subplot(1, 2, 1)
    plt.bar(range(len(n_results_options)), avg_times, tick_label=[str(n) for n in n_results_options])
    plt.xlabel("Nombre de résultats (n_results)")
    plt.ylabel("Temps moyen (secondes)")
    plt.title("Temps moyen d'exécution des requêtes")
    
    # Graphique des temps par requête
    plt.subplot(1, 2, 2)
    for i, n in enumerate(n_results_options):
        plt.boxplot(results["query_times"][n], positions=[i+1])
    
    plt.xlabel("Nombre de résultats (n_results)")
    plt.ylabel("Temps d'exécution (secondes)")
    plt.title("Distribution des temps d'exécution")
    plt.xticks(range(1, len(n_results_options) + 1), [str(n) for n in n_results_options])
    
    plt.tight_layout()
    
    # Sauvegarde du graphique si un fichier est spécifié
    if output_file:
        plt.savefig(output_file)
        logger.info(f"Graphique sauvegardé dans {output_file}")
    
    plt.show()

def optimize_chunk_parameters(
    pipeline: RAGPipeline,
    pdf_path: str,
    test_queries: List[str],
    chunk_sizes: List[int] = [500, 1000, 1500],
    chunk_overlaps: List[int] = [100, 200, 300],
    output_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimise les paramètres de découpage en chunks.
    
    Args:
        pipeline: Instance de RAGPipeline à tester
        pdf_path: Chemin vers un fichier PDF de test
        test_queries: Liste de requêtes de test
        chunk_sizes: Liste des tailles de chunks à tester
        chunk_overlaps: Liste des chevauchements à tester
        output_file: Fichier de sortie pour les résultats (optionnel)
        
    Returns:
        Dictionnaire avec les résultats des tests
    """
    results = {
        "pdf_path": pdf_path,
        "test_queries": test_queries,
        "chunk_sizes": chunk_sizes,
        "chunk_overlaps": chunk_overlaps,
        "results": {}
    }
    
    # Test avec différentes combinaisons de paramètres
    for chunk_size in chunk_sizes:
        for chunk_overlap in chunk_overlaps:
            # Vérification que le chevauchement est inférieur à la taille du chunk
            if chunk_overlap >= chunk_size:
                logger.warning(f"Ignoré: chunk_size={chunk_size}, chunk_overlap={chunk_overlap} (chevauchement >= taille)")
                continue
            
            logger.info(f"Test avec chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
            
            # Réinitialisation de la collection ChromaDB
            try:
                pipeline.db_manager.delete_collection()
            except:
                pass
            
            # Traitement du PDF avec les paramètres actuels
            stats = pipeline.process_single_pdf(
                pdf_path=pdf_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Exécution des requêtes de test
            query_results = []
            for query in test_queries:
                result = pipeline.query(query_text=query, n_results=10)
                query_results.append({
                    "query": query,
                    "response": result["response"],
                    "sources_count": len(result["sources"])
                })
            
            # Stockage des résultats
            results["results"][f"{chunk_size}_{chunk_overlap}"] = {
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_count": stats["chunk_count"],
                "query_results": query_results
            }
    
    # Sauvegarde des résultats si un fichier est spécifié
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Résultats des tests sauvegardés dans {output_file}")
    
    return results

def main():
    """Fonction principale pour exécuter les tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Tests et optimisation du système RAG")
    parser.add_argument("--api-key", required=True, help="Clé API Gemini")
    parser.add_argument("--pdf-dir", help="Répertoire contenant les fichiers PDF de test")
    parser.add_argument("--pdf-file", help="Fichier PDF individuel pour les tests")
    parser.add_argument("--test-dir", default="./test_env", help="Répertoire pour l'environnement de test")
    parser.add_argument("--test-type", choices=["performance", "chunk_optimization", "all"], default="all", help="Type de test à exécuter")
    
    args = parser.parse_args()
    
    # Création de l'environnement de test
    paths = create_test_environment(args.test_dir)
    
    # Initialisation du pipeline
    pipeline = initialize_test_pipeline(paths, args.api_key)
    
    # Traitement des PDF si spécifiés
    if args.pdf_dir:
        if not os.path.exists(args.pdf_dir):
            logger.error(f"Le répertoire {args.pdf_dir} n'existe pas")
            return
        
        pipeline.process_pdf_directory(
            pdf_dir=args.pdf_dir,
            recursive=True,
            chunk_size=1000,
            chunk_overlap=200
        )
    elif args.pdf_file:
        if not os.path.exists(args.pdf_file):
            logger.error(f"Le fichier {args.pdf_file} n'existe pas")
            return
        
        pipeline.process_single_pdf(
            pdf_path=args.pdf_file,
            chunk_size=1000,
            chunk_overlap=200
        )
    
    # Requêtes de test
    test_queries = [
        "Qu'est-ce qu'un système RAG?",
        "Comment fonctionne l'extraction de texte à partir de PDF?",
        "Quels sont les avantages des embeddings TF-IDF?",
        "Comment ChromaDB stocke-t-il les vecteurs?",
        "Expliquez le fonctionnement de l'API Gemini."
    ]
    
    # Exécution des tests selon le type spécifié
    if args.test_type in ["performance", "all"]:
        logger.info("Exécution des tests de performance...")
        performance_results = run_performance_tests(
            pipeline=pipeline,
            test_queries=test_queries,
            n_results_options=[3, 5, 10],
            output_file=os.path.join(paths["results_dir"], "performance_results.json")
        )
        
        # Génération du graphique
        try:
            plot_performance_results(
                results=performance_results,
                output_file=os.path.join(paths["results_dir"], "performance_graph.png")
            )
        except Exception as e:
            logger.error(f"Erreur lors de la génération du graphique: {str(e)}")
    
    if args.test_type in ["chunk_optimization", "all"] and args.pdf_file:
        logger.info("Exécution des tests d'optimisation des chunks...")
        chunk_results = optimize_chunk_parameters(
            pipeline=pipeline,
            pdf_path=args.pdf_file,
            test_queries=test_queries,
            chunk_sizes=[500, 1000, 1500],
            chunk_overlaps=[100, 200, 300],
            output_file=os.path.join(paths["results_dir"], "chunk_optimization_results.json")
        )
    
    logger.info("Tests terminés")

if __name__ == "__main__":
    main()
