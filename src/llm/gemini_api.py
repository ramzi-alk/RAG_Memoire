"""
Module d'intégration de l'API Gemini pour la génération de réponses.

Ce module fournit des fonctions pour configurer l'authentification avec l'API Gemini
et générer des réponses basées sur des contextes récupérés.
"""

import os
import logging
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Union, Tuple
import json
from dotenv import load_dotenv

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GeminiAPI:
    """Classe pour interagir avec l'API Gemini de Google."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model_name: str = "gemini-2.0-flash-001",
                 temperature: float = 0.2,
                 top_p: float = 0.95,
                 top_k: int = 40,
                 max_output_tokens: int = 5048):
        """
        Initialise l'API Gemini avec les paramètres spécifiés.
        
        Args:
            api_key: Clé API Gemini (si None, cherche dans les variables d'environnement)
            model_name: Nom du modèle Gemini à utiliser
            temperature: Température pour la génération (0.0 à 1.0)
            top_p: Paramètre top_p pour la génération
            top_k: Paramètre top_k pour la génération
            max_output_tokens: Nombre maximum de tokens dans la réponse
        """
        # Chargement des variables d'environnement si nécessaire
        if api_key is None:
            load_dotenv()
            api_key = os.getenv("GEMINI_API_KEY")
            
            if api_key is None:
                raise ValueError("Aucune clé API Gemini fournie et GEMINI_API_KEY non trouvée dans les variables d'environnement")
        
        # Configuration de l'API Gemini
        genai.configure(api_key=api_key)
        
        # Paramètres de génération
        self.model_name = model_name
        self.generation_config = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_output_tokens": max_output_tokens,
        }
        
        # Initialisation du modèle
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=self.generation_config
        )
        
        logger.info(f"API Gemini initialisée avec le modèle {model_name}")
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[str] = None,
                         system_instruction: Optional[str] = None) -> str:
        """
        Génère une réponse à partir d'un prompt et d'un contexte optionnel.
        
        Args:
            prompt: Question ou instruction pour le modèle
            context: Contexte supplémentaire pour la génération (documents récupérés)
            system_instruction: Instruction système pour guider le comportement du modèle
            
        Returns:
            Réponse générée par le modèle
        """
        try:
            # Construction du prompt complet avec contexte si fourni
            full_prompt = prompt
            if context:
                full_prompt = f"Contexte:\n{context}\n\nQuestion: {prompt}\n\nRéponse:"
            
            # Création de la conversation avec instruction système si fournie
            if system_instruction:
                
                chat = self.model.start_chat()
                response = chat.send_message(full_prompt)
            else:
                response = self.model.generate_content(full_prompt)
            
            # Extraction du texte de la réponse
            response_text = response.text
            
            logger.info(f"Réponse générée pour le prompt: '{prompt[:50]}...' (tronqué)")
            return response_text
            
        except Exception as e:
            error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
            logger.error(error_msg)
            return f"Désolé, une erreur s'est produite lors de la génération de la réponse: {str(e)}"
    
    def generate_rag_response(self,
                             query: str,
                             retrieved_documents: List[str],
                             document_scores: Optional[List[float]] = None,
                             document_metadatas: Optional[List[Dict[str, Any]]] = None,
                             system_instruction: Optional[str] = None) -> Dict[str, Any]:
        """
        Génère une réponse RAG basée sur les documents récupérés.
        
        Args:
            query: Question de l'utilisateur
            retrieved_documents: Liste des documents récupérés
            document_scores: Scores de similarité des documents (optionnel)
            document_metadatas: Métadonnées des documents (optionnel)
            system_instruction: Instruction système pour guider le comportement du modèle
            
        Returns:
            Dictionnaire contenant la réponse générée et des informations sur les sources
        """
        # Préparation du contexte à partir des documents récupérés
        context_parts = []
        
        for i, doc in enumerate(retrieved_documents):
            # Ajout des métadonnées si disponibles
            metadata_str = ""
            if document_metadatas and i < len(document_metadatas):
                metadata = document_metadatas[i]
                metadata_str = f" [Métadonnées: {json.dumps(metadata, ensure_ascii=False)}]"
            
            # Ajout du score si disponible
            score_str = ""
            if document_scores and i < len(document_scores):
                score = document_scores[i]
                score_str = f" [Score: {score:.4f}]"
            
            # Formatage du document avec ses informations
            context_parts.append(f"Document {i+1}{metadata_str}{score_str}:\n{doc}\n")
        
        # Assemblage du contexte complet
        context = "\n".join(context_parts)
        
        # Instruction système par défaut pour le RAG si non fournie
        if system_instruction is None:
            system_instruction = """
            Tu es un assistant de recherche académique expert qui répond aux questions en t'appuyant exclusivement sur les documents fournis.
            
            INSTRUCTIONS PRINCIPALES:
            1. BASE TES RÉPONSES UNIQUEMENT sur les informations présentes dans les documents fournis.
            2. Si l'information n'est pas disponible dans les documents, indique clairement: "Les documents fournis ne contiennent pas suffisamment d'informations pour répondre à cette question." puis suggère des pistes de recherche alternatives.
            3. Sois précis et concis. Organise ta réponse en paragraphes thématiques courts pour faciliter la lecture.
            4. Évite toute spéculation ou opinion personnelle. Reste factuel et objectif.
            5. Réponds uniquement en français.
            
            CITATIONS ET SOURCES:
            1. Cite systématiquement tes sources après chaque affirmation importante sous forme parenthésée:
               - Si l'auteur est disponible dans le titre du document : (Auteur, Document N)
               - Sans auteur: (Document N)
            2. Utilise les sources à score de similarité élevé en priorité.
            3. Inclus une section "Sources" à la fin de ta réponse avec ce format:
               
               Sources:
               - Document 1: [Titre complet ou début du document] (Score: XX%)
               - Document 2: [Titre complet ou début du document] (Score: XX%)
            
            STRUCTURE DE RÉPONSE:
            1. Commence par une synthèse directe répondant à la question (1-2 phrases).
            2. Développe ensuite les aspects clés en paragraphes distincts.
            3. Si pertinent, présente différentes perspectives ou approches trouvées dans les documents.
            4. Conclus par une synthèse des points essentiels.
            5. Liste les sources utilisées.
            
            CONSIDÉRATIONS IMPORTANTES:
            1. Respecte la nuance et la complexité des documents académiques.
            2. Préserve les définitions techniques précises.
            3. Si les documents contiennent des informations contradictoires, présente les différentes positions en citant leurs sources respectives.
            4. Pour les données quantitatives, cite les chiffres exacts avec leurs sources.
            """
        
        # Génération de la réponse
        response_text = self.generate_response(query, context, system_instruction)
        
        # Construction du résultat avec sources
        sources = []
        for i, doc in enumerate(retrieved_documents):
            source_item = {
                "document": doc,
                "score": document_scores[i] if document_scores and i < len(document_scores) else None,
                "metadata": document_metadatas[i] if document_metadatas and i < len(document_metadatas) else {}
            }
            sources.append(source_item)
        
        result = {
            "query": query,
            "response": response_text,
            "sources": sources
        }
        
        logger.info(f"Réponse RAG générée pour la requête: '{query}'")
        return result
    
    def save_api_key_to_env_file(self, api_key: str, env_file_path: str = ".env") -> None:
        """
        Sauvegarde la clé API Gemini dans un fichier .env.
        
        Args:
            api_key: Clé API à sauvegarder
            env_file_path: Chemin du fichier .env
        """
        try:
            # Vérification si le fichier existe déjà
            if os.path.exists(env_file_path):
                # Lecture du fichier existant
                with open(env_file_path, 'r') as f:
                    lines = f.readlines()
                
                # Vérification si la variable existe déjà
                key_exists = False
                for i, line in enumerate(lines):
                    if line.startswith("GEMINI_API_KEY="):
                        lines[i] = f"GEMINI_API_KEY={api_key}\n"
                        key_exists = True
                        break
                
                # Ajout de la variable si elle n'existe pas
                if not key_exists:
                    lines.append(f"GEMINI_API_KEY={api_key}\n")
                
                # Écriture du fichier mis à jour
                with open(env_file_path, 'w') as f:
                    f.writelines(lines)
            else:
                # Création d'un nouveau fichier
                with open(env_file_path, 'w') as f:
                    f.write(f"GEMINI_API_KEY={api_key}\n")
            
            logger.info(f"Clé API Gemini sauvegardée dans {env_file_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la clé API: {str(e)}")
            raise


# Exemple d'utilisation
if __name__ == "__main__":
    # Cet exemple montre comment utiliser la classe GeminiAPI
    
    # Clé API (à remplacer par votre propre clé)
    api_key = "votre_clé_api_gemini"
    
    try:
        # Initialisation de l'API Gemini
        gemini_api = GeminiAPI(api_key=api_key)
        
        # Exemple de génération simple
        prompt = "Qu'est-ce qu'un système RAG?"
        response = gemini_api.generate_response(prompt)
        print(f"Réponse à '{prompt}':\n{response}\n")
        
        # Exemple de génération RAG
        query = "Comment fonctionne l'extraction de texte dans un système RAG?"
        retrieved_documents = [
            "L'extraction de texte est une étape cruciale dans un système RAG. Elle consiste à extraire le contenu textuel des documents sources, comme les PDF, les pages web ou les documents Word.",
            "PyMuPDF est une bibliothèque Python populaire pour extraire du texte à partir de fichiers PDF. Elle permet d'accéder au contenu page par page et offre des options pour le prétraitement."
        ]
        document_scores = [0.95, 0.87]
        document_metadatas = [
            {"source": "article_rag", "author": "John Doe"},
            {"source": "documentation_pymupdf", "author": "PyMuPDF Team"}
        ]
        
        rag_response = gemini_api.generate_rag_response(
            query=query,
            retrieved_documents=retrieved_documents,
            document_scores=document_scores,
            document_metadatas=document_metadatas
        )
        
        print(f"Réponse RAG à '{query}':\n{rag_response['response']}\n")
        print("Sources utilisées:")
        for i, source in enumerate(rag_response['sources']):
            print(f"Document {i+1} (Score: {source['score']}):")
            print(f"  {source['document'][:100]}... (tronqué)")
            print(f"  Métadonnées: {source['metadata']}")
            print()
            
    except ValueError as e:
        print(f"Erreur: {str(e)}")
        print("Assurez-vous de fournir une clé API valide ou de définir la variable d'environnement GEMINI_API_KEY.")
