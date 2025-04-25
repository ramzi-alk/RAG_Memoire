"""
Module d'extraction de texte à partir de fichiers PDF en utilisant PyMuPDF.

Ce module fournit des fonctions pour extraire le texte de fichiers PDF individuels
ou par lots, avec des options pour le prétraitement du texte.
"""

import os
import fitz  # PyMuPDF
import logging
import re
import json
import hashlib
from typing import List, Dict, Optional, Union, Tuple, Set
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
from functools import partial
import time

try:
    import langdetect
    from langdetect import detect
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mapping des codes de langue pour l'OCR et d'autres traitements
LANGUAGE_MAPPING = {
    # Langues principales
    "en": {"ocr": "eng", "name": "English"},
    "fr": {"ocr": "fra", "name": "French"},
    "es": {"ocr": "spa", "name": "Spanish"},
    "de": {"ocr": "deu", "name": "German"},
    "it": {"ocr": "ita", "name": "Italian"},
    "pt": {"ocr": "por", "name": "Portuguese"},
    "nl": {"ocr": "nld", "name": "Dutch"},
    "ru": {"ocr": "rus", "name": "Russian"},
    "zh": {"ocr": "chi_sim", "name": "Chinese Simplified"},
    "ja": {"ocr": "jpn", "name": "Japanese"},
    "ar": {"ocr": "ara", "name": "Arabic"},
    
    # Autres langues
    "cs": {"ocr": "ces", "name": "Czech"},
    "pl": {"ocr": "pol", "name": "Polish"},
    "sv": {"ocr": "swe", "name": "Swedish"},
    "fi": {"ocr": "fin", "name": "Finnish"},
    "da": {"ocr": "dan", "name": "Danish"},
    "no": {"ocr": "nor", "name": "Norwegian"},
    "ko": {"ocr": "kor", "name": "Korean"},
    "el": {"ocr": "ell", "name": "Greek"},
    "tr": {"ocr": "tur", "name": "Turkish"},
    "hu": {"ocr": "hun", "name": "Hungarian"},
    "ro": {"ocr": "ron", "name": "Romanian"},
}

def detect_language(text: str, default: str = "en") -> str:
    """
    Détecte la langue d'un texte.
    
    Args:
        text: Texte à analyser
        default: Code de langue par défaut si la détection échoue
        
    Returns:
        Code de langue ISO 639-1 (2 lettres)
    """
    if not HAS_LANGDETECT or not text:
        return default
    
    try:
        # Utiliser un échantillon du texte pour la détection
        sample = text[:min(len(text), 5000)]
        lang_code = detect(sample)
        return lang_code
    except Exception as e:
        logger.warning(f"Erreur lors de la détection de langue: {str(e)}")
        return default

def get_ocr_language_code(lang_code: str) -> str:
    """
    Convertit un code de langue ISO 639-1 en code de langue pour Tesseract OCR.
    
    Args:
        lang_code: Code de langue ISO 639-1 (2 lettres)
        
    Returns:
        Code de langue pour Tesseract OCR
    """
    if lang_code in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[lang_code]["ocr"]
    else:
        # Par défaut, retourner le code tel quel
        return lang_code

def get_language_name(lang_code: str) -> str:
    """
    Récupère le nom complet d'une langue à partir de son code.
    
    Args:
        lang_code: Code de langue ISO 639-1 (2 lettres)
        
    Returns:
        Nom de la langue en anglais
    """
    if lang_code in LANGUAGE_MAPPING:
        return LANGUAGE_MAPPING[lang_code]["name"]
    else:
        return lang_code.upper()

def _process_pdf_file(extractor, pdf_file):
    """
    Fonction auxiliaire pour le traitement parallèle des PDF.
    
    Args:
        extractor: Instance de PDFExtractor
        pdf_file: Chemin vers le fichier PDF à traiter
        
    Returns:
        Tuple (chemin du fichier, résultat de l'extraction)
    """
    try:
        return pdf_file, extractor.extract_text_from_file(pdf_file)
    except Exception as e:
        logging.warning(f"Échec de l'extraction pour {pdf_file}: {str(e)}")
        return pdf_file, {"error": str(e)}

class PDFExtractor:
    """Classe pour extraire le texte des fichiers PDF."""
    
    def __init__(self, remove_headers_footers: bool = False, min_line_length: int = 0, cache_dir: str = None):
        """
        Initialise l'extracteur PDF avec des options de configuration.
        
        Args:
            remove_headers_footers: Si True, tente de supprimer les en-têtes et pieds de page
            min_line_length: Longueur minimale des lignes à conserver (pour filtrer le bruit)
            cache_dir: Répertoire pour stocker le cache des extractions (None pour désactiver)
        """
        self.remove_headers_footers = remove_headers_footers
        self.min_line_length = min_line_length
        self.cache_dir = cache_dir
        
        # Créer le répertoire de cache s'il est spécifié
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            logger.info(f"Cache d'extraction activé dans le répertoire: {cache_dir}")
            
        # Vérifier si langdetect est disponible
        if not HAS_LANGDETECT:
            logger.warning("Le module 'langdetect' n'est pas installé. La détection automatique de langue ne sera pas disponible.")
            logger.warning("Installez-le avec 'pip install langdetect' pour activer cette fonctionnalité.")
            
        logger.info("Extracteur PDF initialisé avec remove_headers_footers=%s, min_line_length=%d, cache_dir=%s", 
                   remove_headers_footers, min_line_length, cache_dir)
    
    def _get_cache_path(self, pdf_path: str) -> Optional[str]:
        """
        Obtient le chemin du fichier de cache pour un PDF donné.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Chemin du fichier de cache ou None si le cache est désactivé
        """
        if not self.cache_dir:
            return None
            
        # Créer un hash du chemin absolu et des paramètres d'extraction
        pdf_abs_path = os.path.abspath(pdf_path)
        file_hash = hashlib.md5(f"{pdf_abs_path}_{self.remove_headers_footers}_{self.min_line_length}".encode()).hexdigest()
        
        # Utiliser le hash comme nom de fichier
        return os.path.join(self.cache_dir, f"{file_hash}.json")
    
    def _save_to_cache(self, pdf_path: str, extracted_text: Dict[int, str]) -> None:
        """
        Sauvegarde le texte extrait dans le cache.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            extracted_text: Dictionnaire du texte extrait par page
        """
        cache_path = self._get_cache_path(pdf_path)
        if not cache_path:
            return
            
        try:
            # Préparer les données à mettre en cache
            cache_data = {
                "pdf_path": pdf_path,
                "modification_time": os.path.getmtime(pdf_path),
                "extraction_params": {
                    "remove_headers_footers": self.remove_headers_footers,
                    "min_line_length": self.min_line_length
                },
                "extracted_text": {str(k): v for k, v in extracted_text.items()}  # Convertir les clés int en str pour JSON
            }
            
            # Sauvegarder dans le cache
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.debug(f"Extraction mise en cache pour: {pdf_path}")
        except Exception as e:
            logger.warning(f"Impossible de sauvegarder l'extraction dans le cache pour {pdf_path}: {str(e)}")
    
    def _load_from_cache(self, pdf_path: str) -> Optional[Dict[int, str]]:
        """
        Charge le texte extrait depuis le cache si disponible et valide.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire du texte extrait par page ou None si pas de cache valide
        """
        cache_path = self._get_cache_path(pdf_path)
        if not cache_path or not os.path.exists(cache_path):
            return None
            
        try:
            # Vérifier si le fichier PDF a été modifié depuis la mise en cache
            pdf_mtime = os.path.getmtime(pdf_path)
            
            with open(cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Vérifier si le cache est valide
            if (cache_data["pdf_path"] == pdf_path and
                cache_data["modification_time"] == pdf_mtime and
                cache_data["extraction_params"]["remove_headers_footers"] == self.remove_headers_footers and
                cache_data["extraction_params"]["min_line_length"] == self.min_line_length):
                
                # Convertir les clés str en int
                extracted_text = {int(k): v for k, v in cache_data["extracted_text"].items()}
                logger.info(f"Utilisation de l'extraction en cache pour: {pdf_path}")
                return extracted_text
                
        except Exception as e:
            logger.warning(f"Impossible de charger l'extraction depuis le cache pour {pdf_path}: {str(e)}")
            
        return None
    
    def extract_text_from_file(self, pdf_path: str) -> Dict[int, str]:
        """
        Extrait le texte d'un fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire avec les numéros de page comme clés et le texte comme valeurs
            
        Raises:
            FileNotFoundError: Si le fichier PDF n'existe pas
            ValueError: Si le fichier n'est pas un PDF valide
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
        
        # Vérifier si l'extraction est dans le cache
        cached_extraction = self._load_from_cache(pdf_path)
        if cached_extraction is not None:
            return cached_extraction
        
        try:
            doc = fitz.open(pdf_path)
            result = {}
            
            # Pré-analyse pour détecter les motifs d'en-têtes et pieds de page
            headers_footers = {}
            if self.remove_headers_footers:
                headers_footers = self._detect_headers_footers(doc)
            
            # Extraction et traitement du texte
            for page_num, page in enumerate(doc):
                text = page.get_text()
                
                # Prétraitement du texte si nécessaire
                if self.remove_headers_footers:
                    text = self._remove_headers_footers_improved(
                        text, page_num, doc.page_count, headers_footers
                    )
                
                if self.min_line_length > 0:
                    text = self._filter_short_lines(text)
                
                result[page_num] = text
            
            doc.close()
            logger.info(f"Extraction réussie du fichier: {pdf_path}, {len(result)} pages extraites")
            
            # Mettre l'extraction en cache
            self._save_to_cache(pdf_path, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du texte de {pdf_path}: {str(e)}")
            raise ValueError(f"Erreur lors de l'extraction du texte: {str(e)}")
    
    def extract_text_from_directory(self, 
                                     directory_path: str, 
                                     recursive: bool = False,
                                     file_extension: str = ".pdf",
                                     max_workers: int = None) -> Dict[str, Dict[int, str]]:
        """
        Extrait le texte de tous les fichiers PDF dans un répertoire.
        
        Args:
            directory_path: Chemin vers le répertoire contenant les fichiers PDF
            recursive: Si True, parcourt également les sous-répertoires
            file_extension: Extension des fichiers à traiter (par défaut .pdf)
            max_workers: Nombre maximum de workers pour le traitement parallèle
                        (None pour utiliser le nombre de processeurs)
            
        Returns:
            Dictionnaire avec les noms de fichiers comme clés et les contenus extraits comme valeurs
            
        Raises:
            FileNotFoundError: Si le répertoire n'existe pas
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Le répertoire n'existe pas: {directory_path}")
        
        pdf_files = []
        
        if recursive:
            for root, _, files in os.walk(directory_path):
                for file in files:
                    if file.lower().endswith(file_extension.lower()):
                        pdf_files.append(os.path.join(root, file))
        else:
            pdf_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path) 
                        if f.lower().endswith(file_extension.lower())]
        
        results = {}
        
        # Déterminer s'il faut utiliser le traitement séquentiel ou parallèle
        if max_workers == 1 or max_workers == 0 or len(pdf_files) < 2:
            # Traitement séquentiel
            for pdf_file in tqdm(pdf_files, desc="Extraction des fichiers PDF"):
                try:
                    results[pdf_file] = self.extract_text_from_file(pdf_file)
                except Exception as e:
                    logger.warning(f"Échec de l'extraction pour {pdf_file}: {str(e)}")
                    results[pdf_file] = {"error": str(e)}
        else:
            # Utiliser ThreadPoolExecutor plutôt que ProcessPoolExecutor
            # Cela évite les problèmes de pickling
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Fonction de traitement locale
                    def process_file(pdf_file):
                        try:
                            return pdf_file, self.extract_text_from_file(pdf_file)
                        except Exception as e:
                            logger.warning(f"Échec de l'extraction pour {pdf_file}: {str(e)}")
                            return pdf_file, {"error": str(e)}
                    
                    # Soumettre les tâches
                    futures = [executor.submit(process_file, pdf_file) for pdf_file in pdf_files]
                    
                    # Afficher une barre de progression
                    for future in tqdm(concurrent.futures.as_completed(futures), 
                                     total=len(futures), 
                                     desc="Extraction des fichiers PDF"):
                        try:
                            pdf_file, result = future.result()
                            results[pdf_file] = result
                        except Exception as e:
                            logger.error(f"Erreur non gérée pendant l'extraction parallèle: {str(e)}")
            except Exception as e:
                # En cas d'erreur avec le traitement parallèle, revenir au traitement séquentiel
                logger.warning(f"Échec du traitement parallèle: {str(e)}. Utilisation du traitement séquentiel.")
                results = {}
                for pdf_file in tqdm(pdf_files, desc="Extraction des fichiers PDF (séquentiel)"):
                    try:
                        results[pdf_file] = self.extract_text_from_file(pdf_file)
                    except Exception as e:
                        logger.warning(f"Échec de l'extraction pour {pdf_file}: {str(e)}")
                        results[pdf_file] = {"error": str(e)}
        
        logger.info(f"Extraction terminée pour {len(pdf_files)} fichiers dans {directory_path}")
        return results
    
    def extract_and_save_as_txt(self, 
                               pdf_path: str, 
                               output_dir: str,
                               filename_prefix: str = "",
                               one_file_per_page: bool = False) -> List[str]:
        """
        Extrait le texte d'un PDF et le sauvegarde dans un ou plusieurs fichiers texte.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Répertoire de sortie pour les fichiers texte
            filename_prefix: Préfixe pour les noms de fichiers de sortie
            one_file_per_page: Si True, crée un fichier par page, sinon un seul fichier
            
        Returns:
            Liste des chemins des fichiers créés
            
        Raises:
            FileNotFoundError: Si le fichier PDF ou le répertoire de sortie n'existe pas
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire de sortie créé: {output_dir}")
        
        extracted_text = self.extract_text_from_file(pdf_path)
        created_files = []
        
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        if filename_prefix:
            base_filename = f"{filename_prefix}_{base_filename}"
        
        if one_file_per_page:
            for page_num, text in extracted_text.items():
                output_file = os.path.join(output_dir, f"{base_filename}_page{page_num+1}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                created_files.append(output_file)
        else:
            output_file = os.path.join(output_dir, f"{base_filename}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                for page_num in sorted(extracted_text.keys()):
                    f.write(f"--- Page {page_num+1} ---\n")
                    f.write(extracted_text[page_num])
                    f.write("\n\n")
            created_files.append(output_file)
        
        logger.info(f"Texte extrait et sauvegardé dans {len(created_files)} fichier(s)")
        return created_files
    
    def _detect_headers_footers(self, doc) -> Dict[str, Set[str]]:
        """
        Détecte les motifs récurrents d'en-têtes et de pieds de page dans un document.
        
        Args:
            doc: Document PyMuPDF ouvert
            
        Returns:
            Dictionnaire contenant les ensembles de motifs d'en-têtes et de pieds de page
        """
        # Limite d'échantillonnage pour les documents très longs
        max_pages_to_sample = min(100, doc.page_count)
        sample_step = max(1, doc.page_count // max_pages_to_sample)
        
        # Collecte des premiers et derniers éléments de chaque page
        headers = []
        footers = []
        
        for page_idx in range(0, doc.page_count, sample_step):
            page = doc[page_idx]
            text = page.get_text()
            lines = text.split('\n')
            
            if not lines:
                continue
                
            # Collecter les 2 premières et 2 dernières lignes (si disponibles)
            header_lines = lines[:min(2, len(lines))]
            footer_lines = lines[-min(2, len(lines)):]
            
            for line in header_lines:
                line = line.strip()
                if line and len(line) < 100:  # Ignorer les lignes trop longues
                    headers.append(line)
            
            for line in footer_lines:
                line = line.strip()
                if line and len(line) < 100:  # Ignorer les lignes trop longues
                    footers.append(line)
        
        # Identifier les motifs récurrents
        frequent_headers = self._identify_recurring_patterns(headers)
        frequent_footers = self._identify_recurring_patterns(footers)
        
        return {
            "headers": frequent_headers,
            "footers": frequent_footers
        }
    
    def _identify_recurring_patterns(self, lines: List[str]) -> Set[str]:
        """
        Identifie les motifs récurrents dans un ensemble de lignes.
        
        Args:
            lines: Liste de lignes à analyser
            
        Returns:
            Ensemble des motifs récurrents identifiés
        """
        if not lines or len(lines) < 3:  # Besoin d'au moins 3 lignes pour détecter un motif
            return set()
        
        # Compter les occurrences de chaque ligne (après normalisation)
        pattern_counts = {}
        
        for line in lines:
            # Normaliser en remplaçant les numéros par '#'
            normalized = re.sub(r'\d+', '#', line)
            pattern_counts[normalized] = pattern_counts.get(normalized, 0) + 1
        
        # Identifier les motifs qui apparaissent dans au moins 30% des pages échantillonnées
        threshold = max(2, len(lines) * 0.3 // 2)  # Division par 2 car nous avons collecté 2 lignes par page
        frequent_patterns = {pattern for pattern, count in pattern_counts.items() 
                           if count >= threshold and len(pattern) > 0}
        
        return frequent_patterns
    
    def _remove_headers_footers(self, text: str, page_num: int, total_pages: int) -> str:
        """
        Tente de supprimer les en-têtes et pieds de page du texte extrait.
        
        Cette méthode utilise une heuristique simple: elle supprime la première et 
        la dernière ligne de chaque page si elles sont courtes ou contiennent des 
        numéros de page.
        
        Args:
            text: Texte extrait de la page
            page_num: Numéro de la page actuelle
            total_pages: Nombre total de pages dans le document
            
        Returns:
            Texte avec en-têtes et pieds de page potentiellement supprimés
        """
        lines = text.split('\n')
        if not lines:
            return text
        
        # Vérifier si la première ligne ressemble à un en-tête
        if len(lines[0].strip()) < 50 or str(page_num + 1) in lines[0]:
            lines = lines[1:]
        
        # Vérifier si la dernière ligne ressemble à un pied de page
        if lines and (len(lines[-1].strip()) < 50 or str(page_num + 1) in lines[-1]):
            lines = lines[:-1]
        
        return '\n'.join(lines)
    
    def _remove_headers_footers_improved(self, text: str, page_num: int, total_pages: int, 
                                       headers_footers: Dict[str, Set[str]]) -> str:
        """
        Version améliorée de la suppression des en-têtes et pieds de page.
        
        Args:
            text: Texte extrait de la page
            page_num: Numéro de la page actuelle
            total_pages: Nombre total de pages dans le document
            headers_footers: Dictionnaire contenant les motifs d'en-têtes et pieds de page
            
        Returns:
            Texte avec en-têtes et pieds de page supprimés
        """
        lines = text.split('\n')
        if not lines:
            return text
        
        # Si aucun modèle d'en-tête/pied de page n'a été détecté, utiliser la méthode de base
        if not headers_footers or not (headers_footers.get("headers") or headers_footers.get("footers")):
            return self._remove_headers_footers(text, page_num, total_pages)
        
        header_patterns = headers_footers.get("headers", set())
        footer_patterns = headers_footers.get("footers", set())
        
        result_lines = []
        for i, line in enumerate(lines):
            # Si c'est l'une des 3 premières lignes, vérifier s'il s'agit d'un en-tête
            if i < 3:
                # Normaliser la ligne pour la comparaison
                normalized = re.sub(r'\d+', '#', line.strip())
                
                # Vérifier si la ligne correspond à un motif d'en-tête connu
                if normalized in header_patterns:
                    continue
                
                # Vérifier si la ligne contient un numéro de page
                if re.search(r'\b' + str(page_num + 1) + r'\b', line) or re.search(r'page\s*\d+', line.lower()):
                    continue
                
                # Autres heuristiques pour les en-têtes (lignes courtes sans contenu significatif)
                if len(line.strip()) < 50 and not re.search(r'[a-zA-Z]{5,}', line) and i == 0:
                    continue
            
            # Si c'est l'une des 3 dernières lignes, vérifier s'il s'agit d'un pied de page
            elif i >= len(lines) - 3:
                # Normaliser la ligne pour la comparaison
                normalized = re.sub(r'\d+', '#', line.strip())
                
                # Vérifier si la ligne correspond à un motif de pied de page connu
                if normalized in footer_patterns:
                    continue
                
                # Vérifier si la ligne contient un numéro de page
                if re.search(r'\b' + str(page_num + 1) + r'\b', line) or re.search(r'page\s*\d+', line.lower()):
                    continue
                
                # Autres heuristiques pour les pieds de page
                if len(line.strip()) < 50 and not re.search(r'[a-zA-Z]{5,}', line) and i == len(lines) - 1:
                    continue
            
            # Conserver la ligne si elle n'est pas identifiée comme en-tête ou pied de page
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _filter_short_lines(self, text: str) -> str:
        """
        Filtre les lignes trop courtes qui sont souvent du bruit.
        
        Args:
            text: Texte à filtrer
            
        Returns:
            Texte avec les lignes courtes supprimées
        """
        if self.min_line_length <= 0:
            return text
            
        lines = text.split('\n')
        filtered_lines = [line for line in lines if len(line.strip()) >= self.min_line_length]
        return '\n'.join(filtered_lines)
    
    @staticmethod
    def merge_text_files(input_files: List[str], output_file: str) -> str:
        """
        Fusionne plusieurs fichiers texte en un seul.
        
        Args:
            input_files: Liste des chemins des fichiers à fusionner
            output_file: Chemin du fichier de sortie
            
        Returns:
            Chemin du fichier fusionné
            
        Raises:
            FileNotFoundError: Si un des fichiers d'entrée n'existe pas
        """
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for i, file_path in enumerate(input_files):
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Fichier non trouvé: {file_path}")
                
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                
                outfile.write(f"--- Document {i+1}: {os.path.basename(file_path)} ---\n\n")
                outfile.write(content)
                outfile.write("\n\n")
        
        logger.info(f"{len(input_files)} fichiers fusionnés dans {output_file}")
        return output_file
    
    def extract_metadata(self, pdf_path: str) -> Dict[str, str]:
        """
        Extrait les métadonnées d'un fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire des métadonnées
            
        Raises:
            FileNotFoundError: Si le fichier PDF n'existe pas
            ValueError: Si le fichier n'est pas un PDF valide
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            
            # Métadonnées standard
            metadata = {
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "keywords": doc.metadata.get("keywords", ""),
                "creator": doc.metadata.get("creator", ""),
                "producer": doc.metadata.get("producer", ""),
                "creation_date": doc.metadata.get("creationDate", ""),
                "modification_date": doc.metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "file_size": os.path.getsize(pdf_path),
                "filename": os.path.basename(pdf_path)
            }
            
            # Informations sur les pages
            page_info = []
            for i, page in enumerate(doc):
                page_data = {
                    "page_number": i+1,
                    "width": page.rect.width,
                    "height": page.rect.height,
                    "rotation": page.rotation
                }
                page_info.append(page_data)
            
            metadata["pages"] = page_info
            
            # Extraire le texte de la première page pour aider à l'identification
            if doc.page_count > 0:
                first_page_text = doc[0].get_text()
                # Limiter à 500 caractères
                metadata["first_page_preview"] = first_page_text[:500] if first_page_text else ""
                
                # Détection de langue si du texte est disponible
                if first_page_text and HAS_LANGDETECT:
                    try:
                        detected_lang = detect_language(first_page_text)
                        metadata["language_code"] = detected_lang
                        metadata["language"] = get_language_name(detected_lang)
                        logger.info(f"Langue détectée pour {pdf_path}: {metadata['language']} ({detected_lang})")
                    except Exception as e:
                        logger.warning(f"Erreur lors de la détection de langue pour {pdf_path}: {str(e)}")
            
            doc.close()
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des métadonnées de {pdf_path}: {str(e)}")
            raise ValueError(f"Erreur lors de l'extraction des métadonnées: {str(e)}")
    
    def extract_text_and_metadata_from_file(self, pdf_path: str) -> Dict[str, Union[Dict[int, str], Dict[str, str]]]:
        """
        Extrait à la fois le texte et les métadonnées d'un fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            
        Returns:
            Dictionnaire contenant le texte et les métadonnées
        """
        return {
            "text": self.extract_text_from_file(pdf_path),
            "metadata": self.extract_metadata(pdf_path)
        }
    
    def extract_images(self, pdf_path: str, output_dir: str = None, min_width: int = 100, min_height: int = 100) -> List[Dict[str, any]]:
        """
        Extrait les images d'un fichier PDF.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Répertoire de sortie pour sauvegarder les images (None pour ne pas sauvegarder)
            min_width: Largeur minimale des images à extraire
            min_height: Hauteur minimale des images à extraire
            
        Returns:
            Liste de dictionnaires contenant les informations sur les images extraites
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"Le fichier PDF n'existe pas: {pdf_path}")
        
        # Créer le répertoire de sortie si nécessaire
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Générer un nom de base pour les images basé sur le nom du fichier
        base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        
        images_info = []
        
        try:
            doc = fitz.open(pdf_path)
            image_count = 0
            
            # Parcourir chaque page
            for page_num, page in enumerate(doc):
                # Obtenir les images de la page
                image_list = page.get_images(full=True)
                
                # Traiter chaque image
                for img_index, img in enumerate(image_list):
                    try:
                        # Extraire l'image
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Si les dimensions sont disponibles, vérifier les dimensions minimales
                        if "width" in base_image and "height" in base_image:
                            if base_image["width"] < min_width or base_image["height"] < min_height:
                                continue
                        
                        image_count += 1
                        image_filename = f"{base_filename}_page{page_num+1}_img{img_index}.{image_ext}"
                        
                        # Sauvegarder l'image si un répertoire de sortie est spécifié
                        image_path = None
                        if output_dir:
                            image_path = os.path.join(output_dir, image_filename)
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)
                        
                        # Ajouter les informations sur l'image
                        image_info = {
                            "page_num": page_num,
                            "img_index": img_index,
                            "filename": image_filename,
                            "path": image_path,
                            "ext": image_ext,
                            "width": base_image.get("width"),
                            "height": base_image.get("height"),
                            "colorspace": base_image.get("colorspace"),
                            "xref": xref
                        }
                        
                        images_info.append(image_info)
                    
                    except Exception as e:
                        logger.warning(f"Erreur lors de l'extraction de l'image {img_index} de la page {page_num} de {pdf_path}: {str(e)}")
            
            doc.close()
            logger.info(f"Extraction réussie de {image_count} images depuis {pdf_path}")
            
            return images_info
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des images de {pdf_path}: {str(e)}")
            raise ValueError(f"Erreur lors de l'extraction des images: {str(e)}")
    
    def extract_text_from_file_with_retry(self, pdf_path: str, max_retries: int = 3) -> Dict[int, str]:
        """
        Extrait le texte d'un fichier PDF avec mécanisme de réessai.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            max_retries: Nombre maximum de tentatives en cas d'échec
            
        Returns:
            Dictionnaire avec les numéros de page comme clés et le texte comme valeurs
        """
        retry_count = 0
        last_exception = None
        
        while retry_count < max_retries:
            try:
                return self.extract_text_from_file(pdf_path)
            except ValueError as e:
                # Attendre un peu avant de réessayer
                time.sleep(1)
                retry_count += 1
                last_exception = e
                logger.warning(f"Tentative {retry_count}/{max_retries} échouée pour {pdf_path}: {str(e)}")
        
        # Si toutes les tentatives échouent, lever l'exception
        if last_exception:
            logger.error(f"Échec de l'extraction après {max_retries} tentatives pour {pdf_path}")
            raise last_exception
        
        return {}
    
    def is_scanned_pdf(self, pdf_path: str, text_threshold: int = 100) -> bool:
        """
        Détecte si un PDF est scanné (principalement des images avec peu de texte).
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            text_threshold: Seuil de caractères par page pour considérer qu'une page contient du texte
            
        Returns:
            True si le PDF est probablement scanné, False sinon
        """
        try:
            doc = fitz.open(pdf_path)
            
            # Vérifier chaque page
            total_pages = doc.page_count
            pages_with_text = 0
            
            for page in doc:
                text = page.get_text()
                if len(text.strip()) > text_threshold:
                    pages_with_text += 1
            
            doc.close()
            
            # Si moins de 20% des pages contiennent du texte, c'est probablement un PDF scanné
            return pages_with_text / total_pages < 0.2 if total_pages > 0 else False
            
        except Exception as e:
            logger.error(f"Erreur lors de la détection de PDF scanné pour {pdf_path}: {str(e)}")
            return False
    
    def extract_text_with_ocr(self, pdf_path: str, language: str = "fra", use_gpu: bool = False, 
                           detect_lang: bool = True) -> Dict[int, str]:
        """
        Extrait le texte d'un PDF scanné en utilisant OCR.
        
        Note: Cette fonction nécessite que pytesseract et PyTesseract soient installés.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            language: Code de langue pour Tesseract (fra pour français, eng pour anglais)
            use_gpu: Utiliser l'accélération GPU pour l'OCR si disponible
            detect_lang: Si True, tente de détecter automatiquement la langue de chaque page
            
        Returns:
            Dictionnaire avec les numéros de page comme clés et le texte comme valeurs
        """
        try:
            # Vérifier la disponibilité de pytesseract
            import pytesseract
            from PIL import Image
            import io
        except ImportError:
            logger.error("pytesseract ou Pillow n'est pas installé. Impossible d'utiliser l'OCR.")
            raise ImportError("pytesseract ou Pillow n'est pas installé. Installez-les avec 'pip install pytesseract Pillow'.")
        
        # Définir la configuration Tesseract
        config = ""
        if use_gpu:
            config += " --oem 2"  # Utiliser le moteur LSTM avec l'accélération GPU si disponible
        
        result = {}
        
        try:
            doc = fitz.open(pdf_path)
            
            # Si détection de langue activée, détecter la langue du document entier
            document_lang = language
            if detect_lang and HAS_LANGDETECT:
                # Échantillonner quelques pages pour la détection
                sample_text = ""
                sample_pages = min(3, doc.page_count)
                for i in range(sample_pages):
                    page_idx = i * (doc.page_count // (sample_pages + 1))
                    sample_text += doc[page_idx].get_text() + "\n"
                
                if sample_text.strip():
                    detected_lang = detect_language(sample_text)
                    document_lang = get_ocr_language_code(detected_lang)
                    logger.info(f"Langue détectée pour l'OCR: {get_language_name(detected_lang)} ({document_lang})")
            
            for page_num, page in enumerate(tqdm(range(doc.page_count), desc="OCR des pages", leave=False)):
                page = doc[page_num]
                
                # Obtenir un rendu de la page en tant qu'image
                pix = page.get_pixmap(dpi=300)  # Plus haute résolution pour l'OCR
                img_bytes = pix.tobytes("png")
                
                # Convertir en image PIL
                img = Image.open(io.BytesIO(img_bytes))
                
                # Extraire le texte avec Tesseract
                text = pytesseract.image_to_string(img, lang=document_lang, config=config)
                
                result[page_num] = text
            
            doc.close()
            logger.info(f"OCR réussi pour {pdf_path}, {len(result)} pages extraites")
            
            return result
            
        except Exception as e:
            logger.error(f"Erreur lors de l'OCR de {pdf_path}: {str(e)}")
            raise ValueError(f"Erreur lors de l'OCR: {str(e)}")
    
    def extract_text_auto(self, pdf_path: str, ocr_language: str = "fra", text_threshold: int = 100, 
                         detect_lang: bool = True) -> Dict[int, str]:
        """
        Extrait automatiquement le texte en détectant si un PDF est scanné et en utilisant l'OCR si nécessaire.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            ocr_language: Code de langue pour Tesseract
            text_threshold: Seuil de caractères par page pour détecter un PDF scanné
            detect_lang: Si True, tente de détecter automatiquement la langue du document
            
        Returns:
            Dictionnaire avec les numéros de page comme clés et le texte comme valeurs
        """
        # Vérifier si le PDF est scanné
        is_scanned = self.is_scanned_pdf(pdf_path, text_threshold)
        
        if is_scanned:
            logger.info(f"PDF scanné détecté: {pdf_path}, utilisation de l'OCR")
            try:
                return self.extract_text_with_ocr(pdf_path, ocr_language, detect_lang=detect_lang)
            except ImportError:
                logger.warning("OCR non disponible, essai d'extraction standard")
                result = self.extract_text_from_file(pdf_path)
                
                # Si détection de langue activée, détecter la langue
                if detect_lang and HAS_LANGDETECT:
                    all_text = ""
                    for page_num in sorted(result.keys()):
                        all_text += result[page_num] + "\n"
                    
                    if all_text.strip():
                        detected_lang = detect_language(all_text)
                        
                        # Ajouter l'information de langue dans les métadonnées
                        result["metadata"] = {
                            "language_code": detected_lang,
                            "language": get_language_name(detected_lang)
                        }
                        
                        logger.info(f"Langue détectée: {get_language_name(detected_lang)} ({detected_lang})")
                
                return result
        else:
            logger.info(f"PDF numérique détecté: {pdf_path}, extraction standard")
            result = self.extract_text_from_file(pdf_path)
            
            # Si détection de langue activée, détecter la langue
            if detect_lang and HAS_LANGDETECT:
                all_text = ""
                for page_num in sorted(result.keys()):
                    all_text += result[page_num] + "\n"
                
                if all_text.strip():
                    detected_lang = detect_language(all_text)
                    
                    # Ajouter l'information de langue dans les métadonnées
                    result["metadata"] = {
                        "language_code": detected_lang,
                        "language": get_language_name(detected_lang)
                    }
                    
                    logger.info(f"Langue détectée: {get_language_name(detected_lang)} ({detected_lang})")
            
            return result


# Exemple d'utilisation
if __name__ == "__main__":
    # Cet exemple montre comment utiliser la classe PDFExtractor
    extractor = PDFExtractor(remove_headers_footers=True, min_line_length=10)
    
    # Exemple avec un seul fichier
    pdf_path = "chemin/vers/votre/document.pdf"
    try:
        # Extraction simple
        text_by_page = extractor.extract_text_from_file(pdf_path)
        print(f"Nombre de pages extraites: {len(text_by_page)}")
        
        # Sauvegarder en tant que fichier texte
        output_dir = "chemin/vers/sortie"
        saved_files = extractor.extract_and_save_as_txt(pdf_path, output_dir)
        print(f"Fichiers sauvegardés: {saved_files}")
        
    except FileNotFoundError:
        print("Cet exemple ne peut pas être exécuté car le fichier n'existe pas.")
        print("Modifiez les chemins pour utiliser vos propres fichiers PDF.")
