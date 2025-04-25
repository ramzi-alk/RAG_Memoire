import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from src.pipeline.rag_pipeline import RAGPipeline
import os
import json
import threading
import time
import logging

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Système RAG - Assistant de recherche documentaire")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        
        # Variables
        self.pipeline = None
        self.pdf_dir = None
        self.default_api_key = "YOUR_DEFAULT_API_KEY"  # Remplacez par votre clé API par défaut
        self.default_pdf_dir = "./documents"  # Répertoire par défaut
        self.default_embedding_model = "all-MiniLM-L6-v2"  # Modèle par défaut (plus petit et plus rapide)
        self.default_exclude_patterns = ".git, .DS_Store, .py, .txt, .exe, .json, ~$"
        self.default_reset_collection = False
        
        # Liste des modèles d'embedding disponibles
        self.embedding_models = [
            "all-MiniLM-L6-v2",                    # Plus petit et plus rapide (384 dimensions)
            "paraphrase-multilingual-MiniLM-L12-v2", # Bon compromis taille/performance, multilingue
            "distiluse-base-multilingual-cased-v1",  # Multilingue
            "all-mpnet-base-v2"                     # Meilleure performance mais plus grand
        ]
        
        # Chargement de la configuration sauvegardée
        self.load_saved_settings()
        
        # Style
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 10), padding=4)
        style.configure("TLabel", font=("Arial", 11), background="#f0f0f0")
        style.configure("Header.TLabel", font=("Arial", 12, "bold"), background="#f0f0f0")
        style.configure("TFrame", background="#f0f0f0")
        
        # Frame principal
        main_frame = ttk.Frame(root, padding="10", style="TFrame")
        main_frame.pack(fill="both", expand=True)
        
        # Section d'initialisation - Réduit en hauteur
        init_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="5")
        init_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(init_frame, text="Clé API Gemini:", style="TLabel").grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.api_key_entry = ttk.Entry(init_frame, width=50)
        self.api_key_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)
        self.api_key_entry.insert(0, self.default_api_key)
        
        ttk.Label(init_frame, text="Modèle d'embedding:", style="TLabel").grid(row=1, column=0, sticky="w", padx=5, pady=2)
        self.embedding_model_combo = ttk.Combobox(init_frame, values=self.embedding_models, width=48)
        self.embedding_model_combo.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        self.embedding_model_combo.set(self.default_embedding_model)
        
        ttk.Label(init_frame, text="Répertoire PDF:", style="TLabel").grid(row=2, column=0, sticky="w", padx=5, pady=2)
        self.dir_frame = ttk.Frame(init_frame)
        self.dir_frame.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        
        self.dir_entry = ttk.Entry(self.dir_frame, width=40)
        self.dir_entry.pack(side="left", fill="x", expand=True)
        self.dir_entry.insert(0, self.default_pdf_dir)
        
        ttk.Button(self.dir_frame, text="Parcourir", command=self.select_pdf_dir).pack(side="right", padx=(5, 0))
        
        # Ajout de la section pour les motifs d'exclusion
        ttk.Label(init_frame, text="Motifs à exclure:", style="TLabel").grid(row=3, column=0, sticky="w", padx=5, pady=2)
        self.exclude_frame = ttk.Frame(init_frame)
        self.exclude_frame.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        self.exclude_entry = ttk.Entry(self.exclude_frame, width=40)
        self.exclude_entry.pack(side="left", fill="x", expand=True)
        self.exclude_entry.insert(0, self.default_exclude_patterns)
        
        ttk.Label(self.exclude_frame, text="(séparés par virgules)", style="TLabel", font=("Arial", 8)).pack(side="right", padx=(5, 0))
        
        # Ajout de la case à cocher pour réinitialiser la collection
        ttk.Label(init_frame, text="Options:", style="TLabel").grid(row=4, column=0, sticky="w", padx=5, pady=2)
        self.options_frame = ttk.Frame(init_frame)
        self.options_frame.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        
        self.reset_collection_var = tk.BooleanVar(value=self.default_reset_collection)
        self.reset_collection_check = ttk.Checkbutton(
            self.options_frame, 
            text="Réinitialiser la collection (lors du changement de modèle d'embedding)",
            variable=self.reset_collection_var
        )
        self.reset_collection_check.pack(side="left", fill="x", expand=True)
        
        init_button_frame = ttk.Frame(init_frame)
        init_button_frame.grid(row=5, column=0, columnspan=2, pady=5)
        
        ttk.Button(init_button_frame, text="Initialiser le pipeline", command=self.init_pipeline_async).pack(side="left", padx=3)
        ttk.Button(init_button_frame, text="Traiter les documents", command=self.process_pdfs_async).pack(side="left", padx=3)
        ttk.Button(init_button_frame, text="Sauvegarder config", command=self.save_settings).pack(side="left", padx=3)
        
        # Création des onglets
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Premier onglet: Recherche
        search_tab = ttk.Frame(self.notebook)
        self.notebook.add(search_tab, text="Recherche")
        
        # Section de requête
        query_frame = ttk.LabelFrame(search_tab, text="Recherche documentaire", padding="5")
        query_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(query_frame, text="Posez votre question:", style="TLabel").pack(anchor="w", padx=5, pady=2)
        self.query_entry = ttk.Entry(query_frame, width=70)
        self.query_entry.pack(fill="x", padx=5, pady=2)
        
        ttk.Button(query_frame, text="Rechercher", command=self.execute_query_async).pack(pady=5)
        
        # Section des résultats - Zone de réponse agrandie
        result_frame = ttk.LabelFrame(search_tab, text="Résultats", padding="5")
        result_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Réponse - Agrandie
        response_frame = ttk.Frame(result_frame)
        response_frame.pack(fill="both", expand=True, padx=2, pady=2)
        
        ttk.Label(response_frame, text="Réponse:", style="TLabel").pack(anchor="w", padx=3, pady=2)
        
        # Zone de texte agrandie pour la réponse (70% de l'espace disponible)
        self.result_text = tk.Text(response_frame, width=70, wrap="word", font=("Arial", 10))
        self.result_text.pack(fill="both", expand=True, padx=3, pady=2)
        
        result_scroll = ttk.Scrollbar(self.result_text, orient="vertical", command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        result_scroll.pack(side="right", fill="y")
        
        # Sources - Réduite en hauteur
        sources_frame = ttk.Frame(result_frame)
        sources_frame.pack(fill="x", padx=2, pady=2)
        
        ttk.Label(sources_frame, text="Sources:", style="TLabel").pack(anchor="w", padx=3, pady=2)
        
        # Tableau des sources plus compact
        tree_container = ttk.Frame(sources_frame)
        tree_container.pack(fill="x", expand=False, padx=3, pady=2)
        
        self.sources_tree = ttk.Treeview(tree_container, columns=("source", "score"), show="headings", height=4)
        self.sources_tree.heading("source", text="Document source")
        self.sources_tree.heading("score", text="Score de pertinence")
        self.sources_tree.column("source", width=500)
        self.sources_tree.column("score", width=250)
        self.sources_tree.pack(side="left", fill="x", expand=True)
        
        sources_scroll = ttk.Scrollbar(tree_container, orient="vertical", command=self.sources_tree.yview)
        self.sources_tree.configure(yscrollcommand=sources_scroll.set)
        sources_scroll.pack(side="right", fill="y")
        
        # Deuxième onglet: Historique des requêtes
        history_tab = ttk.Frame(self.notebook)
        self.notebook.add(history_tab, text="Historique")
        
        # Cadre pour le filtre de recherche
        filter_frame = ttk.Frame(history_tab, padding="5")
        filter_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(filter_frame, text="Filtrer l'historique:", style="TLabel").pack(side="left", padx=5)
        self.history_filter_entry = ttk.Entry(filter_frame, width=40)
        self.history_filter_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        ttk.Button(filter_frame, text="Rechercher", command=self.filter_history).pack(side="left", padx=5)
        ttk.Button(filter_frame, text="Effacer l'historique", command=self.clear_history).pack(side="left", padx=5)
        
        # Tableau de l'historique
        history_container = ttk.Frame(history_tab, padding="5")
        history_container.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Création du tableau d'historique avec colonnes pour date, requête et nombre de sources
        self.history_tree = ttk.Treeview(
            history_container, 
            columns=("timestamp", "query", "sources_count"), 
            show="headings", 
            height=15
        )
        
        self.history_tree.heading("timestamp", text="Date/Heure")
        self.history_tree.heading("query", text="Requête")
        self.history_tree.heading("sources_count", text="Nb sources")
        
        self.history_tree.column("timestamp", width=150, anchor="w")
        self.history_tree.column("query", width=450, anchor="w")
        self.history_tree.column("sources_count", width=80, anchor="center")
        
        # Configuration des événements et scrollbar
        self.history_tree.pack(side="left", fill="both", expand=True)
        self.history_tree.bind("<Double-1>", self.on_history_double_click)
        
        history_scroll = ttk.Scrollbar(history_container, orient="vertical", command=self.history_tree.yview)
        self.history_tree.configure(yscrollcommand=history_scroll.set)
        history_scroll.pack(side="right", fill="y")
        
        # Cadre pour les boutons d'action
        history_action_frame = ttk.Frame(history_tab, padding="5")
        history_action_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(history_action_frame, text="Réutiliser cette requête", command=self.reuse_selected_query).pack(side="left", padx=5)
        ttk.Button(history_action_frame, text="Afficher les détails", command=self.show_history_details).pack(side="left", padx=5)
        ttk.Button(history_action_frame, text="Actualiser", command=self.refresh_history).pack(side="left", padx=5)
        
        # Barre de statut
        self.status_var = tk.StringVar()
        self.status_var.set("Prêt")
        self.status_bar = ttk.Label(root, textvariable=self.status_var, relief="sunken", anchor="w", padding=(10, 2))
        self.status_bar.pack(side="bottom", fill="x")
        
        # Progress bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=600, mode="indeterminate")
        self.progress.pack(side="bottom", fill="x", before=self.status_bar, pady=5)
        
        # Variable de suivi pour les threads
        self.thread_running = False

    def load_saved_settings(self):
        try:
            if os.path.exists("./app_settings.json"):
                with open("./app_settings.json", "r") as f:
                    settings = json.load(f)
                    self.default_api_key = settings.get("api_key", self.default_api_key)
                    self.default_pdf_dir = settings.get("pdf_dir", self.default_pdf_dir)
                    self.default_embedding_model = settings.get("embedding_model", self.default_embedding_model)
                    self.default_exclude_patterns = settings.get("exclude_patterns", self.default_exclude_patterns)
                    self.default_reset_collection = settings.get("reset_collection", False)
        except Exception as e:
            print(f"Erreur lors du chargement des paramètres: {e}")

    def save_settings(self):
        settings = {
            "api_key": self.api_key_entry.get(),
            "pdf_dir": self.dir_entry.get(),
            "embedding_model": self.embedding_model_combo.get(),
            "exclude_patterns": self.exclude_entry.get(),
            "reset_collection": self.reset_collection_var.get()
        }
        try:
            with open("./app_settings.json", "w") as f:
                json.dump(settings, f)
            messagebox.showinfo("Sauvegarde réussie", "Configuration sauvegardée avec succès.")
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de sauvegarder les paramètres: {str(e)}")
    
    def update_status(self, message):
        """Met à jour le message de statut de manière thread-safe"""
        if self.root.winfo_exists():  # Vérifier si la fenêtre existe toujours
            self.root.after(0, lambda: self.status_var.set(message))
            
    def start_progress(self):
        """Démarre l'animation de la barre de progression"""
        if self.root.winfo_exists():
            self.root.after(0, lambda: self.progress.start(10))
    
    def stop_progress(self):
        """Arrête l'animation de la barre de progression"""
        if self.root.winfo_exists():
            self.root.after(0, lambda: self.progress.stop())
            
    def init_pipeline_async(self):
        """Version asynchrone de l'initialisation du pipeline"""
        if self.thread_running:
            messagebox.showinfo("Opération en cours", "Une opération est déjà en cours. Veuillez patienter.")
            return
            
        api_key = self.api_key_entry.get()
        embedding_model = self.embedding_model_combo.get()
        reset_collection = self.reset_collection_var.get()
        
        if not api_key:
            messagebox.showerror("Erreur", "Veuillez entrer une clé API Gemini.")
            return
        
        # Démarrer une thread pour l'initialisation
        self.thread_running = True
        self.update_status("Initialisation du pipeline... (cela peut prendre quelques minutes)")
        self.start_progress()
        
        def init_task():
            try:
                # Initialisation dans un thread séparé
                pipeline = RAGPipeline(
                    data_dir="./data",
                    db_dir="./chroma_db",
                    collection_name="docss",
                    gemini_api_key=api_key,
                    gemini_model="gemini-2.0-flash-001",
                    language="french",
                    embedding_model_name=embedding_model,
                    reset_collection=reset_collection
                )
                pipeline.save_config("./config.json")
                
                # Mise à jour de l'UI dans le thread principal
                if self.root.winfo_exists():
                    self.root.after(0, lambda p=pipeline, m=embedding_model: self._on_init_complete(p, m))
                
            except Exception as e:
                # Gestion des erreurs
                error_msg = str(e)
                if self.root.winfo_exists():
                    self.root.after(0, lambda msg=error_msg: self._on_init_error(msg))
            finally:
                # Réinitialiser l'état du thread dans tous les cas
                if self.root.winfo_exists():
                    self.root.after(0, self._reset_thread_state)
        
        # Lancer le thread
        threading.Thread(target=init_task, daemon=True).start()
    
    def _on_init_complete(self, pipeline, embedding_model):
        """Callback appelé quand l'initialisation est terminée avec succès"""
        self.pipeline = pipeline
        self.update_status("Pipeline initialisé avec succès.")
        messagebox.showinfo("Succès", f"Pipeline initialisé avec succès.\nModèle d'embedding: {embedding_model}")
        # Charger l'historique après l'initialisation du pipeline
        self.refresh_history()

    def _on_init_error(self, error_message):
        """Callback appelé quand l'initialisation échoue"""
        self.update_status("Erreur d'initialisation")
        messagebox.showerror("Erreur", f"Erreur lors de l'initialisation : {error_message}")

    def select_pdf_dir(self):
        selected_dir = filedialog.askdirectory(initialdir=self.default_pdf_dir)
        if selected_dir:
            self.pdf_dir = selected_dir
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, selected_dir)

    def process_pdfs_async(self):
        """Version asynchrone du traitement des PDFs"""
        if self.thread_running:
            messagebox.showinfo("Opération en cours", "Une opération est déjà en cours. Veuillez patienter.")
            return
            
        if not self.pipeline:
            messagebox.showerror("Erreur", "Veuillez initialiser le pipeline d'abord.")
            return
            
        pdf_dir = self.dir_entry.get()
        if not pdf_dir or not os.path.exists(pdf_dir):
            messagebox.showerror("Erreur", "Veuillez sélectionner un répertoire valide contenant des PDF.")
            return
        
        # Récupération des motifs d'exclusion
        exclude_patterns_text = self.exclude_entry.get()
        exclude_patterns = [p.strip() for p in exclude_patterns_text.split(',') if p.strip()]
        
        # Démarrer une thread pour le traitement
        self.thread_running = True
        self.update_status("Traitement des documents en cours...")
        self.start_progress()
        
        def process_task():
            try:
                # Traitement dans un thread séparé
                stats = self.pipeline.process_pdf_directory(
                    pdf_dir=pdf_dir,
                    recursive=True,
                    chunk_size=1000,
                    chunk_overlap=200,
                    exclude_patterns=exclude_patterns,
                    file_extension=".pdf"
                )
                
                # Mise à jour de l'UI dans le thread principal
                if self.root.winfo_exists():
                    self.root.after(0, lambda s=stats: self._on_process_complete(s))
                
            except Exception as e:
                # Gestion des erreurs
                error_msg = str(e)
                if self.root.winfo_exists():
                    self.root.after(0, lambda msg=error_msg: self._on_process_error(msg))
            finally:
                # Réinitialiser l'état du thread dans tous les cas
                if self.root.winfo_exists():
                    self.root.after(0, self._reset_thread_state)
        
        # Lancer le thread
        threading.Thread(target=process_task, daemon=True).start()
    
    def _on_process_complete(self, stats):
        """Callback appelé quand le traitement est terminé avec succès"""
        status_msg = f"Traitement terminé: {stats['pdf_count']} PDF, {stats['chunk_count']} chunks."
        self.update_status(status_msg)
        
        # Message plus détaillé
        detail_msg = (f"Traitement terminé :\n"
                      f"- Documents traités: {stats['pdf_count']}\n"
                      f"- Fragments générés: {stats['chunk_count']}\n"
                      f"- Fichiers ignorés: {len(stats.get('skipped_files', []))}\n"
                      f"- Fichiers en erreur: {len(stats.get('failed_files', []))}")
                      
        messagebox.showinfo("Succès", detail_msg)
        
    def _on_process_error(self, error_message):
        """Callback appelé quand le traitement échoue"""
        self.update_status("Erreur lors du traitement des documents.")
        messagebox.showerror("Erreur", f"Erreur lors du traitement des PDF : {error_message}")

    def execute_query_async(self):
        """Version asynchrone de l'exécution de requête"""
        if self.thread_running:
            messagebox.showinfo("Opération en cours", "Une opération est déjà en cours. Veuillez patienter.")
            return
            
        query_text = self.query_entry.get()
        if not self.pipeline:
            messagebox.showerror("Erreur", "Veuillez initialiser le pipeline d'abord.")
            return
        if not query_text:
            messagebox.showerror("Erreur", "Veuillez entrer une question.")
            return
        
        # Démarrer une thread pour la requête
        self.thread_running = True
        self.update_status("Recherche en cours...")
        self.start_progress()
        
        # Effacer les résultats précédents
        self.result_text.delete(1.0, tk.END)
        for item in self.sources_tree.get_children():
            self.sources_tree.delete(item)
        
        def query_task():
            try:
                # Requête dans un thread séparé
                logger.info(f"Exécution de la requête: '{query_text}'")
                result = self.pipeline.query(query_text=query_text, n_results=10)
                
                # Log de diagnostic pour inspecter la structure du résultat
                logger.info(f"Résultat obtenu avec clés: {list(result.keys())}")
                if 'sources' in result:
                    logger.info(f"Nombre de sources: {len(result['sources'])}")
                    if result['sources']:
                        first_source = result['sources'][0]
                        logger.info(f"Structure de la première source: clés={list(first_source.keys())}")
                else:
                    logger.warning("Pas de sources retournées dans le résultat")
                
                # Mise à jour de l'UI dans le thread principal
                if self.root.winfo_exists():
                    self.root.after(0, lambda r=result: self._on_query_complete(r))
                
            except Exception as e:
                # Gestion des erreurs
                error_msg = str(e)
                logger.error(f"Erreur lors de l'exécution de la requête: {error_msg}", exc_info=True)
                if self.root.winfo_exists():
                    self.root.after(0, lambda msg=error_msg: self._on_query_error(msg))
            finally:
                # Important : Réinitialiser l'état thread_running dans tous les cas
                if self.root.winfo_exists():
                    self.root.after(0, self._reset_thread_state)
        
        # Lancer le thread
        threading.Thread(target=query_task, daemon=True).start()
    
    def _reset_thread_state(self):
        """Réinitialise l'état du thread et arrête la barre de progression"""
        self.thread_running = False
        self.stop_progress()
    
    def _on_query_complete(self, result):
        """Callback appelé quand la requête est terminée avec succès"""
        # Afficher la réponse
        self.result_text.insert(tk.END, f"{result['response']}")
        
        # Afficher les sources dans le tableau
        for i, source in enumerate(result.get('sources', [])):
            try:
                # Convertir le score en pourcentage et ajouter une représentation visuelle
                # Gérer le cas où score est None ou non présent
                score_value = source.get('score')
                if score_value is None:
                    score_percent = 0
                else:
                    score_percent = int(score_value * 100)
                
                # Créer une représentation visuelle avec des étoiles
                stars = ""
                if score_percent >= 70:
                    stars = "★★★★★ (Excellent)"
                elif score_percent >= 60:
                    stars = "★★★★☆ (Très pertinent)"
                elif score_percent >= 50:
                    stars = "★★★☆☆ (Pertinent)"
                elif score_percent >= 40:
                    stars = "★★☆☆☆ (Modérément pertinent)"
                else:
                    stars = "★☆☆☆☆ (Peu pertinent)"
                
                # Formater le score pour l'affichage
                score_display = f"{score_percent}% {stars}"
                
                # Gérer le cas où metadata est None ou non présent
                metadata = source.get('metadata')
                if metadata is None:
                    source_text = 'Source inconnue'
                else:
                    source_text = metadata.get('source', 'Source inconnue')
                
                self.sources_tree.insert("", "end", values=(source_text, score_display))
            except Exception as e:
                # En cas d'erreur lors du traitement d'une source, enregistrer et continuer
                error_msg = str(e)
                logger.error(f"Erreur lors du traitement de la source {i}: {error_msg}")
                self.sources_tree.insert("", "end", values=(f"Source {i+1} (erreur)", "N/A"))
        
        # Rafraîchir l'historique après une nouvelle requête
        self.refresh_history()
        
        self.update_status("Recherche terminée avec succès.")
        
    def _on_query_error(self, error_message):
        """Callback appelé quand la requête échoue"""
        self.update_status("Erreur lors de la recherche.")
        messagebox.showerror("Erreur", f"Erreur lors de l'exécution de la requête : {error_message}")

    def filter_history(self):
        """Filtre l'historique des requêtes selon le texte saisi"""
        if not self.pipeline:
            messagebox.showerror("Erreur", "Veuillez initialiser le pipeline d'abord.")
            return
            
        filter_text = self.history_filter_entry.get()
        self.load_history_data(filter_text)
    
    def clear_history(self):
        """Efface tout l'historique des requêtes après confirmation"""
        if not self.pipeline:
            messagebox.showerror("Erreur", "Veuillez initialiser le pipeline d'abord.")
            return
            
        if not messagebox.askyesno("Confirmation", "Êtes-vous sûr de vouloir effacer tout l'historique des requêtes ?"):
            return
            
        if self.pipeline.clear_query_history():
            # Vider le tableau d'historique
            for item in self.history_tree.get_children():
                self.history_tree.delete(item)
            messagebox.showinfo("Succès", "L'historique des requêtes a été effacé.")
        else:
            messagebox.showwarning("Avertissement", "L'historique des requêtes est désactivé ou inaccessible.")
    
    def on_history_double_click(self, event):
        """Gère le double-clic sur une entrée de l'historique"""
        selected_items = self.history_tree.selection()
        if selected_items:
            self.reuse_selected_query()
    
    def reuse_selected_query(self):
        """Réutilise la requête sélectionnée dans l'historique"""
        selected_items = self.history_tree.selection()
        if not selected_items:
            messagebox.showinfo("Sélection", "Veuillez sélectionner une requête dans l'historique.")
            return
            
        # Récupérer la requête sélectionnée
        item_id = selected_items[0]
        query_text = self.history_tree.item(item_id, "values")[1]
        
        # Passer à l'onglet de recherche
        self.notebook.select(0)  # Premier onglet (Recherche)
        
        # Insérer la requête dans le champ de recherche
        self.query_entry.delete(0, tk.END)
        self.query_entry.insert(0, query_text)
        
        # Option: exécuter automatiquement la requête
        if messagebox.askyesno("Exécution automatique", "Souhaitez-vous exécuter cette requête immédiatement ?"):
            self.execute_query_async()
    
    def show_history_details(self):
        """Affiche les détails d'une entrée d'historique sélectionnée"""
        if not self.pipeline:
            messagebox.showerror("Erreur", "Veuillez initialiser le pipeline d'abord.")
            return
            
        selected_items = self.history_tree.selection()
        if not selected_items:
            messagebox.showinfo("Sélection", "Veuillez sélectionner une requête dans l'historique.")
            return
            
        # Récupérer l'ID de l'entrée sélectionnée
        item_id = selected_items[0]
        item_index = int(item_id[1:])  # Supposant que l'ID est de la forme "I1", "I2", etc.
        
        # Récupérer toutes les entrées d'historique
        history_entries = self.pipeline.get_query_history()
        if not history_entries or item_index >= len(history_entries):
            messagebox.showwarning("Erreur", "Entrée d'historique introuvable.")
            return
        
        # Récupérer l'entrée d'historique complète
        entry = history_entries[item_index]
        
        # Créer une fenêtre popup pour afficher les détails
        details_window = tk.Toplevel(self.root)
        details_window.title(f"Détails de la requête - {entry['timestamp']}")
        details_window.geometry("700x500")
        details_window.transient(self.root)
        details_window.grab_set()
        
        # Zone de défilement pour le contenu
        main_frame = ttk.Frame(details_window, padding="10")
        main_frame.pack(fill="both", expand=True)
        
        # Requête
        ttk.Label(main_frame, text="Requête:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        query_frame = ttk.Frame(main_frame)
        query_frame.pack(fill="x", pady=(0, 10))
        
        query_text = tk.Text(query_frame, wrap="word", height=2, font=("Arial", 10))
        query_text.pack(fill="x", side="left", expand=True)
        query_text.insert("1.0", entry["query"])
        query_text.config(state="disabled")
        
        query_scroll = ttk.Scrollbar(query_frame, orient="vertical", command=query_text.yview)
        query_text.configure(yscrollcommand=query_scroll.set)
        query_scroll.pack(side="right", fill="y")
        
        # Réponse
        ttk.Label(main_frame, text="Réponse:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        response_frame = ttk.Frame(main_frame)
        response_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        response_text = tk.Text(response_frame, wrap="word", font=("Arial", 10))
        response_text.pack(fill="both", side="left", expand=True)
        response_text.insert("1.0", entry["response"])
        response_text.config(state="disabled")
        
        response_scroll = ttk.Scrollbar(response_frame, orient="vertical", command=response_text.yview)
        response_text.configure(yscrollcommand=response_scroll.set)
        response_scroll.pack(side="right", fill="y")
        
        # Sources utilisées
        ttk.Label(main_frame, text=f"Sources utilisées ({len(entry['sources'])}):", font=("Arial", 10, "bold")).pack(anchor="w", pady=(0, 5))
        
        sources_frame = ttk.Frame(main_frame)
        sources_frame.pack(fill="x", expand=False, pady=(0, 10))
        
        sources_tree = ttk.Treeview(sources_frame, columns=("source", "score", "title", "author"), show="headings", height=5)
        sources_tree.heading("source", text="Fichier source")
        sources_tree.heading("score", text="Score")
        sources_tree.heading("title", text="Titre")
        sources_tree.heading("author", text="Auteur")
        
        sources_tree.column("source", width=150)
        sources_tree.column("score", width=80)
        sources_tree.column("title", width=200)
        sources_tree.column("author", width=150)
        sources_tree.pack(side="left", fill="x", expand=True)
        
        sources_scroll = ttk.Scrollbar(sources_frame, orient="vertical", command=sources_tree.yview)
        sources_tree.configure(yscrollcommand=sources_scroll.set)
        sources_scroll.pack(side="right", fill="y")
        
        # Remplir le tableau des sources
        for i, source in enumerate(entry["sources"]):
            score_value = source.get("score", 0)
            score_str = f"{score_value:.4f}" if score_value is not None else "N/A"
            
            sources_tree.insert("", "end", values=(
                source.get("source", "Inconnu"),
                score_str,
                source.get("title", ""),
                source.get("author", "")
            ))
        
        # Bouton de fermeture
        ttk.Button(main_frame, text="Fermer", command=details_window.destroy).pack(pady=10)
    
    def refresh_history(self):
        """Actualise l'affichage de l'historique des requêtes"""
        if not self.pipeline:
            # Si appelé lors de l'initialisation et que le pipeline n'est pas prêt
            return
            
        self.load_history_data()
    
    def load_history_data(self, filter_query=None):
        """Charge les données d'historique dans le tableau"""
        # Effacer le tableau existant
        for item in self.history_tree.get_children():
            self.history_tree.delete(item)
            
        # Récupérer l'historique des requêtes
        history_entries = self.pipeline.get_query_history(filter_query=filter_query)
        if not history_entries:
            self.update_status("Aucune entrée dans l'historique des requêtes")
            return
            
        # Remplir le tableau avec les entrées d'historique
        for i, entry in enumerate(history_entries):
            query = entry.get("query", "")
            timestamp = entry.get("timestamp", "")
            sources_count = entry.get("sources_count", 0)
            
            # Tronquer les requêtes trop longues pour l'affichage
            query_display = query[:100] + "..." if len(query) > 100 else query
            
            self.history_tree.insert("", "end", iid=f"I{i}", values=(timestamp, query_display, sources_count))
            
        self.update_status(f"{len(history_entries)} entrées d'historique chargées")
        
        # Sélectionner automatiquement la première entrée
        if history_entries:
            self.history_tree.selection_set("I0")
            self.history_tree.focus("I0")
            
    def _on_query_complete(self, result):
        """Callback appelé quand la requête est terminée avec succès"""
        # Afficher la réponse
        self.result_text.insert(tk.END, f"{result['response']}")
        
        # Afficher les sources dans le tableau
        for i, source in enumerate(result.get('sources', [])):
            try:
                # Convertir le score en pourcentage et ajouter une représentation visuelle
                # Gérer le cas où score est None ou non présent
                score_value = source.get('score')
                if score_value is None:
                    score_percent = 0
                else:
                    score_percent = int(score_value * 100)
                
                # Créer une représentation visuelle avec des étoiles
                stars = ""
                if score_percent >= 90:
                    stars = "★★★★★ (Excellent)"
                elif score_percent >= 80:
                    stars = "★★★★☆ (Très pertinent)"
                elif score_percent >= 70:
                    stars = "★★★☆☆ (Pertinent)"
                elif score_percent >= 60:
                    stars = "★★☆☆☆ (Modérément pertinent)"
                else:
                    stars = "★☆☆☆☆ (Peu pertinent)"
                
                # Formater le score pour l'affichage
                score_display = f"{score_percent}% {stars}"
                
                # Gérer le cas où metadata est None ou non présent
                metadata = source.get('metadata')
                if metadata is None:
                    source_text = 'Source inconnue'
                else:
                    source_text = metadata.get('source', 'Source inconnue')
                
                self.sources_tree.insert("", "end", values=(source_text, score_display))
            except Exception as e:
                # En cas d'erreur lors du traitement d'une source, enregistrer et continuer
                error_msg = str(e)
                logger.error(f"Erreur lors du traitement de la source {i}: {error_msg}")
                self.sources_tree.insert("", "end", values=(f"Source {i+1} (erreur)", "N/A"))
        
        # Rafraîchir l'historique après une nouvelle requête
        self.refresh_history()
        
        self.update_status("Recherche terminée avec succès.")

if __name__ == "__main__":
    root = tk.Tk()
    app = RAGApp(root)
    root.mainloop()
