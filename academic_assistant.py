# academic_assistant.py
import os
import glob
import sys
import time
import datetime
import json
from typing import TypedDict, List, Dict, Any, Optional, Annotated, Union, Callable
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.schema import Document
from langchain.schema.language_model import BaseLanguageModel
import re
import logging
from langgraph.graph import StateGraph, END
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
import operator
from pydantic import BaseModel, Field
import uuid

# Import des modules de l'application
from model_connectors import ModelConnector, create_model_connector
from query_classifier import QueryClassifier, QUERY_TYPES

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration pour la visualisation LangGraph
LANGGRAPH_API_KEY = os.environ.get("LANGGRAPH_API_KEY", "")
ENABLE_TRACING = os.environ.get("ENABLE_TRACING", "false").lower() == "true"

# Définition des types pour LangGraph
class QueryState(TypedDict):
    """État de la requête dans le processus LangGraph"""
    session_id: str  # Identifiant unique de session
    original_question: str  # Question originale
    reformulated_question: Optional[str]  # Question reformulée
    query_type: str  # Type de requête
    retrieved_documents: Optional[List[Document]]  # Documents récupérés
    additional_context: Optional[str]  # Contexte additionnel
    answer: Optional[str]  # Réponse finale
    sources: Optional[List[str]]  # Sources citées
    thinking: Optional[str]  # Raisonnement interne (non visible par l'utilisateur)
    needs_refinement: bool  # Indique si la requête a besoin d'être affinée
    start_time: float  # Heure de début du traitement
    steps_completed: List[str]  # Étapes terminées
    current_step: str  # Étape actuelle
    errors: List[str]  # Erreurs rencontrées
    verbose: bool  # Mode verbeux


class AcademicAssistant:
    def __init__(
        self, 
        docs_folder: str, 
        model_type: str = "ollama",
        model_name: Optional[str] = None,
        db_path: str = "./chroma_db",
        verbose: bool = False,
        api_key: Optional[str] = None
    ):
        """
        Initialise l'assistant académique 3.0 optimisé avec LangGraph
        
        Args:
            docs_folder: Dossier contenant les documents PDF
            model_type: Type de modèle à utiliser ('ollama' ou 'claude')
            model_name: Nom spécifique du modèle (optionnel)
            db_path: Chemin de la base de données vectorielle
            verbose: Activer le mode verbeux pour plus de détails
            api_key: Clé API pour Claude (ignorée pour Ollama)
        """
        self.docs_folder = docs_folder
        self.model_type = model_type
        self.model_name = model_name
        self.db_path = db_path
        self.verbose = verbose
        self.api_key = api_key
        
        # Création du connecteur de modèle
        kwargs = {"api_key": api_key} if model_type == "claude" and api_key else {}
        self.model_connector = create_model_connector(model_type, model_name, **kwargs)
        
        # Initialisation des composants
        self.llm = None
        self.vector_store = None
        self.document_sources = {}
        self.section_headers = {}
        self.workflow = None
        self.classifier = None
        
        # Pour le suivi et la visualisation
        self.session_history = {}
        
        logger.info(f"Assistant initialisé avec le modèle {self.model_connector.get_name()}")
        if verbose:
            logger.info(f"Mode verbeux activé")
        
    def load_documents(self):
        """
        Charge tous les PDFs du dossier spécifié
        
        Returns:
            Liste de documents chargés
        """
        logger.info(f"Chargement des documents depuis {self.docs_folder}...")
        pdf_files = glob.glob(os.path.join(self.docs_folder, "*.pdf"))
        
        if not pdf_files:
            raise ValueError(f"Aucun fichier PDF trouvé dans {self.docs_folder}")
        
        documents = []
        
        for pdf_file in pdf_files:
            try:
                logger.info(f"Traitement de: {os.path.basename(pdf_file)}")
                loader = PyPDFLoader(pdf_file)
                docs = loader.load()
                
                # Tentative d'extraction de la structure des sections
                for i, doc in enumerate(docs):
                    content = doc.page_content
                    possible_headers = re.findall(r'^([A-Z][^.!?]{0,50})[.!?]', content, re.MULTILINE)
                    if possible_headers and self.verbose:
                        logger.debug(f"En-têtes potentiels dans {os.path.basename(pdf_file)} p{i+1}: {possible_headers[:3]}...")
                        self.section_headers[f"{os.path.basename(pdf_file)}_p{i+1}"] = possible_headers
                
                # Enrichir les métadonnées
                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(pdf_file)
                    doc.metadata["page"] = doc.metadata.get("page", "?")
                    doc.metadata["id"] = f"{os.path.basename(pdf_file)}_p{doc.metadata['page']}"
                
                documents.extend(docs)
                self.document_sources[os.path.basename(pdf_file)] = len(docs)
                
            except Exception as e:
                logger.error(f"Erreur lors du chargement de {pdf_file}: {e}")
        
        logger.info(f"Chargé {len(documents)} pages depuis {len(pdf_files)} documents")
        return documents

    def prepare_database(self, documents):
        """
        Divise les documents en chunks et crée une base de données vectorielle
        
        Args:
            documents: Liste des documents à indexer
            
        Returns:
            Base de données vectorielle Chroma
        """
        logger.info("Préparation de la base de connaissances...")
        
        # Configuration du text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=300,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Enrichir les chunks avec des informations de position
        total_chunks = len(chunks)
        for i, chunk in enumerate(chunks):
            chunk.metadata["rel_position"] = i / total_chunks
            if i / total_chunks < 0.2:
                chunk.metadata["document_section"] = "beginning"
            elif i / total_chunks > 0.8:
                chunk.metadata["document_section"] = "end"
            else:
                chunk.metadata["document_section"] = "middle"
            
            # Ajouter des tags pour faciliter le filtrage
            source_file = chunk.metadata.get("source_file", "")
            if "introduction" in chunk.page_content.lower():
                chunk.metadata["contains_introduction"] = True
            if "conclusion" in chunk.page_content.lower():
                chunk.metadata["contains_conclusion"] = True
            if "methodologie" in chunk.page_content.lower() or "méthode" in chunk.page_content.lower():
                chunk.metadata["contains_methodology"] = True
            
            # Marquer le chunk comme potentiellement contenant une question de recherche
            if ("question" in chunk.page_content.lower() and "recherche" in chunk.page_content.lower()) or \
               "problématique" in chunk.page_content.lower():
                chunk.metadata["potential_research_question"] = True
        
        # Utiliser des embeddings optimisés
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}  # Pour de meilleurs résultats
        )
        
        if not chunks:
            logger.warning("Attention: Aucun chunk de document à indexer.")
            return None
        
        # Création de la base vectorielle
        db = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=self.db_path
        )
        
        logger.info(f"Base de données créée avec {len(chunks)} chunks")
        
        return db
    
    def setup_components(self, db):
        """
        Configure les composants nécessaires au workflow LangGraph
        
        Args:
            db: Base de données vectorielle Chroma
        """
        logger.info(f"Configuration des composants pour LangGraph avec {self.model_connector.get_name()}...")
        
        # Initialiser le modèle de langage
        self.llm = self.model_connector.get_llm(
            streaming=self.verbose
        )
        
        # Initialiser le classificateur de requêtes
        self.classifier = QueryClassifier(self.model_connector)
        
        self.vector_store = db
        
        # Créer les différents prompts nécessaires pour chaque étape du workflow
        self.prompts = {
            # Prompt pour la reformulation de requêtes
            "query_reformulation": PromptTemplate.from_template(
                """Tu es un assistant de recherche académique en sciences de l'éducation, expert en reformulation de requêtes.
                Ta mission est de reformuler la question initiale pour maximiser la pertinence des résultats.
                
                Question originale: {question}
                Type de requête identifié: {query_type}
                
                Reformule cette question pour un système de recherche documentaire académique en:
                1. Identifiant les concepts clés et leur terminologie académique équivalente
                2. Élargissant la question pour capturer des formulations alternatives
                3. Structurant la requête pour cibler précisément l'information recherchée
                
                La reformulation doit être optimisée pour le type de requête identifié.
                
                Reformulation:"""
            ),
            
            # Prompt pour l'analyse de pertinence des documents récupérés
            "relevance_assessment": PromptTemplate.from_template(
                """Tu es un expert en sciences de l'éducation qui évalue la pertinence des documents pour répondre à une question spécifique.
                
                Question: {question}
                Type de requête: {query_type}
                
                Voici les extraits de documents récupérés:
                {documents}
                
                Évalue si ces documents contiennent suffisamment d'informations pour répondre à la question.
                Analyse chaque document en fonction de sa pertinence pour le type de requête spécifié.
                
                1. Analyse de pertinence:
                2. Informations manquantes (le cas échéant):
                3. Faut-il affiner la recherche (oui/non):"""
            ),
            
            # Prompt pour la génération de réponse standard
            "standard_answer": PromptTemplate.from_template(
                """Tu es un assistant de recherche académique spécialisé en sciences de l'éducation, qui répond à des questions à partir d'une base documentaire d'études et de publications scientifiques.
                Tu dois extraire avec précision les informations importantes et toujours citer tes sources.

                Contexte documentaire:
                {context}

                Question: {question}

                Instructions détaillées:
                - Adopte une méthodologie rigoureuse et scientifique dans ta réponse
                - Réponds en te basant exclusivement sur le contexte fourni et sois précis
                - Examine attentivement le contexte pour trouver l'information pertinente, même si elle est présentée sous différentes formulations
                - Pour des questions comme "quelle est la question de recherche", cherche dans l'introduction, la problématique ou la méthodologie
                - Si tu trouves une information qui répond même partiellement à la question, partage-la
                - Cite systématiquement les sources précises (nom du fichier et numéro de page) pour chaque information
                - Si l'information est absolument absente du contexte, indique-le clairement sans spéculer
                - Structure ta réponse de façon académique avec une introduction, un développement organisé et une conclusion
                - Pour une synthèse, organise l'information chronologiquement ou thématiquement selon ce qui est le plus pertinent
                
                Réponse académique:"""
            ),
            
            # Prompt pour la génération de synthèse
            "synthesis_answer": PromptTemplate.from_template(
                """Tu es un chercheur expert en sciences de l'éducation qui doit réaliser une synthèse académique.
                
                Contexte documentaire:
                {context}
                
                Demande de synthèse: {question}
                
                Instructions pour la synthèse:
                - Organise la synthèse de façon thématique et cohérente
                - Identifie les idées clés, concepts et théories présents dans les documents
                - Mets en évidence les convergences et divergences entre les sources
                - Présente les résultats empiriques et leur signification
                - Structure ta synthèse avec une introduction, des sections thématiques et une conclusion
                - Cite systématiquement les sources (document et page) pour chaque élément mentionné
                
                Synthèse académique:"""
            ),
            
            # Prompt pour l'extraction de questions de recherche
            "research_question_answer": PromptTemplate.from_template(
                """Tu es un chercheur en sciences de l'éducation spécialiste de l'extraction de questions de recherche.
                
                Contexte documentaire:
                {context}
                
                Tâche: Identifie et cite exactement la question de recherche ou la problématique principale formulée dans le document.
                
                Instructions spécifiques:
                - Cherche dans l'introduction, le résumé, ou les sections méthodologiques
                - Cite littéralement la question de recherche telle qu'elle est formulée dans le document
                - Distingue la question principale des questions secondaires si elles existent
                - Inclus le contexte entourant la question pour en comprendre la portée
                - Cite précisément la source (document et page)
                
                Question de recherche identifiée:"""
            ),
            
            # Prompt pour l'élaboration complexe
            "elaboration_answer": PromptTemplate.from_template(
                """Tu es un expert académique en sciences de l'éducation chargé de développer une élaboration complexe et approfondie sur un sujet.
                
                Contexte documentaire:
                {context}
                
                Sujet d'élaboration: {question}
                
                Instructions pour l'élaboration:
                - Développe une analyse approfondie qui intègre les différentes perspectives présentes dans les documents
                - Articule les concepts théoriques avec les résultats empiriques
                - Présente une progression logique et une argumentation solide
                - Identifie les implications théoriques et pratiques
                - Structure ton élaboration avec une introduction, un développement détaillé avec sous-sections, et une conclusion
                - Cite systématiquement les sources (document et page) pour chaque élément mentionné
                
                Élaboration académique:"""
            ),
            
            # Prompt pour la comparaison
            "comparison_answer": PromptTemplate.from_template(
                """Tu es un chercheur en sciences de l'éducation spécialisé dans l'analyse comparative d'études et d'approches.
                
                Contexte documentaire:
                {context}
                
                Demande de comparaison: {question}
                
                Instructions pour la comparaison:
                - Identifie clairement les éléments à comparer (théories, méthodes, résultats, etc.)
                - Établis des critères de comparaison pertinents et explicites
                - Présente les similitudes et différences de manière structurée
                - Analyse les forces et limites de chaque approche
                - Structure ta comparaison de manière logique (par critère ou par élément comparé)
                - Cite systématiquement les sources (document et page) pour chaque élément mentionné
                
                Analyse comparative:"""
            ),
            
            # Prompt pour l'analyse méthodologique
            "methodology_answer": PromptTemplate.from_template(
                """Tu es un méthodologue expert en sciences de l'éducation chargé d'analyser les approches méthodologiques.
                
                Contexte documentaire:
                {context}
                
                Demande d'analyse méthodologique: {question}

                Instructions pour l'analyse:
                - Identifie précisément la méthodologie utilisée (qualitative, quantitative, mixte)
                - Décris les instruments de collecte de données et les procédures
                - Analyse la population/échantillon et les méthodes d'échantillonnage
                - Examine les techniques d'analyse de données
                - Évalue la rigueur méthodologique (validité, fiabilité, crédibilité)
                - Structure ton analyse méthodologique de façon cohérente
                - Cite systématiquement les sources (document et page) pour chaque élément mentionné
                
                Analyse méthodologique:"""
            )
        }

    def setup_nodes(self):
        """
        Configure les nœuds du graphe de workflow LangGraph
        
        Returns:
            Dictionnaire des fonctions de nœuds
        """
        # Nœud 1: Analyse initiale de la question
        def analyze_question(state):
            """Analyse la question et détermine le type de requête optimal"""
            state["current_step"] = "analyze_question"
            question = state["original_question"]
            
            # Si le type de requête est "auto", le déterminer automatiquement
            if state["query_type"] == "auto":
                query_type = self.classifier.classify(question)
                state["query_type"] = query_type
                
                if self.verbose:
                    logger.info(f"Type de requête détecté: {query_type}")
            
            state["thinking"] = f"Type de requête détecté: {state['query_type']}"
            state["steps_completed"].append("analyze_question")
            return state
        
        # Nœud 2: Reformulation de la question
        def reformulate_question(state):
            """Reformule la question pour améliorer la récupération"""
            state["current_step"] = "reformulate_question"
            question = state["original_question"]
            query_type = state["query_type"]
            
            try:
                # Utilisation du prompt de reformulation
                prompt = self.prompts["query_reformulation"]
                reformulated_question = self.llm.invoke(
                    prompt.format(
                        question=question,
                        query_type=QUERY_TYPES.get(query_type, "standard")
                    )
                ).strip()
                
                state["reformulated_question"] = reformulated_question
                state["thinking"] = f"{state.get('thinking', '')}\nQuestion reformulée: {reformulated_question}"
                
                if self.verbose:
                    logger.info(f"Question reformulée: {reformulated_question}")
            
            except Exception as e:
                error_msg = f"Erreur lors de la reformulation: {str(e)}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                # En cas d'erreur, utiliser la question originale
                state["reformulated_question"] = question
            
            state["steps_completed"].append("reformulate_question")
            return state
        
        # Nœud 3: Récupération des documents
        def retrieve_documents(state):
            """Récupère les documents pertinents en fonction du type de requête"""
            state["current_step"] = "retrieve_documents"
            query = state["reformulated_question"] or state["original_question"]
            query_type = state["query_type"]
            
            try:
                # Adapter la stratégie de recherche selon le type de requête
                search_kwargs = {"k": 8}
                filter_dict = {}
                
                if query_type == "synthesis" or query_type == "elaboration":
                    # Pour une synthèse ou élaboration, récupérer plus de documents
                    search_kwargs = {"k": 12}
                
                elif query_type == "research_question":
                    # Pour une question de recherche, favoriser l'introduction et les sections contenant des problématiques
                    query = "question de recherche problématique principale " + query
                    filter_dict = {"$or": [
                        {"document_section": "beginning"},
                        {"potential_research_question": True}
                    ]}
                    search_kwargs = {"k": 6, "filter": filter_dict}
                
                elif query_type == "conclusion":
                    # Pour une conclusion, favoriser la fin des documents
                    filter_dict = {"$or": [
                        {"document_section": "end"},
                        {"contains_conclusion": True}
                    ]}
                    search_kwargs = {"k": 6, "filter": filter_dict}
                
                elif query_type == "methodology":
                    # Pour une méthodologie, chercher les sections contenant des informations méthodologiques
                    query = "méthodologie méthode protocole " + query
                    filter_dict = {"contains_methodology": True}
                    search_kwargs = {"k": 8, "filter": filter_dict}
                
                # Récupérer les documents
                retriever = self.vector_store.as_retriever(
                    search_type="mmr",  # Maximum Marginal Relevance pour diversité
                    search_kwargs=search_kwargs
                )
                
                if self.verbose:
                    logger.info(f"Recherche avec la requête: {query}")
                    if filter_dict:
                        logger.info(f"Filtres appliqués: {filter_dict}")
                
                documents = retriever.invoke(query)
                
                # Formater les documents pour le logging
                doc_info = []
                for i, doc in enumerate(documents):
                    source = f"{doc.metadata.get('source_file', 'Unknown')} (p.{doc.metadata.get('page', '?')})"
                    preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    doc_info.append(f"Doc {i+1}: {source} - {preview}")
                
                state["retrieved_documents"] = documents
                state["thinking"] = f"{state.get('thinking', '')}\nDocuments récupérés: {len(documents)}\n" + "\n".join(doc_info[:3]) + "\n..."
                
                if self.verbose:
                    logger.info(f"Récupéré {len(documents)} documents")
                    for info in doc_info[:3]:
                        logger.info(info)
            
            except Exception as e:
                error_msg = f"Erreur lors de la récupération des documents: {str(e)}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["retrieved_documents"] = []
            
            state["steps_completed"].append("retrieve_documents")
            return state
        
        # Nœud 4: Évaluation de la pertinence des documents
        def assess_relevance(state):
            """Évalue si les documents récupérés sont pertinents pour la question"""
            state["current_step"] = "assess_relevance"
            
            if not state["retrieved_documents"]:
                state["needs_refinement"] = False
                state["thinking"] = f"{state.get('thinking', '')}\nAucun document récupéré, impossible d'évaluer la pertinence."
                state["steps_completed"].append("assess_relevance")
                return state
            
            try:
                question = state["reformulated_question"] or state["original_question"]
                query_type = state["query_type"]
                
                # Formater les documents pour l'évaluation
                doc_texts = []
                for i, doc in enumerate(state["retrieved_documents"][:5]):  # Limiter aux 5 premiers pour l'évaluation
                    source = f"{doc.metadata.get('source_file', 'Unknown')} (p.{doc.metadata.get('page', '?')})"
                    doc_texts.append(f"Document {i+1} ({source}):\n{doc.page_content[:300]}...\n")
                
                # Utiliser le prompt d'évaluation de pertinence
                prompt = self.prompts["relevance_assessment"]
                assessment = self.llm.invoke(
                    prompt.format(
                        question=question,
                        documents="\n".join(doc_texts),
                        query_type=QUERY_TYPES.get(query_type, "standard")
                    )
                )
                
                # Déterminer si un affinage est nécessaire
                needs_refinement = "oui" in assessment.lower().split("3. Faut-il affiner la recherche")[-1].lower()
                
                state["needs_refinement"] = needs_refinement
                state["thinking"] = f"{state.get('thinking', '')}\nÉvaluation de pertinence:\n{assessment}\nBesoins d'affinage: {needs_refinement}"
                
                if self.verbose:
                    logger.info(f"Évaluation de pertinence: {'Affinage nécessaire' if needs_refinement else 'Documents suffisants'}")
            
            except Exception as e:
                error_msg = f"Erreur lors de l'évaluation de pertinence: {str(e)}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["needs_refinement"] = False  # Par défaut, ne pas affiner en cas d'erreur
            
            state["steps_completed"].append("assess_relevance")
            return state
        
        # Nœud 5: Affinage de la recherche (si nécessaire)
        def refine_search(state):
            """Affine la recherche si nécessaire"""
            state["current_step"] = "refine_search"
            
            if not state["needs_refinement"]:
                state["thinking"] = f"{state.get('thinking', '')}\nAucun affinage nécessaire."
                state["steps_completed"].append("refine_search")
                return state
            
            try:
                # Enrichir la requête avec des termes supplémentaires basés sur l'évaluation
                original_question = state["reformulated_question"] or state["original_question"]
                
                # Ajouter des termes spécifiques selon le type de requête
                query_type = state["query_type"]
                enriched_terms = {
                    "standard": "détails précis information spécifique",
                    "synthesis": "concepts clés résultats analyse théorie",
                    "research_question": "objectif méthodologie problématique question",
                    "objective": "but finalité objectif ambition",
                    "conclusion": "résultat conclusion recommandation implications",
                    "elaboration": "développement analyse profondeur complexité",
                    "comparison": "différence similarité comparaison contraste",
                    "methodology": "méthode protocole approche procédure"
                }
                
                enriched_query = enriched_terms.get(query_type, "") + " " + original_question
                
                if self.verbose:
                    logger.info(f"Affinage de la recherche avec: {enriched_query}")
                
                # Récupérer des documents supplémentaires
                retriever = self.vector_store.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 5}  # Documents supplémentaires
                )
                
                additional_docs = retriever.invoke(enriched_query)
                
                # Ajouter les nouveaux documents uniques
                existing_ids = {doc.metadata.get("id") for doc in state["retrieved_documents"]}
                unique_new_docs = [doc for doc in additional_docs if doc.metadata.get("id") not in existing_ids]
                
                state["retrieved_documents"].extend(unique_new_docs)
                state["thinking"] = f"{state.get('thinking', '')}\nRecherche affinée: {len(unique_new_docs)} nouveaux documents ajoutés."
                
                if self.verbose:
                    logger.info(f"Affinage: {len(unique_new_docs)} nouveaux documents ajoutés")
            
            except Exception as e:
                error_msg = f"Erreur lors de l'affinage de la recherche: {str(e)}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
            
            state["steps_completed"].append("refine_search")
            return state
        
        # Nœud 6: Génération de la réponse
        def generate_answer(state):
            """Génère une réponse basée sur les documents récupérés"""
            state["current_step"] = "generate_answer"
            
            if not state["retrieved_documents"]:
                state["answer"] = "Je n'ai pas trouvé d'informations pertinentes pour répondre à cette question dans les documents disponibles."
                state["sources"] = []
                state["steps_completed"].append("generate_answer")
                return state
            
            try:
                question = state["original_question"]
                query_type = state["query_type"]
                
                # Sélectionner le prompt approprié selon le type de requête
                prompt_mapping = {
                    "standard": self.prompts["standard_answer"],
                    "synthesis": self.prompts["synthesis_answer"],
                    "research_question": self.prompts["research_question_answer"],
                    "objective": self.prompts["standard_answer"],  # Utiliser standard pour les objectifs
                    "conclusion": self.prompts["standard_answer"],  # Utiliser standard pour les conclusions
                    "elaboration": self.prompts["elaboration_answer"],
                    "comparison": self.prompts["comparison_answer"],
                    "methodology": self.prompts["methodology_answer"]
                }
                
                prompt_template = prompt_mapping.get(query_type, self.prompts["standard_answer"])
                
                if self.verbose:
                    logger.info(f"Génération de réponse avec prompt de type: {query_type}")
                
                # Préparer le contexte pour le LLM
                context = ""
                for i, doc in enumerate(state["retrieved_documents"]):
                    source = f"{doc.metadata.get('source_file', 'Unknown')} (p.{doc.metadata.get('page', '?')})"
                    context += f"\n--- Document {i+1}: {source} ---\n{doc.page_content}\n"
                
                # Générer la réponse
                answer = self.llm.invoke(
                    prompt_template.format(
                        context=context,
                        question=question
                    )
                )
                
                # Extraire les sources
                sources = []
                for doc in state["retrieved_documents"]:
                    file = doc.metadata.get("source_file", "Source inconnue")
                    page = doc.metadata.get("page", "?")
                    source = f"- {file}, page {page}"
                    if source not in sources:
                        sources.append(source)
                
                state["answer"] = answer
                state["sources"] = sources
                
                if self.verbose:
                    logger.info(f"Réponse générée ({len(answer)} caractères) avec {len(sources)} sources")
            
            except Exception as e:
                error_msg = f"Erreur lors de la génération de la réponse: {str(e)}"
                logger.error(error_msg)
                state["errors"].append(error_msg)
                state["answer"] = f"Désolé, une erreur s'est produite lors de la génération de la réponse: {str(e)}"
                state["sources"] = []
            
            state["steps_completed"].append("generate_answer")
            return state
        
        # Retourner tous les nœuds
        return {
            "analyze_question": analyze_question,
            "reformulate_question": reformulate_question,
            "retrieve_documents": retrieve_documents,
            "assess_relevance": assess_relevance,
            "refine_search": refine_search,
            "generate_answer": generate_answer
        }

    def define_workflow(self):
        """
        Définit le workflow LangGraph pour le traitement des requêtes
        """
        # Créer le graphe d'états
        workflow = StateGraph(QueryState)
        
        # Configurer les nœuds
        nodes = self.setup_nodes()
        
        # Ajouter les nœuds au graphe
        workflow.add_node("analyze_question", nodes["analyze_question"])
        workflow.add_node("reformulate_question", nodes["reformulate_question"])
        workflow.add_node("retrieve_documents", nodes["retrieve_documents"])
        workflow.add_node("assess_relevance", nodes["assess_relevance"])
        workflow.add_node("refine_search", nodes["refine_search"])
        workflow.add_node("generate_answer", nodes["generate_answer"])
        
        # Définir les transitions entre les nœuds
        workflow.add_edge("analyze_question", "reformulate_question")
        workflow.add_edge("reformulate_question", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "assess_relevance")
        
        # Transition conditionnelle basée sur la pertinence
        workflow.add_conditional_edges(
            "assess_relevance",
            lambda state: "refine_search" if state["needs_refinement"] else "generate_answer",
            {
                "refine_search": "refine_search",
                "generate_answer": "generate_answer"
            }
        )
        
        workflow.add_edge("refine_search", "generate_answer")
        workflow.add_edge("generate_answer", END)
        
        # Définir l'état initial
        workflow.set_entry_point("analyze_question")
        
        # Activer le traçage pour la visualisation si demandé
        if ENABLE_TRACING and LANGGRAPH_API_KEY:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                from langgraph.tracing import get_tracer
                
                # Configurer le traçage
                tracer = get_tracer(
                    project_name="academic_assistant",
                    api_key=LANGGRAPH_API_KEY
                )
                
                # Configurer un saver pour la persistance
                saver = SqliteSaver(db_file="./academic_assistant_traces.db")
                
                # Compiler le workflow avec traçage et persistance
                self.workflow = workflow.compile(
                    checkpointer=saver,
                    tracer=tracer
                )
                
                logger.info("Workflow compilé avec traçage LangGraph activé")
            except Exception as e:
                logger.warning(f"Impossible d'activer le traçage LangGraph: {e}")
                self.workflow = workflow.compile()
        else:
            # Compiler le workflow sans traçage
            self.workflow = workflow.compile()
            
        logger.info("Workflow LangGraph configuré et compilé")

    def initialize(self):
        """
        Initialise l'assistant académique complet
        
        Returns:
            bool: True si l'initialisation a réussi, False sinon
        """
        try:
            logger.info("Démarrage de l'initialisation de l'assistant académique...")
            
            # Vérifier si une base de données existe déjà
            db_exists = os.path.exists(self.db_path) and os.listdir(self.db_path)
            
            if db_exists:
                logger.info(f"Base de données existante trouvée à {self.db_path}")
                # Charger directement la base existante
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
                db = Chroma(
                    persist_directory=self.db_path,
                    embedding_function=embeddings
                )
                logger.info(f"Base de données chargée avec {db._collection.count()} chunks")
            else:
                # Procéder à l'indexation complète
                logger.info("Aucune base de données trouvée, création d'une nouvelle base...")
                documents = self.load_documents()
                
                # Vérifier si des documents ont été chargés
                if not documents:
                    logger.error("Aucun document n'a été chargé.")
                    return False
                    
                db = self.prepare_database(documents)
                
                # Vérifier si la base de données a été créée
                if not db:
                    logger.error("Impossible de créer la base de données vectorielle.")
                    return False
            
            self.setup_components(db)
            self.define_workflow()
            
            logger.info(f"Assistant académique 3.0 prêt avec {self.model_connector.get_name()}!")
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def ask(self, question, query_type="auto"):
        """
        Pose une question à l'assistant académique
        
        Args:
            question: Question de l'utilisateur
            query_type: Type de requête ('auto', 'standard', 'synthesis', 'research_question', etc.)
            
        Returns:
            Dictionnaire contenant la réponse et les métadonnées
        """
        if not self.workflow:
            raise ValueError("L'assistant n'a pas été initialisé. Appelez initialize() d'abord.")
        
        try:
            # Enregistrer la question
            logger.info(f"Question reçue: {question}")
            
            # Générer un ID de session unique
            session_id = str(uuid.uuid4())
            
            # Initialiser l'état
            initial_state = {
                "session_id": session_id,
                "original_question": question,
                "reformulated_question": None,
                "query_type": query_type,
                "retrieved_documents": None,
                "additional_context": None,
                "answer": None,
                "sources": None,
                "thinking": None,
                "needs_refinement": False,
                "start_time": time.time(),
                "steps_completed": [],
                "current_step": "",
                "errors": [],
                "verbose": self.verbose
            }
            
            # Exécuter le workflow
            final_state = self.workflow.invoke(initial_state)
            
            # Calculer le temps d'exécution
            execution_time = time.time() - final_state["start_time"]
            
            # Préparer la réponse
            response = {
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "query_type": final_state["query_type"],
                "reformulated_question": final_state["reformulated_question"],
                "thinking": final_state["thinking"],
                "execution_time": execution_time,
                "steps_completed": final_state["steps_completed"],
                "errors": final_state["errors"]
            }
            
            # Enregistrer la session pour référence future
            self.session_history[session_id] = {
                "timestamp": datetime.datetime.now().isoformat(),
                "question": question,
                "query_type": query_type,
                "execution_time": execution_time,
                "steps_completed": final_state["steps_completed"],
                "has_errors": len(final_state["errors"]) > 0
            }
            
            logger.info(f"Réponse générée en {execution_time:.2f} secondes")
            
            return response
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "answer": f"Erreur lors du traitement de la question: {str(e)}",
                "sources": [],
                "query_type": query_type,
                "execution_time": time.time() - initial_state["start_time"] if 'initial_state' in locals() else 0,
                "errors": [str(e)],
                "error": True
            }
    
    def get_model_info(self):
        """
        Retourne les informations sur le modèle utilisé
        
        Returns:
            Dictionnaire contenant les informations sur le modèle
        """
        return {
            "model_type": self.model_type,
            "model_name": self.model_connector.get_name(),
            "is_local": self.model_connector.is_local,
            "max_tokens": self.model_connector.max_tokens
        }
    
    def get_documents_info(self):
        """
        Retourne les informations sur les documents indexés
        
        Returns:
            Dictionnaire contenant les informations sur les documents
        """
        if not self.vector_store:
            return {"status": "non_initialized"}
        
        try:
            total_chunks = self.vector_store._collection.count() if hasattr(self.vector_store, '_collection') else 0
            
            return {
                "total_documents": len(self.document_sources),
                "document_list": list(self.document_sources.keys()),
                "total_chunks": total_chunks,
                "documents_detail": self.document_sources
            }
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des informations sur les documents: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_session_history(self, limit=10):
        """
        Retourne l'historique des sessions récentes
        
        Args:
            limit: Nombre maximum de sessions à retourner
            
        Returns:
            Liste des sessions récentes
        """
        # Trier les sessions par timestamp (la plus récente d'abord)
        sorted_sessions = sorted(
            self.session_history.items(),
            key=lambda x: x[1]["timestamp"],
            reverse=True
        )
        
        # Limiter le nombre de sessions retournées
        return dict(sorted_sessions[:limit])
    
    def export_graph_visualization(self, output_file="workflow_graph.html"):
        """
        Génère une visualisation du graphe de workflow
        
        Args:
            output_file: Chemin du fichier HTML de sortie
            
        Returns:
            bool: True si la visualisation a été générée avec succès, False sinon
        """
        try:
            # Vérifier si le workflow a été initialisé
            if not hasattr(self, 'workflow') or not self.workflow:
                logger.error("Le workflow n'a pas été initialisé.")
                return False
            
            # Essayer d'importer les bibliothèques nécessaires
            try:
                from pyvis.network import Network
                import networkx as nx
            except ImportError:
                logger.error("Les bibliothèques pyvis et networkx sont nécessaires pour la visualisation.")
                logger.error("Installez-les avec: pip install pyvis networkx")
                return False
            
            # Créer un graphe NetworkX depuis notre workflow
            G = nx.DiGraph()
            
            # Ajouter les nœuds
            nodes = ["analyze_question", "reformulate_question", "retrieve_documents", 
                     "assess_relevance", "refine_search", "generate_answer", "END"]
            
            for node in nodes:
                G.add_node(node)
            
            # Ajouter les arêtes
            edges = [
                ("analyze_question", "reformulate_question"),
                ("reformulate_question", "retrieve_documents"),
                ("retrieve_documents", "assess_relevance"),
                ("assess_relevance", "refine_search"),
                ("assess_relevance", "generate_answer"),
                ("refine_search", "generate_answer"),
                ("generate_answer", "END")
            ]
            
            for edge in edges:
                G.add_edge(*edge)
            
            # Créer la visualisation avec pyvis
            net = Network(notebook=False, height="800px", width="100%", directed=True)
            
            # Personnaliser l'apparence
            net.from_nx(G)
            
            # Configurer les options
            net.set_options("""
            var options = {
                "physics": {
                    "forceAtlas2Based": {
                        "gravitationalConstant": -50,
                        "centralGravity": 0.01,
                        "springLength": 100,
                        "springConstant": 0.08
                    },
                    "solver": "forceAtlas2Based"
                },
                "nodes": {
                    "font": {"size": 16, "face": "Tahoma"},
                    "shape": "box"
                },
                "edges": {
                    "arrows": {"to": {"enabled": true}}
                }
            }
            """)
            
            # Enregistrer la visualisation
            net.save_graph(output_file)
            logger.info(f"Visualisation du graphe enregistrée dans {output_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la visualisation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False


# Interface en ligne de commande pour tester
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Assistant académique 3.0 avec LangGraph')
    parser.add_argument('--folder', '-f', required=True, help='Dossier contenant les PDFs')
    parser.add_argument('--model_type', '-mt', default='ollama', choices=['ollama', 'claude'], 
                        help='Type de modèle à utiliser')
    parser.add_argument('--model_name', '-m', help='Nom du modèle spécifique à utiliser')
    parser.add_argument('--db_path', '-d', default='./chroma_db', help='Chemin de la base de données vectorielle')
    parser.add_argument('--verbose', '-v', action='store_true', help='Mode verbeux')
    parser.add_argument('--api_key', '-k', help='Clé API (pour Claude)')
    parser.add_argument('--visualize', '-vis', action='store_true', help='Générer une visualisation du workflow')
    
    args = parser.parse_args()
    
    try:
        # Créer l'assistant avec les paramètres spécifiés
        assistant = AcademicAssistant(
            docs_folder=args.folder, 
            model_type=args.model_type,
            model_name=args.model_name,
            db_path=args.db_path,
            verbose=args.verbose,
            api_key=args.api_key
        )
        
        init_success = assistant.initialize()
        if not init_success:
            logger.error("Initialisation échouée. Arrêt du programme.")
            return
        
        # Générer la visualisation si demandé
        if args.visualize:
            vis_success = assistant.export_graph_visualization()
            if vis_success:
                print("Visualisation du workflow générée avec succès.")
            else:
                print("Échec de la génération de la visualisation du workflow.")
        
        print("\n===== Assistant Académique 3.0 avec LangGraph =====")
        print(f"Modèle: {assistant.get_model_info()['model_name']}")
        print("Posez vos questions ou tapez 'exit' pour quitter")
        
        while True:
            print("\n")
            question = input("Votre question (ou 'exit' pour quitter): ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
                
            query_type = input("Type de requête [standard/synthesis/research_question/objective/conclusion/elaboration/methodology/comparison/auto]: ").lower() or "auto"
            if query_type not in QUERY_TYPES and query_type != "auto":
                print(f"Type de requête non reconnu: {query_type}. Utilisation de 'auto'.")
                query_type = "auto"
                
            try:
                # Mesurer le temps d'exécution
                start_time = time.time()
                
                # Obtenir la réponse
                response = assistant.ask(question, query_type)
                
                # Afficher le temps d'exécution
                total_time = time.time() - start_time
                print(f"\nTraitement effectué en {total_time:.2f} secondes")
                
                # Afficher le type de requête détecté
                if query_type == "auto":
                    print(f"Type de requête détecté: {response['query_type']}")
                
                # Afficher la réponse
                print("\n------ Réponse ------")
                if "reformulated_question" in response and response["reformulated_question"]:
                    print(f"\nQuestion reformulée: {response['reformulated_question']}")
                print(f"\n{response['answer']}")
                
                # Afficher les sources
                print("\n------ Sources ------")
                for source in response["sources"]:
                    print(source)
                
                # Pour le débogage
                if args.verbose:
                    print("\n------ Raisonnement interne ------")
                    print(response.get("thinking", "Pas de raisonnement disponible"))
                    
                    print("\n------ Étapes complétées ------")
                    for step in response["steps_completed"]:
                        print(f"- {step}")
                    
                    # Afficher les erreurs s'il y en a
                    if response.get("errors"):
                        print("\n------ Erreurs ------")
                        for error in response["errors"]:
                            print(f"! {error}")
                    
            except Exception as e:
                print(f"Erreur: {e}")
                logger.error(f"Erreur non gérée: {e}")
                import traceback
                logger.error(traceback.format_exc())
    
    except Exception as e:
        logger.error(f"Erreur générale: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    import sys
    main()