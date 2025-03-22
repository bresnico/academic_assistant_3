# streamlit_interface.py
import streamlit as st
import os
import glob
import pandas as pd
import time
import matplotlib.pyplot as plt
import datetime
from academic_assistant import AcademicAssistant
from query_classifier import QUERY_TYPES
import logging
import json
import sys
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import base64
import io
import tempfile

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("academic_assistant.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Assistant Académique 3.0",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/votre-repo/academic-assistant',
        'Report a bug': "https://github.com/votre-repo/academic-assistant/issues",
        'About': "# Assistant Académique 3.0\nPropulsé par LangGraph et divers modèles de langage."
    }
)

# Fonctions utilitaires
def get_pdf_files(folder_path):
    """Récupère la liste des fichiers PDF dans un dossier"""
    if not os.path.exists(folder_path):
        return []
    return glob.glob(os.path.join(folder_path, "*.pdf"))

def format_sources(sources):
    """Formate les sources pour l'affichage"""
    if not sources:
        return ""
    return "\n".join(sources)

def visualize_graph_execution(thinking):
    """Crée une visualisation du flux d'exécution LangGraph"""
    if not thinking:
        return "Aucune donnée de raisonnement disponible"
    
    steps = thinking.split("\n")
    return "\n".join([f"- {step}" for step in steps if step.strip()])

def get_model_options():
    """Retourne les options de modèles disponibles"""
    ollama_models = [
        "deepseek-r1:8b", 
        "deepseek-r1:10b", 
        "llama3.1:8b", 
        "llama3.1:70b", 
        "gemma3:12b",
        "deepseek-coder:33b"
    ]
    
    claude_models = [
        "claude-3-7-sonnet-20250219",
        "claude-3-5-sonnet-20240229",
        "claude-3-opus-20240229",
        "claude-3-5-haiku-20240307"
    ]
    
    return {
        "ollama": ollama_models,
        "claude": claude_models
    }

def initialize_assistant(docs_folder, model_type, model_name, db_path, verbose, api_key=None, reindex=False):
    """Initialise ou réinitialise l'assistant"""
    try:
        if reindex and os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            st.sidebar.success("Base de données vectorielle réinitialisée.")
        
        assistant = AcademicAssistant(
            docs_folder=docs_folder,
            model_type=model_type,
            model_name=model_name,
            db_path=db_path,
            verbose=verbose,
            api_key=api_key
        )
        
        with st.spinner("Initialisation de l'assistant en cours..."):
            init_success = assistant.initialize()
        
        if not init_success:
            st.error("Échec de l'initialisation de l'assistant. Vérifiez les logs pour plus de détails.")
            return None
        
        return assistant
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation: {e}")
        st.error(f"Erreur: {str(e)}")
        return None

def get_query_type_description(query_type):
    """Retourne la description d'un type de requête"""
    return QUERY_TYPES.get(query_type, "Type de requête non reconnu")

def create_workflow_visualization():
    """Crée une visualisation simplifiée du workflow LangGraph"""
    # Créer un diagramme Plotly
    fig = go.Figure()
    
    # Définir les nœuds et leurs positions
    nodes = [
        {"name": "analyze_question", "x": 0, "y": 0, "description": "Analyse de la question"},
        {"name": "reformulate_question", "x": 1, "y": 0, "description": "Reformulation"},
        {"name": "retrieve_documents", "x": 2, "y": 0, "description": "Récupération des documents"},
        {"name": "assess_relevance", "x": 3, "y": 0, "description": "Évaluation de pertinence"},
        {"name": "refine_search", "x": 4, "y": -0.5, "description": "Affinement de la recherche"},
        {"name": "generate_answer", "x": 4, "y": 0.5, "description": "Génération de réponse"},
        {"name": "END", "x": 5, "y": 0, "description": "Fin du processus"}
    ]
    
    # Créer des marqueurs pour les nœuds
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]],
            y=[node["y"]],
            mode="markers+text",
            marker=dict(size=30, color="skyblue", line=dict(width=2, color="black")),
            text=[node["description"]],
            textposition="bottom center",
            name=node["name"],
            hoverinfo="text",
            hovertext=node["description"]
        ))
    
    # Définir les arêtes
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5), (5, 6)
    ]
    
    # Ajouter les arêtes au diagramme
    for edge in edges:
        start_node = nodes[edge[0]]
        end_node = nodes[edge[1]]
        
        fig.add_trace(go.Scatter(
            x=[start_node["x"], end_node["x"]],
            y=[start_node["y"], end_node["y"]],
            mode="lines",
            line=dict(width=2, color="gray"),
            hoverinfo="none",
            showlegend=False
        ))
    
    # Configurer la mise en page
    fig.update_layout(
        title="Workflow de l'Assistant Académique",
        showlegend=False,
        hovermode="closest",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        margin=dict(l=20, r=20, t=60, b=20),
        height=300
    )
    
    return fig

def highlight_active_steps(fig, active_steps):
    """Met en évidence les étapes actives dans la visualisation"""
    # Modifier la couleur des nœuds actifs
    for i, node in enumerate(fig.data[:7]):  # Les 7 premiers traces sont des nœuds
        if node.name in active_steps:
            fig.data[i].marker.color = "green"
        else:
            fig.data[i].marker.color = "skyblue"
    
    return fig

# Titre de l'application
st.title("📚 Assistant de Recherche Académique 3.0")
st.caption("Propulsé par LangGraph, LangChain et IA multi-modèles")

# Barre latérale pour les contrôles
with st.sidebar:
    st.header("Configuration")
    
    # Configuration du dossier de documents
    docs_folder = st.text_input(
        "Dossier des documents",
        value="./documents",
        help="Chemin vers le dossier contenant vos PDFs"
    )
    
    # Sélection du type de modèle
    model_type = st.selectbox(
        "Type de modèle",
        options=["ollama", "claude"],
        index=0,
        help="Choisir entre Ollama (local) ou Claude (API)"
    )
    
    # Obtenir les options de modèles disponibles
    model_options = get_model_options()
    
    # Sélection du modèle spécifique
    model_name = st.selectbox(
        f"Modèle {model_type.capitalize()}",
        options=model_options[model_type],
        index=0,
        help=f"Modèle {model_type} spécifique à utiliser"
    )
    
    # Clé API (uniquement pour Claude)
    api_key = None
    if model_type == "claude":
        # Vérifier si une clé existe déjà dans l'environnement
        env_api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY") or ""
        
        api_key = st.text_input(
            "Clé API Anthropic",
            value=env_api_key,  # Pré-remplir avec la clé d'environnement si elle existe
            type="password",
            help="Clé API Anthropic pour accéder à Claude"
        )
        
        # Si une clé a été saisie, la définir dans l'environnement pour les utilisations futures
        if api_key and api_key != env_api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            
        if not api_key and not env_api_key:
            st.warning("Une clé API est nécessaire pour utiliser Claude.")
    
    # Chemin de la base de données
    db_path = st.text_input(
        "Chemin de la base de données",
        value="./chroma_db_3.0",
        help="Emplacement où stocker les embeddings"
    )
    
    # Mode verbeux
    verbose = st.checkbox(
        "Mode verbeux", 
        value=False,
        help="Afficher des informations détaillées pendant le traitement"
    )
    
    # Bouton pour réindexer les documents
    reindex = st.checkbox(
        "Réindexer les documents",
        value=False, 
        help="Cochez pour recréer la base de données vectorielle"
    )
    
    # Bouton d'initialisation
    if st.button("Initialiser l'Assistant"):
        st.session_state.assistant = initialize_assistant(
            docs_folder, model_type, model_name, db_path, verbose, api_key, reindex
        )
        
        if st.session_state.assistant:
            st.success("Assistant initialisé avec succès!")
            # Récupérer la liste des PDFs et les informations sur le modèle
            pdf_files = get_pdf_files(docs_folder)
            st.session_state.pdf_files = [os.path.basename(pdf) for pdf in pdf_files]
            st.session_state.model_info = st.session_state.assistant.get_model_info()
            
            # Générer la visualisation du workflow
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
                vis_success = st.session_state.assistant.export_graph_visualization(temp_file.name)
                if vis_success:
                    st.session_state.workflow_visualization_path = temp_file.name
                    st.success("Visualisation du workflow générée avec succès.")
                else:
                    st.warning("Impossible de générer la visualisation du workflow.")
            except Exception as e:
                st.warning(f"Erreur de visualisation: {str(e)}")
    
    # Afficher les documents disponibles
    st.subheader("Documents disponibles")
    if 'pdf_files' in st.session_state and st.session_state.pdf_files:
        for pdf in st.session_state.pdf_files:
            st.write(f"- {pdf}")
    else:
        pdf_files = get_pdf_files(docs_folder)
        if pdf_files:
            st.session_state.pdf_files = [os.path.basename(pdf) for pdf in pdf_files]
            for pdf in st.session_state.pdf_files:
                st.write(f"- {pdf}")
        else:
            st.warning(f"Aucun PDF trouvé dans {docs_folder}")
    
    # Afficher les informations sur le modèle
    if 'model_info' in st.session_state:
        st.subheader("Informations sur le modèle")
        st.info(f"Modèle: {st.session_state.model_info['model_name']}")
        st.info(f"Exécution: {'Locale' if st.session_state.model_info['is_local'] else 'API distante'}")
        st.info(f"Capacité: {st.session_state.model_info['max_tokens']} tokens")

# Zone principale - divisée en deux colonnes
col1, col2 = st.columns([3, 2])

with col1:
    # Zone de conversation
    st.header("Conversation")
    
    # Initialiser l'historique s'il n'existe pas
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Afficher l'historique des messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("Sources"):
                    st.markdown(message["sources"])
            if message["role"] == "assistant" and "thinking" in message and st.session_state.get("show_thinking", False):
                with st.expander("Processus de raisonnement"):
                    st.markdown(message["thinking"])
            if message["role"] == "assistant" and "steps_completed" in message and st.session_state.get("show_thinking", False):
                with st.expander("Étapes complétées"):
                    st.markdown("\n".join([f"- {step}" for step in message["steps_completed"]]))
            if message["role"] == "assistant" and "execution_time" in message:
                st.caption(f"Temps d'exécution: {message['execution_time']:.2f} secondes")
    
    # Zone de saisie
    if prompt := st.chat_input("Posez votre question académique..."):
        # Ajouter la question à l'historique
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Afficher la question
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Vérifier si l'assistant est initialisé
        if 'assistant' not in st.session_state or st.session_state.assistant is None:
            with st.chat_message("assistant"):
                st.warning("Veuillez initialiser l'assistant d'abord en cliquant sur le bouton dans la barre latérale.")
            st.session_state.messages.append({
                "role": "assistant",
                "content": "Veuillez initialiser l'assistant d'abord en cliquant sur le bouton dans la barre latérale."
            })
        else:
            # Options de requête
            with col2:
                st.subheader("Type de requête")
                query_type = st.radio(
                    "Sélectionnez le type de requête:",
                    ["auto"] + list(QUERY_TYPES.keys()),
                    index=0,
                    horizontal=True
                )
                
                if query_type != "auto":
                    st.info(get_query_type_description(query_type))
                else:
                    st.info("Mode automatique: l'assistant déterminera le type de requête le plus approprié.")
            
            # Traiter la réponse
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("⏳ Je réfléchis...")
                
                try:
                    # Obtenir la réponse de l'assistant
                    start_time = time.time()
                    response = st.session_state.assistant.ask(prompt, query_type)
                    total_time = time.time() - start_time
                    
                    # Mise à jour de la visualisation du workflow pour montrer les étapes actives
                    if 'workflow_fig' in st.session_state:
                        st.session_state.workflow_fig = highlight_active_steps(
                            st.session_state.workflow_fig,
                            response.get("steps_completed", [])
                        )
                    
                    # Afficher la réponse
                    message_placeholder.markdown(response["answer"])
                    
                    # Afficher les sources
                    with st.expander("Sources"):
                        st.markdown(format_sources(response["sources"]))
                    
                    # Afficher la reformulation (si disponible)
                    if "reformulated_question" in response and response["reformulated_question"]:
                        with st.expander("Question reformulée"):
                            st.markdown(f"**Reformulation:** {response['reformulated_question']}")
                    
                    # Afficher le type de requête détecté (si auto)
                    if query_type == "auto" and "query_type" in response:
                        detected_type = response["query_type"]
                        with st.expander("Type de requête détecté"):
                            st.markdown(f"**Type détecté:** {detected_type}")
                            st.markdown(f"**Description:** {get_query_type_description(detected_type)}")
                    
                    # Afficher le raisonnement (si activé)
                    if "thinking" in response and st.session_state.get("show_thinking", False):
                        with st.expander("Processus de raisonnement"):
                            st.markdown(response["thinking"])
                    
                    # Afficher les étapes complétées (si activé)
                    if "steps_completed" in response and st.session_state.get("show_thinking", False):
                        with st.expander("Étapes complétées"):
                            st.markdown("\n".join([f"- {step}" for step in response["steps_completed"]]))
                    
                    # Afficher les erreurs s'il y en a
                    if response.get("errors") and len(response["errors"]) > 0:
                        with st.expander("Erreurs"):
                            st.error("\n".join([f"- {error}" for error in response["errors"]]))
                    
                    # Afficher le temps d'exécution
                    st.caption(f"Temps d'exécution: {response.get('execution_time', total_time):.2f} secondes")
                    
                    # Ajouter à l'historique
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": format_sources(response["sources"]),
                        "thinking": response.get("thinking", ""),
                        "steps_completed": response.get("steps_completed", []),
                        "execution_time": response.get("execution_time", total_time),
                        "errors": response.get("errors", [])
                    })
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de la question: {e}")
                    message_placeholder.markdown(f"⚠️ Erreur: {str(e)}")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"⚠️ Erreur: {str(e)}"
                    })

with col2:
    # Zone d'informations et d'outils
    st.header("Outils et Visualisations")
    
    # Option pour effacer l'historique
    if st.button("Effacer l'historique"):
        st.session_state.messages = []
        st.experimental_rerun()
    
    # Option pour afficher le raisonnement
    st.session_state.show_thinking = st.checkbox(
        "Afficher les détails du raisonnement", 
        value=st.session_state.get("show_thinking", False),
        help="Affiche les étapes intermédiaires du raisonnement de l'assistant"
    )
    
    # Visualisation du workflow LangGraph
    st.subheader("Workflow LangGraph")
    
    # Créer une visualisation simple du workflow
    if 'workflow_fig' not in st.session_state:
        st.session_state.workflow_fig = create_workflow_visualization()
    
    # Afficher la visualisation
    st.plotly_chart(st.session_state.workflow_fig, use_container_width=True)
    
    # Afficher la visualisation complète si disponible
    if 'workflow_visualization_path' in st.session_state:
        with st.expander("Visualisation avancée du workflow"):
            with open(st.session_state.workflow_visualization_path, 'r') as file:
                html_content = file.read()
                st.components.v1.html(html_content, height=600)
    
    # Outil d'exploration des documents
    st.subheader("Exploration des documents")
    if 'pdf_files' in st.session_state and st.session_state.pdf_files:
        selected_pdf = st.selectbox(
            "Sélectionnez un document",
            options=st.session_state.pdf_files
        )
        
        if selected_pdf:
            st.write(f"Document sélectionné: {selected_pdf}")
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("Extraire les points clés"):
                    if 'assistant' in st.session_state and st.session_state.assistant:
                        try:
                            # Demande automatique de points clés
                            with st.spinner("Extraction des points clés en cours..."):
                                response = st.session_state.assistant.ask(
                                    f"Fais une analyse des points clés du document '{selected_pdf}'. Extrais la problématique, les objectifs, la méthodologie et les conclusions principales.",
                                    query_type="synthesis"
                                )
                                
                                # Afficher les résultats
                                st.markdown("### Points clés du document")
                                st.markdown(response["answer"])
                                
                                with st.expander("Sources"):
                                    st.markdown(format_sources(response["sources"]))
                        except Exception as e:
                            st.error(f"Erreur lors de l'extraction des points clés: {str(e)}")
            
            with col_b:
                # Option pour générer automatiquement des questions
                if st.button("Suggérer des questions"):
                    if 'assistant' in st.session_state and st.session_state.assistant:
                        try:
                            with st.spinner("Génération de questions..."):
                                suggestion_prompt = f"En te basant sur le document '{selected_pdf}', génère 5 questions pertinentes qu'un chercheur en sciences de l'éducation pourrait se poser. Numérote les questions."
                                response = st.session_state.assistant.ask(suggestion_prompt)
                                
                                st.markdown("### Questions suggérées")
                                st.markdown(response["answer"])
                        except Exception as e:
                            st.error(f"Erreur lors de la génération de questions: {str(e)}")
    
    # Statistiques d'utilisation
    if 'assistant' in st.session_state and hasattr(st.session_state.assistant, 'get_session_history'):
        st.subheader("Statistiques d'utilisation")
        
        # Récupérer l'historique des sessions
        history = st.session_state.assistant.get_session_history()
        
        if history:
            # Créer un compteur pour les types de requêtes
            query_type_counts = {}
            for session_id, session in history.items():
                query_type = session.get("query_type", "unknown")
                if query_type in query_type_counts:
                    query_type_counts[query_type] += 1
                else:
                    query_type_counts[query_type] = 1
            
            # Créer un graphique des types de requêtes
            if query_type_counts:
                fig = px.pie(
                    names=list(query_type_counts.keys()),
                    values=list(query_type_counts.values()),
                    title="Distribution des types de requêtes"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Afficher les statistiques de temps d'exécution
            execution_times = []
            for session_id, session in history.items():
                if "execution_time" in session:
                    execution_times.append(session["execution_time"])
            
            if execution_times:
                avg_time = sum(execution_times) / len(execution_times)
                max_time = max(execution_times)
                min_time = min(execution_times)
                
                st.metric("Temps moyen de réponse", f"{avg_time:.2f}s")
                
                col_x, col_y = st.columns(2)
                with col_x:
                    st.metric("Temps minimum", f"{min_time:.2f}s")
                with col_y:
                    st.metric("Temps maximum", f"{max_time:.2f}s")
            
            # Afficher le nombre total de questions
            st.metric("Questions posées", len(history))
            
            # Afficher le nombre d'erreurs
            error_count = sum(1 for session in history.values() if session.get("has_errors", False))
            st.metric("Sessions avec erreurs", error_count)

# Pied de page
st.markdown("---")
st.caption("Assistant Académique 3.0 | Développé avec LangGraph, LangChain et Streamlit")
st.caption("© 2025 - Tous droits réservés")