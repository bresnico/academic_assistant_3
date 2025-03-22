# query_classifier.py
from typing import Dict, List, Tuple, Optional, Any
import re
import logging
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from model_connectors import ModelConnector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Types de requêtes supportés
QUERY_TYPES = {
    "standard": "Recherche d'informations spécifiques et précises",
    "synthesis": "Demande de synthèse ou résumé sur un thème/sujet",
    "research_question": "Identification de questions ou problématiques de recherche",
    "objective": "Recherche d'objectifs ou buts d'une étude",
    "conclusion": "Extraction de conclusions ou résultats d'une étude",
    "elaboration": "Élaboration complexe basée sur plusieurs documents",
    "comparison": "Comparaison entre différentes études ou approches",
    "methodology": "Analyse de la méthodologie utilisée dans une étude"
}

class QueryClassifier:
    """
    Classe pour classifier automatiquement les requêtes utilisateur
    """
    
    def __init__(self, model_connector: ModelConnector):
        """
        Initialise le classificateur de requêtes
        
        Args:
            model_connector: Connecteur vers le modèle de langage à utiliser
        """
        self.model_connector = model_connector
        self.pattern_mapping = {
            r"synth[èe]se|r[ée]sum[ée]|synth[ée]tiser": "synthesis",
            r"question\s+de\s+recherche|probl[ée]matique|hypoth[èe]se": "research_question",
            r"objectif|but|finalit[ée]": "objective",
            r"conclusion|r[ée]sultat|trouv[ée]|d[ée]couverte": "conclusion",
            r"m[ée]thodologie|m[ée]thode|protocole|proc[ée]dure": "methodology",
            r"compar(e|er|aison)|diff[ée]ren(t|ce)|versus|vs\.?|par rapport": "comparison",
            r"[ée]labor(er|ation)|d[ée]velopp(er|ement)|approfondi": "elaboration"
        }
        
        # Prompt pour l'analyse avancée avec LLM
        self.llm_classification_prompt = PromptTemplate.from_template(
            """Tu es un expert en classification de requêtes académiques.
            
            Tu dois déterminer le type de requête parmi les catégories suivantes:
            - standard: Recherche d'informations spécifiques et précises
            - synthesis: Demande de synthèse ou résumé sur un thème/sujet
            - research_question: Identification de questions ou problématiques de recherche
            - objective: Recherche d'objectifs ou buts d'une étude
            - conclusion: Extraction de conclusions ou résultats d'une étude
            - elaboration: Élaboration complexe basée sur plusieurs documents
            - comparison: Comparaison entre différentes études ou approches
            - methodology: Analyse de la méthodologie utilisée dans une étude
            
            Voici la requête à classifier:
            ```
            {query}
            ```
            
            N'explique pas ton raisonnement. Réponds uniquement avec le type de requête qui correspond le mieux.
            """
        )
    
    def simple_classify(self, query: str) -> str:
        """
        Classification basée sur des règles simples (expressions régulières)
        
        Args:
            query: Requête utilisateur à classifier
        
        Returns:
            Type de requête identifié
        """
        query_lower = query.lower()
        
        # Recherche par motifs
        for pattern, query_type in self.pattern_mapping.items():
            if re.search(pattern, query_lower):
                logger.debug(f"Motif '{pattern}' détecté pour la classification '{query_type}'")
                return query_type
        
        # Par défaut, considérer comme une requête standard
        return "standard"
    
    def llm_classify(self, query: str) -> str:
        """
        Classification avancée utilisant le LLM
        
        Args:
            query: Requête utilisateur à classifier
        
        Returns:
            Type de requête identifié
        """
        try:
            # Obtenir le LLM à partir du connecteur
            llm = self.model_connector.get_llm()
            
            # Générer le prompt
            prompt = self.llm_classification_prompt.format(query=query)
            
            # Obtenir la classification du LLM
            response = llm.invoke(prompt).strip().lower()
            
            # Extraire le type de requête de la réponse (au cas où le LLM inclut du texte supplémentaire)
            for query_type in QUERY_TYPES.keys():
                if query_type in response:
                    logger.debug(f"LLM a classifié la requête comme '{query_type}'")
                    return query_type
            
            # Si aucun type reconnu n'est trouvé dans la réponse, utiliser la méthode simple
            logger.warning(f"Classification LLM non reconnue: '{response}', on utilise la méthode simple")
            return self.simple_classify(query)
            
        except Exception as e:
            logger.error(f"Erreur lors de la classification LLM: {e}")
            # En cas d'erreur, fallback sur la méthode simple
            return self.simple_classify(query)
    
    def classify(self, query: str, use_llm: bool = True) -> str:
        """
        Classifie une requête utilisateur en déterminant son type
        
        Args:
            query: Requête utilisateur à classifier
            use_llm: Utiliser le LLM pour une classification plus précise
        
        Returns:
            Type de requête identifié
        """
        logger.info(f"Classification de la requête: '{query}'")
        
        if use_llm:
            return self.llm_classify(query)
        else:
            return self.simple_classify(query)