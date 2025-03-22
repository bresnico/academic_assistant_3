# model_connectors.py
from typing import Dict, Any, Optional, List, Union
from abc import ABC, abstractmethod
import os
import logging
from typing import Any, Optional
from langchain_anthropic import ChatAnthropic
from langchain_ollama import OllamaLLM
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema import Document
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelConnector(ABC):
    """Interface abstraite pour les connecteurs de modèles"""
    
    @abstractmethod
    def get_llm(self, **kwargs) -> BaseLanguageModel:
        """Retourne une instance du modèle de langage configuré"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Retourne le nom du modèle"""
        pass
    
    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Indique si le modèle est exécuté localement"""
        pass

    @property
    @abstractmethod
    def max_tokens(self) -> int:
        """Retourne le nombre maximum de tokens supportés par le modèle"""
        pass


class OllamaConnector(ModelConnector):
    """Connecteur pour les modèles Ollama (Deepseek, Llama, etc.)"""
    
    def __init__(self, model_name: str = "deepseek-r1:8b", temperature: float = 0.1):
        """
        Initialise le connecteur Ollama
        
        Args:
            model_name: Nom du modèle Ollama à utiliser (ex: deepseek-r1:8b, llama3.1:8b)
            temperature: Température pour la génération (0.0 à 1.0)
        """
        self.model_name = model_name
        self.temperature = temperature
        self._max_tokens = {
            "deepseek-r1:8b": 4096,
            "deepseek-r1:10b": 8192,
            "deepseek-coder:33b": 16384,
            "llama3.1:8b": 8192,
            "llama3.1:70b": 8192,
            "gemma3:12b": 8192,
        }.get(model_name, 4096)  # Valeur par défaut si le modèle n'est pas reconnu
    
    def get_llm(self, streaming: bool = False, **kwargs) -> OllamaLLM:
        """
        Retourne une instance du modèle Ollama configuré
        
        Args:
            streaming: Activer le streaming de la sortie
            **kwargs: Arguments supplémentaires pour la configuration
        
        Returns:
            Instance configurée de OllamaLLM
        """
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        
        return OllamaLLM(
            model=self.model_name,
            temperature=self.temperature,
            num_ctx=self._max_tokens,
            callbacks=callbacks,
            **kwargs
        )
    
    def get_name(self) -> str:
        """Retourne le nom du modèle Ollama"""
        return f"Ollama:{self.model_name}"
    
    @property
    def is_local(self) -> bool:
        """Les modèles Ollama sont exécutés localement"""
        return True
    
    @property
    def max_tokens(self) -> int:
        """Retourne le nombre maximum de tokens supportés par le modèle"""
        return self._max_tokens


class ClaudeConnector(ModelConnector):
    """Connecteur pour les modèles Claude via l'API Anthropic"""
    
    def __init__(self, model_name: str = "claude-3-7-sonnet-20250219", api_key: Optional[str] = None, temperature: float = 0.1):
        """
        Initialise le connecteur Claude
        
        Args:
            model_name: Nom du modèle Claude à utiliser
            api_key: Clé API Anthropic (si None, elle sera recherchée dans les variables d'environnement)
            temperature: Température pour la génération (0.0 à 1.0)
        """
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
        self.temperature = temperature
        
        if not self.api_key:
            logger.warning("Aucune clé API Anthropic trouvée. Veuillez définir la variable d'environnement ANTHROPIC_API_KEY ou CLAUDE_API_KEY.")
        
        # Définir les limites de tokens pour les différents modèles Claude
        self._max_tokens = {
            "claude-3-7-sonnet-20250219": 200000,
            "claude-3-5-sonnet-20240229": 200000,
            "claude-3-opus-20240229": 200000,
            "claude-3-5-haiku-20240307": 200000,
        }.get(model_name, 100000)  # Valeur par défaut si le modèle n'est pas reconnu
    
    def get_llm(self, streaming: bool = False, **kwargs) -> Any:
        """
        Retourne une instance du modèle Claude configuré avec un wrapper pour gérer les réponses
        
        Args:
            streaming: Activer le streaming de la sortie
            **kwargs: Arguments supplémentaires pour la configuration
        
        Returns:
            Instance configurée de ChatAnthropic avec wrapper
        """
        if not self.api_key:
            raise ValueError("Clé API Anthropic non définie. Impossible d'initialiser Claude.")
        
        callbacks = [StreamingStdOutCallbackHandler()] if streaming else None
        
        model = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            anthropic_api_key=self.api_key,
            callbacks=callbacks,
            **kwargs
        )
        
        # Créer un wrapper pour extraire le contenu des messages automatiquement
        class ClaudeWrapper:
            def __init__(self, model):
                self.model = model
                
            def invoke(self, prompt):
                response = self.model.invoke(prompt)
                # Extraire le contenu texte du message AI
                if hasattr(response, 'content'):
                    # Si c'est une liste (format multi-modal possible)
                    if isinstance(response.content, list):
                        # Concaténer les éléments textuels
                        text_content = ""
                        for item in response.content:
                            if hasattr(item, 'text'):
                                text_content += item.text
                            elif isinstance(item, str):
                                text_content += item
                        return text_content
                    # Si c'est une chaîne directement
                    elif isinstance(response.content, str):
                        return response.content
                    # Pour tout autre type
                    else:
                        return str(response.content)
                # Fallback: convertir le message entier en chaîne
                return str(response)
                
        # Retourner le wrapper au lieu du modèle directement
        return ClaudeWrapper(model)
    
    def get_name(self) -> str:
        """Retourne le nom du modèle Claude"""
        return f"Claude:{self.model_name}"
    
    @property
    def is_local(self) -> bool:
        """Les modèles Claude ne sont pas exécutés localement"""
        return False
    
    @property
    def max_tokens(self) -> int:
        """Retourne le nombre maximum de tokens supportés par le modèle"""
        return self._max_tokens

def create_model_connector(model_type: str, model_name: Optional[str] = None, **kwargs) -> ModelConnector:
    """
    Fonction utilitaire pour créer le connecteur de modèle approprié
    
    Args:
        model_type: Type de modèle ('ollama' ou 'claude')
        model_name: Nom du modèle à utiliser
        **kwargs: Arguments supplémentaires pour la configuration du modèle
    
    Returns:
        Une instance de ModelConnector appropriée
    """
    model_type = model_type.lower()
    
    if model_type == "ollama":
        default_model = "deepseek-r1:8b"
        return OllamaConnector(model_name=model_name or default_model, **kwargs)
    elif model_type == "claude":
        default_model = "claude-3-7-sonnet-20250219"
        return ClaudeConnector(model_name=model_name or default_model, **kwargs)
    else:
        raise ValueError(f"Type de modèle non pris en charge: {model_type}. Utilisez 'ollama' ou 'claude'.")