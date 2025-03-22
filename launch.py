#!/usr/bin/env python3
# launch.py
import os
import sys
import subprocess
import argparse
import logging
import platform
from typing import Optional, List, Dict

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies() -> List[str]:
    """Vérifie les dépendances requises et retourne celles qui manquent"""
    required_packages = [
        "langchain", "langchain_community", "langchain_huggingface", 
        "langchain_chroma", "langchain_text_splitters", "langchain_ollama",
        "chromadb", "langgraph", "streamlit", "plotly", "pyvis", "networkx",
        "sentence-transformers", "matplotlib", "pandas", "pydantic",
        "typing", "pypdf", "langchain_anthropic"
    ]
    
    missing_packages = []
    
    # Vérifier les packages Python
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    # Vérifier Ollama (si on est sur un système Unix-like)
    if platform.system() != "Windows":
        try:
            result = subprocess.run(["which", "ollama"], capture_output=True, text=True)
            if result.returncode != 0:
                missing_packages.append("ollama")
        except Exception:
            missing_packages.append("ollama")
    
    return missing_packages

def install_dependencies(missing_packages: List[str]) -> bool:
    """Installe les dépendances manquantes"""
    if not missing_packages:
        return True
    
    logger.info(f"Installation des dépendances manquantes: {', '.join(missing_packages)}")
    
    # Installer les packages Python manquants
    python_packages = [pkg for pkg in missing_packages if pkg != "ollama"]
    if python_packages:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", *python_packages], check=True)
            logger.info("Packages Python installés avec succès.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Erreur lors de l'installation des packages Python: {e}")
            return False
    
    # Instructions pour Ollama si manquant
    if "ollama" in missing_packages:
        logger.warning("Ollama n'est pas installé. Veuillez l'installer manuellement:")
        logger.warning("Linux/macOS: curl -fsSL https://ollama.com/install.sh | sh")
        logger.warning("Plus d'informations: https://ollama.com/download")
    
    return True

def setup_documents_folder(folder_path: str) -> None:
    """Crée le dossier de documents s'il n'existe pas"""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            logger.info(f"Dossier de documents créé: {folder_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la création du dossier de documents: {e}")

def launch_cli(docs_folder: str, model_type: str, model_name: Optional[str] = None, verbose: bool = False) -> None:
    """Lance l'assistant en mode CLI"""
    from academic_assistant import main
    
    args = ["--folder", docs_folder]
    
    if model_type:
        args.extend(["--model_type", model_type])
    
    if model_name:
        args.extend(["--model_name", model_name])
    
    if verbose:
        args.append("--verbose")
    
    # Remplacer sys.argv pour main()
    sys.argv = [sys.argv[0]] + args
    main()

def launch_streamlit(docs_folder: str) -> None:
    """Lance l'interface Streamlit"""
    try:
        cmd = [
            "streamlit", "run", "streamlit_interface.py", 
            "--server.port", "8501",
            "--"
        ]
        
        # Ajouter les arguments pour streamlit_interface.py
        if docs_folder:
            cmd.extend(["--folder", docs_folder])
        
        logger.info(f"Lancement de Streamlit avec la commande: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Erreur lors du lancement de Streamlit: {e}")
    except Exception as e:
        logger.error(f"Erreur inattendue: {e}")

def get_anthropic_api_key() -> Optional[str]:
    """Récupère la clé API Anthropic depuis l'environnement ou un fichier de configuration"""
    # Vérifier d'abord dans les variables d'environnement
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
    
    # Si aucune clé n'est trouvée, vérifier dans un fichier de configuration
    if not api_key:
        config_paths = [
            os.path.expanduser("~/.academic_assistant_config"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        ]
        
        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    logger.info(f"Lecture du fichier de configuration: {config_path}")
                    with open(config_path, 'r') as f:
                        for line in f:
                            if line.strip().startswith("ANTHROPIC_API_KEY=") or line.strip().startswith("CLAUDE_API_KEY="):
                                api_key = line.split('=', 1)[1].strip().strip('"').strip("'")
                                break
                except Exception as e:
                    logger.warning(f"Erreur lors de la lecture du fichier de configuration {config_path}: {e}")
    
    # Si une clé est trouvée, la définir dans l'environnement
    if api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key
        logger.info("Clé API Anthropic configurée")
    
    return api_key

def main() -> None:
    # Récupérer la clé API Anthropic
    get_anthropic_api_key()
    
    parser = argparse.ArgumentParser(description="Lanceur pour l'Assistant Académique 3.0")
    
    # Options communes
    parser.add_argument("--docs", "-d", default="./documents", help="Dossier contenant les PDFs")
    parser.add_argument("--skip-check", action="store_true", help="Ignore la vérification des dépendances")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mode verbeux")
    
    # Sous-parseurs pour les différents modes
    subparsers = parser.add_subparsers(dest="mode", help="Mode de lancement")
    
    # Mode CLI
    cli_parser = subparsers.add_parser("cli", help="Lancer en mode ligne de commande")
    cli_parser.add_argument("--model-type", "-mt", default="ollama", choices=["ollama", "claude"], help="Type de modèle à utiliser")
    cli_parser.add_argument("--model-name", "-m", help="Nom du modèle spécifique")
    
    # Mode Web (Streamlit)
    web_parser = subparsers.add_parser("web", help="Lancer l'interface web Streamlit")
    
    args = parser.parse_args()
    
    # Vérifier les dépendances
    if not args.skip_check:
        missing_packages = check_dependencies()
        if missing_packages:
            success = install_dependencies(missing_packages)
            if not success:
                logger.error("Impossible d'installer toutes les dépendances. Veuillez les installer manuellement.")
                sys.exit(1)
    
    # Créer le dossier de documents s'il n'existe pas
    setup_documents_folder(args.docs)
    
    # Lancer dans le mode approprié
    if args.mode == "cli":
        launch_cli(args.docs, args.model_type, args.model_name, args.verbose)
    elif args.mode == "web":
        launch_streamlit(args.docs)
    else:
        # Mode par défaut: poser la question à l'utilisateur
        print("=== Assistant Académique 3.0 ===")
        print("Choisissez le mode de lancement:")
        print("1. Interface en ligne de commande")
        print("2. Interface web (Streamlit)")
        choice = input("Votre choix (1/2): ")
        
        if choice == "1":
            print("\nDémarrage de l'interface en ligne de commande...")
            model_type = input("Type de modèle (ollama/claude) [ollama]: ") or "ollama"
            launch_cli(args.docs, model_type, None, args.verbose)
        else:
            print("\nDémarrage de l'interface web Streamlit...")
            launch_streamlit(args.docs)

if __name__ == "__main__":
    main()