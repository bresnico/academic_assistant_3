# Assistant Académique 3.0

Un assistant de recherche académique intelligent, propulsé par LangGraph, LangChain et les grands modèles de langage.

## 🚀 Fonctionnalités

- **Moteur d'Intelligence Hybride**: Choix entre Ollama (local) et Claude API (cloud)
- **Classification Automatique de Requêtes**: Détection intelligente du type de question posée
- **Workflow Optimisé**: Architecture LangGraph pour un traitement séquentiel et conditionnel
- **Visualisation du Raisonnement**: Exploration visuelle du cheminement de l'assistant
- **Mode Verbeux**: Détails complets du processus de réflexion
- **Interface Double**: Ligne de commande ou application web Streamlit
- **Analyse de Documents PDF**: Extraction et exploration de contenu académique

## 📋 Prérequis

- Python 3.9 ou supérieur
- [Ollama](https://ollama.com/download) (pour les modèles locaux)
- Clé API Anthropic (pour Claude)

## 🔧 Installation

1. **Cloner le dépôt**

```bash
git clone https://github.com/votre-repo/academic-assistant.git
cd academic-assistant
```

2. **Installer les dépendances**

Vous pouvez utiliser le script de lancement qui installera automatiquement les dépendances requises:

```bash
python launch.py
```

Ou les installer manuellement:

```bash
pip install langchain langchain_community langchain_huggingface langchain_chroma
pip install langchain_text_splitters langchain_ollama chromadb langgraph
pip install streamlit plotly pyvis networkx sentence-transformers matplotlib pandas pydantic
```

3. **Installer et configurer Ollama** (Pour les modèles locaux)

Suivez les instructions sur [ollama.com/download](https://ollama.com/download)

Téléchargez les modèles que vous souhaitez utiliser:

```bash
ollama pull deepseek-r1:8b
ollama pull llama3.1:8b
```

## 🔑 Configuration de la clé API Anthropic

Pour utiliser Claude, vous devez configurer une clé API Anthropic. Plusieurs méthodes sont disponibles :

1. **Variable d'environnement**:
   - Définissez `ANTHROPIC_API_KEY` ou `CLAUDE_API_KEY` dans votre environnement
   - Exemple: `export ANTHROPIC_API_KEY=votre_clé_api_ici`

2. **Fichier de configuration global**:
   - Créez un fichier `.academic_assistant_config` dans votre répertoire utilisateur
   - Ajoutez la ligne: `ANTHROPIC_API_KEY=votre_clé_api_ici`

3. **Fichier de configuration local**:
   - Créez un fichier `.env` dans le répertoire du projet (vous pouvez utiliser `.env.example` comme modèle)
   - Ajoutez la ligne: `ANTHROPIC_API_KEY=votre_clé_api_ici`

4. **Interface Streamlit**:
   - Saisissez votre clé API dans l'interface web
   - Elle sera automatiquement enregistrée dans l'environnement pour les utilisations futures

L'assistant cherchera la clé dans cet ordre et l'utilisera automatiquement quand vous sélectionnez Claude comme modèle.

## 💻 Utilisation

### Lancer l'application

```bash
python launch.py
```

Vous serez invité à choisir entre l'interface en ligne de commande ou l'interface web Streamlit.

### Lancer directement l'interface en ligne de commande

```bash
python launch.py cli --docs ./votre-dossier-documents --model-type ollama --model-name deepseek-r1:8b
```

Options disponibles:
- `--docs`: Dossier contenant les documents PDF (par défaut: ./documents)
- `--model-type`: Type de modèle (ollama ou claude)
- `--model-name`: Nom du modèle spécifique
- `--verbose`: Mode verbeux

### Lancer directement l'interface web Streamlit

```bash
python launch.py web --docs ./votre-dossier-documents
```

L'interface web sera accessible à l'adresse [http://localhost:8501](http://localhost:8501)

## 🧠 Types de Requêtes Supportés

- **standard**: Recherche d'informations spécifiques et précises
- **synthesis**: Demande de synthèse ou résumé sur un thème/sujet
- **research_question**: Identification de questions ou problématiques de recherche
- **objective**: Recherche d'objectifs ou buts d'une étude
- **conclusion**: Extraction de conclusions ou résultats d'une étude
- **elaboration**: Élaboration complexe basée sur plusieurs documents
- **comparison**: Comparaison entre différentes études ou approches
- **methodology**: Analyse de la méthodologie utilisée dans une étude
- **auto**: Détection automatique du type de requête (par défaut)

## 📄 Structure du Projet

- `academic_assistant.py`: Module principal avec l'implémentation LangGraph
- `model_connectors.py`: Connecteurs pour les différents modèles (Ollama, Claude)
- `query_classifier.py`: Classification automatique des types de requêtes
- `streamlit_interface.py`: Interface utilisateur web avec Streamlit
- `launch.py`: Script de lancement unifié

## 📊 Visualisation du Workflow

L'assistant utilise un workflow LangGraph structuré avec les étapes suivantes:

1. **Analyse de la question**: Détermination du type de requête
2. **Reformulation**: Optimisation de la requête pour la recherche
3. **Récupération de documents**: Recherche dans la base de connaissances
4. **Évaluation de pertinence**: Analyse de la qualité des résultats
5. **Affinement de recherche** (conditionnel): Amélioration des résultats si nécessaire
6. **Génération de réponse**: Production d'une réponse académique précise

## 📚 Ajouter des Documents

Placez vos fichiers PDF dans le dossier `documents` (ou celui spécifié via l'option `--docs`).

L'assistant indexera automatiquement ces documents lors de la première exécution.

## 🔄 Mise à Jour de la Base de Connaissances

Pour réindexer les documents (après avoir ajouté de nouveaux fichiers):

- **Interface CLI**: Utilisez l'option `--reindex` lors du lancement
- **Interface Web**: Cochez la case "Réindexer les documents" dans la barre latérale

## 🛠️ Configuration Avancée

### Variables d'Environnement

- `ANTHROPIC_API_KEY`: Clé API pour Claude (principale)
- `CLAUDE_API_KEY`: Alternative pour la clé API Claude
- `LANGGRAPH_API_KEY`: Clé API pour la visualisation LangGraph (facultatif)
- `ENABLE_TRACING`: Activer le traçage LangGraph (true/false)

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier `LICENSE` pour plus de détails.

## 🙏 Remerciements

- [LangChain](https://www.langchain.com/) pour le framework d'orchestration
- [LangGraph](https://www.langchain.com/langgraph) pour le moteur de workflow
- [Anthropic](https://www.anthropic.com/) pour Claude
- [Ollama](https://ollama.com/) pour l'exécution locale de modèles