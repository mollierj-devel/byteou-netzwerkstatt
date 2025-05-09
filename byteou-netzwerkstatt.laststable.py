#!/usr/bin/env python3
"""
byteou-netzwerkstatt

Génère des résumés et des concepts "à la Zettelkasten" depuis les transcriptions de vidéos YouTube.
"""

import argparse
import os
import re
import sys
import yaml
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Bibliothèques externes requises
try:
    from openai import OpenAI
    import google.generativeai as genai
    from youtube_transcript_api import YouTubeTranscriptApi
    import yt_dlp
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
    from loguru import logger
except ImportError:
    print("Certaines bibliothèques requises ne sont pas installées.")
    sys.exit(1)

# Configuration du logger
def setup_logger(verbosity: int = 0) -> None:
    """Configure le logger avec le niveau de verbosité spécifié."""
    logger.remove()  # Supprimer les handlers par défaut
    
    # Définir le niveau de verbosité
    levels = {
        0: "WARNING",
        1: "INFO",
        2: "DEBUG",
        3: "TRACE"
    }
    level = levels.get(verbosity, "TRACE")
    
    # Format selon la verbosité
    format_simple = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
    format_detailed = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    log_format = format_detailed if verbosity >= 2 else format_simple
    
    # Ajouter le handler avec le niveau et format appropriés
    logger.add(sys.stderr, level=level, format=log_format)
    logger.info(f"Logger configuré au niveau {level}")

# Classe de configuration
class Config:
    """Gestionnaire de configuration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialise la configuration à partir d'un fichier YAML."""
        self.config_path = config_path
        self.config = {}
        self.prompts = {}
        
        # Charger la configuration ou utiliser les valeurs par défaut
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                self.config = yaml.safe_load(file)
                logger.info(f"Configuration chargée depuis {config_path}")
                logger.trace(f"Contenu de la configuration: {self.config}")
        except Exception as e:
            logger.warning(f"Impossible de charger la configuration: {e}")
            # Configuration par défaut
            self.config = {
                "openai": {
                    "api_key": "API_KEY",
                    "base_url": "https://api.openai.com/v1/",
                    "model": "gpt-4o"
                },
                "gemini": {
                    "api_key": "",
                    "model": "gemini-2.0-flash"
                },
                "ai": {
                    "provider": "openai"  # par défaut
                },
                "youtube": {
                    "languages": ["fr", "en"]
                },
                "prompts_dir": "prompts"
            }
            logger.warning("Utilisation de la configuration par défaut")
            logger.trace(f"Configuration par défaut: {self.config}")
    
    def load_prompts(self, prompts_dir: Optional[str] = None) -> None:
        """Charge les prompts depuis le répertoire spécifié."""
        dir_path = prompts_dir or self.config.get("prompts_dir", "prompts")
        logger.debug(f"Chargement des prompts depuis {dir_path}")
        
        try:
            prompt_path = Path(dir_path)
            if not prompt_path.exists():
                logger.warning(f"Le répertoire {dir_path} n'existe pas, création des prompts par défaut")
                self._create_default_prompts(dir_path)
                
            # Charger tous les fichiers .txt du répertoire
            for file_path in Path(dir_path).glob("*.txt"):
                prompt_name = file_path.stem.upper()
                with open(file_path, "r", encoding="utf-8") as file:
                    self.prompts[f"PROMPT_{prompt_name}"] = file.read().strip()
                logger.debug(f"Prompt chargé: {prompt_name}")
                logger.trace(f"Contenu du prompt {prompt_name}: {self.prompts[f'PROMPT_{prompt_name}'][:100]}...")
                
            logger.info(f"{len(self.prompts)} prompts chargés")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des prompts: {e}")
            self._create_default_prompts(dir_path)
    
    def _create_default_prompts(self, dir_path: str) -> None:
        """Crée les prompts par défaut si le répertoire n'existe pas."""
        try:
            os.makedirs(dir_path, exist_ok=True)
            
            # Créer les prompts par défaut
            prompts = {
                "synthese.txt": """
Crée un document de synthèse.
Pour cela :
1. Analyse l'ensemble des documents fournis sur le sujet.
2. Identifie les concepts clés, les arguments principaux et les informations essentielles présents dans ces documents.
3. Organise ces informations de manière cohérente selon les thématiques suivantes :
   - Résumé en deux phrases.
   - Définitions et concepts fondamentaux
   - Principales problématiques identifiées
   - Solutions ou approches proposées
   - Points de consensus et de désaccord entre les sources
4. Présente une synthèse structurée qui :
   - Maintient l'objectivité
   - Cite précisément les sources pour chaque information importante
   - Met en évidence les relations entre les différentes idées
   - Identifie les zones d'incertitude ou les questions non résolues
1. Termine par 5 hashtags en français. Un seul mot correspondant au contenu. La mise en forme doit être une simple liste de hashtags séparés par un espace. Si un seul mot ne permet pas d'être suffisamment précis, concatène le. Si un acronyme est important, utilise le.
Format souhaité : document Markdown structuré avec titres et sous-titres
                """,
                
                "zettelkasten.txt": """
Tu es spécialiste de la méthode Zettelkasten et tu es un expert à identifier les concepts.
Ton rôle est d'identifier dans le transcript joint les concepts. Dans un premier temps, fais-moi une liste des concepts que tu trouves et ajoute une citation et mets une phrase d'explication. 
Fais des bons titres. Termine chaque concept par 5 hashtags et 5 propositions alternatives de titres.
Formate le résultat dans un style Markdown, chaque concept commence avec zzCONCEPT **IDEE** sans , ou **IDEE** est une phrase courte ne contenant aucun caractère spécial résumant l'idée comme des "API". Elle doit pouvoir résumer et représenter l'ensemble du concept de manière abstraite et réutilisable.
Voici les consignes que tu dois respecter pour trouver les titres et les concepts à mettre dans les notes permanentes :
Pour identifier et formuler efficacement les concepts :
1. Créer des titres qui fonctionnent comme des "API" - Le titre doit pouvoir résumer et représenter l'ensemble du concept de manière abstraite et réutilisable.
2. Orienter les notes par concept plutôt que par source - Privilégier l'organisation par idées/concepts plutôt que par auteur, livre ou projet. Cela permet de :
    - Découvrir des connexions inattendues entre différentes sources
    - Accumuler et synthétiser les connaissances sur un même concept
    - Approfondir la compréhension en confrontant différentes perspectives
3. Utiliser des phrases complètes comme titres :
    - **Préférer les phrases déclaratives ou impératives qui formulent une affirmation claire**
    - Les questions peuvent aussi servir de titres si elles ciblent le cœur du sujet
    - Le titre doit y systematiquement et obligatoirement avoir un **verbe** dans le titre
    - La casse dois être en minuscules
    - Exceptions: définitions de termes clés, plans/outlines
4. Critères de qualité d'un concept bien identifié :
    - Il est atomique (une seule idée centrale)
    - Il peut être relié densément à d'autres concepts
    - Son titre résume clairement sa contribution
    - Il permet l'accumulation progressive des connaissances 
                """,
                
                "notes.txt": """
TU es un modèle de langage d'IA capable de prendre des notes détaillées, concises et faciles à comprendre sur divers sujets sous forme de puces. 
Votre format de sortie est markdown et vous devez commencer par "# Notes enrichies." 
Lorsqu'on vous fournit des **notes :** sur un **document :**, votre tâche est de :
- Créez des notes avancées résumant les parties importantes du **document :** en te basant seulement sur chaque concept ou thème de la **notes :**.
- Incluez toutes les informations essentielles, telles que les termes de vocabulaire et les concepts clés, qui doivent être mises en gras avec des astérisques.
- Supprimez tout langage superflu, en vous concentrant uniquement sur les aspects critiques du passage dans le **document :**.
- Basez strictement vos notes sur les informations fournies dans le **document :**., sans ajouter d'informations externes.
- assure toi que tous les concepts ou thèmes de la **notes :** sont traité avant de terminer.
                """
            }
            
            # Écrire les prompts dans des fichiers
            for name, content in prompts.items():
                with open(f"{dir_path}/{name}", "w", encoding="utf-8") as file:
                    file.write(content.strip())
                # Ajouter également au dictionnaire de prompts
                self.prompts[f"PROMPT_{name.split('.')[0].upper()}"] = content.strip()
                
            logger.info(f"Prompts par défaut créés dans {dir_path}")
        except Exception as e:
            logger.error(f"Erreur lors de la création des prompts par défaut: {e}")
    
    def get(self, key: str, default=None):
        """Récupère une valeur de configuration."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def get_prompt(self, name: str) -> str:
        """Récupère un prompt par son nom."""
        prompt_key = f"PROMPT_{name.upper()}"
        if prompt_key in self.prompts:
            return self.prompts[prompt_key]
        else:
            logger.error(f"Prompt non trouvé: {name}")
            return ""
    

    @property
    def gemini_api_key(self) -> str:
        """Récupère la clé API Google Gemini."""
        return self.get("gemini.api_key", "")
    
    @property
    def gemini_model(self) -> str:
        """Récupère le modèle Google Gemini à utiliser."""
        return self.get("gemini.model", "gemini-2.0-flash")
    
    @property
    def ai_provider(self) -> str:
        """Récupère le fournisseur d'IA à utiliser (openai ou gemini)."""
        return self.get("ai.provider", "openai").lower()
    
    @property
    def openai_api_key(self) -> str:
        """Récupère la clé API OpenAI."""
        return self.get("openai.api_key", "")
    
    @property
    def openai_base_url(self) -> str:
        """Récupère l'URL de base de l'API OpenAI."""
        return self.get("openai.base_url", "https://api.openai.com/v1/")
    
    @property
    def openai_model(self) -> str:
        """Récupère le modèle OpenAI à utiliser."""
        return self.get("openai.model", "gpt-4o")
    
    @property
    def youtube_languages(self) -> List[str]:
        """Récupère les langues préférées pour les transcriptions YouTube."""
        return self.get("youtube.languages", ["fr", "en"])

# Téléchargement des transcriptions
class TranscriptDownloader:
    """Gestionnaire de téléchargement des transcriptions YouTube."""
    
    def __init__(self, languages: List[str]):
        """Initialise avec les langues préférées."""
        self.languages = languages
        logger.trace(f"TranscriptDownloader initialisé avec les langues: {languages}")

    def get_transcript(self, video_id: str) -> str:
        """Télécharge la transcription d'une vidéo YouTube avec retry automatique."""
        logger.debug(f"Téléchargement de la transcription pour la vidéo {video_id}")
        
        # Paramètres du retry
        max_attempts = 3
        base_delay = 2  # secondes
        
        # Délai initial pour laisser le temps de charger les métadonnées
        initial_delay = 1  # secondes
        logger.debug(f"Attente de {initial_delay}s avant le chargement des métadonnées")
        time.sleep(initial_delay)
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Tentative {attempt}/{max_attempts} de récupération de la transcription pour {video_id}")
                
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
                full_text = " ".join([item['text'] for item in transcript])
                
                logger.debug(f"Transcription récupérée avec succès: {len(full_text)} caractères")
                logger.trace(f"Début de la transcription: {full_text[:100]}...")
                return full_text
                
            except Exception as e:
                delay = base_delay * (2 ** (attempt - 1))  # Backoff exponentiel
                logger.warning(f"Échec de la tentative {attempt}/{max_attempts}: {e}")
                logger.debug(f"Détails de l'erreur: {type(e).__name__}, {str(e)}")
                
                if attempt < max_attempts:
                    logger.info(f"Nouvelle tentative dans {delay} secondes...")
                    time.sleep(delay)
                else:
                    logger.error(f"Échec après {max_attempts} tentatives pour {video_id}")
                    logger.error(f"Exception finale: {e}", exc_info=True)
                    raise Exception(f"Impossible de récupérer la transcription après {max_attempts} tentatives: {e}")
    
    def save_transcript(self, video_id: str, output_dir: str) -> str:
        """Télécharge et sauvegarde la transcription d'une vidéo YouTube."""
        logger.info(f"Téléchargement et sauvegarde de la transcription pour {video_id}")
        
        try:
            transcript_text = self.get_transcript(video_id)
            
            # Assurer que le répertoire de sortie existe
            os.makedirs(output_dir, exist_ok=True)
            
            # Créer le chemin du fichier de sortie
            output_file = Path(output_dir) / f"{video_id}.transcript.txt"
            
            # Sauvegarder la transcription
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(transcript_text)
            
            logger.info(f"Transcription sauvegardée dans {output_file}")
            return str(output_file)
        
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la transcription: {e}")
            raise Exception(f"Impossible de sauvegarder la transcription pour {video_id}: {e}")
        
    def get_video_metadata(self, video_id: str) -> Dict[str, str]:
        """
        Récupère les métadonnées d'une vidéo YouTube (titre, auteur/chaîne).
        
        Args:
            video_id: Identifiant de la vidéo YouTube.
        
        Returns:
            Dictionnaire contenant les métadonnées de la vidéo.
        """
        logger.debug(f"Récupération des métadonnées pour la vidéo {video_id}")
        
        try:
            # Construire l'URL de la vidéo
            url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Options pour yt-dlp
            ytdl_options = {
                'quiet': True,
                'skip_download': True,
                'force_json': True
            }
            
            # Extraction des métadonnées
            with yt_dlp.YoutubeDL(ytdl_options) as ytdl:
                info = ytdl.extract_info(url, download=False)
                
            metadata = {
                "title": info['title'],
                "author": info['channel'],
                "channel_id": info.get('channel_id', ''),
                "publish_date": info.get('upload_date', None),
                "views": info.get('view_count', 0),
                "length_seconds": info.get('duration', 0),
                "description": info.get('description', ''),
                "tags": info.get('tags', [])
            }
            
            logger.info(f"Métadonnées récupérées: {metadata['title']} par {metadata['author']}")
            logger.trace(f"Métadonnées complètes: {metadata}")
            
            return metadata
            
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des métadonnées: {e}")
            # Retourner des métadonnées minimales en cas d'échec
            return {
                "title": f"Video {video_id}",
                "author": "Unknown",
                "error": str(e)
            }    
    


# Génération de contenu avec l'IA
class AIGenerator:
    """Générateur de contenu basé sur différents fournisseurs d'IA (OpenAI ou Gemini)."""
    
    def __init__(self, provider: str, api_key: str, model: str, base_url: str = None):
        """
        Initialise le client d'IA approprié.
        
        Args:
            provider: Le fournisseur d'IA à utiliser ("openai" ou "gemini")
            api_key: La clé API du fournisseur
            model: Le nom du modèle à utiliser
            base_url: L'URL de base pour l'API (seulement pour OpenAI)
        """
        self.provider = provider.lower()
        self.model_name = model
        
        if self.provider == "openai":
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            logger.debug(f"Générateur d'IA OpenAI initialisé avec le modèle {model}")
        elif self.provider == "gemini":
            try:
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(model)
                logger.debug(f"Générateur d'IA Gemini initialisé avec le modèle {model}")
            except Exception as e:
                logger.error(f"Erreur d'initialisation de Gemini: {e}")
                raise Exception(f"Échec de l'initialisation de Gemini: {e}")
        else:
            raise ValueError(f"Fournisseur d'IA non supporté: {provider}")
    
    def generate_content(self, prompt: str, content: str) -> str:
        """Génère du contenu en utilisant le fournisseur d'IA configuré."""
        logger.debug(f"Génération de contenu avec {self.provider} et le modèle {self.model_name}")
        logger.trace(f"Prompt utilisé: {prompt[:100]}...")
        logger.trace(f"Contenu à analyser: {content[:100]}...")
        
        try:
            if self.provider == "openai":
                # Construire le message pour OpenAI
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ]
                
                # Appeler l'API OpenAI
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4000,
                )
                
                # Extraire le contenu généré
                generated_content = response.choices[0].message.content.strip()
                
            elif self.provider == "gemini":
                # Construire le prompt complet pour Gemini
                full_prompt = f"{prompt}\n\n{content}"
                
                # Appeler l'API Google Gemini
                response = self.client.generate_content(full_prompt)
                
                # Extraire le contenu généré
                generated_content = response.text.strip()
            
            logger.info(f"Contenu généré avec succès ({len(generated_content)} caractères)")
            logger.trace(f"Début du contenu généré: {generated_content[:100]}...")
            return generated_content
            
        except Exception as e:
            logger.error(f"Erreur lors de la génération de contenu avec {self.provider}: {e}")
            raise Exception(f"Échec de la génération de contenu: {e}")
    
    def generate_summary(self, prompt: str, transcript: str) -> str:
        """
        Génère un résumé à partir d'une transcription et des notes existantes.
        
        Args:
            prompt: Prompt de synthèse
            transcript: Transcription à résumer
            video_id: Identifiant de la vidéo
            input_dir: Répertoire d'entrée contenant les notes potentielles
        
        Returns:
            Résumé généré
        """
        logger.info("Génération d'un résumé")
        return self.generate_content(prompt, transcript)
        
    
    def generate_zettelkasten(self, prompt: str, transcript: str) -> str:
        """Génère des notes au format Zettelkasten."""
        logger.info("Génération de notes Zettelkasten")
        return self.generate_content(prompt, transcript)
    
    def generate_notes(self, prompt: str, transcript: str, video_id: str, input_dir: str) -> str:
        """Génère des notes enrichies."""
        logger.info("Génération de notes enrichies")
        # Préparation du contenu enrichi
        
        
        # Tentative de lecture du fichier de notes correspondant au video_id
        try:
            note_file_path = Path(input_dir) / video_id
            if note_file_path.exists() and note_file_path.is_file():
                with open(note_file_path, 'r', encoding='utf-8') as note_file:
                    notes_content = note_file.read()
                    enhanced_content = f"\n\n**notes :** \"{notes_content}\""
                logger.debug(f"Notes ajoutées depuis {note_file_path}")
                logger.trace(f"Contenu des notes: {notes_content[:100]}...")
            else:
                logger.debug(f"Aucun fichier de notes trouvé pour {video_id}")
        except Exception as e:
            logger.warning(f"Impossible de lire le fichier de notes pour {video_id}: {e}")
        
        #JMT logger.trace(f"Contenu enrichi: {enhanced_content[:200]}...")
        enhanced_content += f"\n\**document :** \"{transcript}\""
        #JMT logger.trace(f"Contenu enrichi: {enhanced_content}")
        return self.generate_content(prompt, enhanced_content)

# Extraction et gestion des concepts
class ConceptExtractor:
    """Extracteur de concepts depuis les documents Zettelkasten."""
    
    def __init__(self, output_dir: str):
        """Initialise avec le répertoire de sortie."""
        self.output_dir = output_dir
        logger.trace(f"ConceptExtractor initialisé avec le répertoire de sortie: {output_dir}")
    
    def extract_concepts(self, zettelkasten_file: str) -> List[Dict[str, str]]:
        """Extrait les concepts depuis un fichier Zettelkasten."""
        logger.info(f"Extraction des concepts depuis {zettelkasten_file}")
        concepts = []
        
        try:
            # Lire le contenu du fichier
            with open(zettelkasten_file, 'r', encoding='utf-8') as file:
                content = file.read()
            
            logger.debug(f"Contenu du fichier Zettelkasten: longueur={len(content)} caractères")
            #JMT
            #logger.trace(f"Content: {content}")
            
            # Rechercher tous les marqueurs #CONCEPT
            concept_markers = re.finditer(r'zzCONCEPT\s+\*\*([^*\n]+)\*\*', content)
            marker_positions = [(m.start(), m.group(1).strip()) for m in concept_markers]
            
            if not marker_positions:
                logger.warning(f"Aucun markeur de concept trouvé dans {zettelkasten_file}. Votre modèle est peut être trop faible ?")
                logger.trace("Vérifier le format du contenu généré")
                return []
                
            logger.debug(f"Marqueurs de concepts trouvés: {len(marker_positions)}")
            
            # Extraire chaque concept avec son contenu
            for i, (start_pos, title) in enumerate(marker_positions):
                # Définir la fin du concept (au prochain #CONCEPT ou à la fin du fichier)
                end_pos = marker_positions[i+1][0] if i < len(marker_positions)-1 else len(content)
                
                # Extraire le contenu complet du concept
                concept_full_content = content[start_pos:end_pos].strip()
                
                # Supprimer le marqueur zzCONCEPT et le titre formaté avec une regex
                # On utilise re.escape(title) pour échapper les caractères spéciaux qui pourraient être dans le titre
                clean_content = re.sub(r'^zzCONCEPT\s+\*\*' + re.escape(title) + r'\*\*', '', concept_full_content, 1).strip()

                logger.trace(f"Concept {i+1}: position={start_pos}, titre='{title}'")
                logger.trace(f"Aperçu du contenu: {concept_full_content[:100]}...")
                
                concept = {
                    "title": title,
                    "content": clean_content
                }
                concepts.append(concept)
                logger.debug(f"Concept extrait: {title}")
            
            logger.info(f"{len(concepts)} concepts extraits avec succès")
            return concepts
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction des concepts: {e}")
            logger.trace(f"Détails de l'erreur:", exc_info=True)
            return []
    
    def save_concepts(self, video_id: str, concepts: List[Dict[str, str]]) -> List[str]:
        """Sauvegarde les concepts dans des fichiers distincts."""
        logger.info(f"Sauvegarde de {len(concepts)} concepts pour {video_id}")
        concept_files = []
        
        try:
            # Assurer que le répertoire existe
            os.makedirs(self.output_dir, exist_ok=True)
            
            for concept in concepts:
                # Créer un nom de fichier valide
                title = concept["title"]
                safe_title = re.sub(r"[^\w\s\-']", '', title).strip()
            
                #safe_title = re.sub(r'[^\w\s-]', '', title).strip()
                #JMT safe_title = re.sub(r'[-\s]+', '-', safe_title)
                filename = f"{safe_title}.md"
                file_path = Path(self.output_dir) / filename
                
                # Sauvegarder le concept
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(concept["content"])
                
                concept_files.append(str(file_path))
                logger.debug(f"Concept sauvegardé dans {file_path}")
            
            return concept_files
            
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des concepts: {e}")
            raise Exception(f"Échec de la sauvegarde des concepts: {e}")
    
    def create_liaison_file(self, video_id: str, concept_files: List[str]) -> str:
        """Crée un fichier de liaison entre les concepts."""
        logger.info(f"Création du fichier de liaison pour {video_id}")
        
        try:
            # Créer le fichier de liaison
            liaison_file = Path(self.output_dir) / f"{video_id}.liaison.md"
            
            content = f"# Concepts associés\n"
            #content += "blah blah\n"
            
            for file_path in concept_files:
                file_name = Path(file_path).name
                concept_name = file_name.replace(f"{video_id}.", "").replace(".concept.md", "")
                content += f"- [[{file_name}|{concept_name}]]\n"
            
            # Sauvegarder le fichier de liaison
            with open(liaison_file, 'w', encoding='utf-8') as file:
                file.write(content)
            
            logger.info(f"Fichier de liaison créé: {liaison_file}")
            logger.trace(f"Contenu du fichier de liaison: {content}")
            return str(liaison_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de la création du fichier de liaison: {e}")
            raise Exception(f"Échec de la création du fichier de liaison: {e}")
        
# Processeur principal
class Processor:
 
    def __init__(self, config: Config, input_dir: str, output_dir: str):
        """Initialise le processeur avec la configuration et les répertoires."""
        self.config = config
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Initialiser les composants
        self.transcript_downloader = TranscriptDownloader(config.youtube_languages)
        
        # Initialiser le générateur IA selon la configuration
        provider = config.ai_provider
        
        if provider == "gemini" and config.gemini_api_key:
            try:
                self.ai_generator = AIGenerator(
                    provider="gemini",
                    api_key=config.gemini_api_key,
                    model=config.gemini_model
                )
                logger.info(f"Utilisation de Google Gemini ({config.gemini_model})")
            except Exception as e:
                logger.warning(f"Échec de l'initialisation de Gemini, utilisation d'OpenAI par défaut: {e}")
                self.ai_generator = AIGenerator(
                    provider="openai",
                    api_key=config.openai_api_key,
                    base_url=config.openai_base_url,
                    model=config.openai_model
                )
        else:
            self.ai_generator = AIGenerator(
                provider="openai",
                api_key=config.openai_api_key,
                base_url=config.openai_base_url,
                model=config.openai_model
            )
            logger.info(f"Utilisation d'OpenAI ({config.openai_model})")
        
        self.concept_extractor = ConceptExtractor(output_dir)
        
        logger.debug(f"Processeur initialisé: input={input_dir}, output={output_dir}")
    
    def get_video_ids(self) -> List[str]:
        """Récupère les IDs de vidéos YouTube depuis le répertoire d'entrée."""
        logger.info(f"Recherche des IDs YouTube dans {self.input_dir}")
        video_ids = []
        
        try:
            input_path = Path(self.input_dir)
            if not input_path.exists():
                logger.warning(f"Le répertoire {self.input_dir} n'existe pas, création...")
                os.makedirs(self.input_dir, exist_ok=True)
                return []
            
            # Lire chaque fichier
            for file in input_path.iterdir():
                if file.is_file():
                    video_id = file.stem
                    video_ids.append(video_id)
                    logger.debug(f"ID YouTube trouvé: {video_id}")
                    logger.trace(f"Chemin du fichier: {file}")
            
            logger.info(f"{len(video_ids)} IDs YouTube trouvés")
            return video_ids
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche des IDs YouTube: {e}")
            return []
    
    def generate_metadata_header(self, video_id: str) -> str:
        """
        Génère un en-tête de métadonnées pour le fichier maître au format YAML frontmatter.
        
        Args:
            video_id: Identifiant de la vidéo YouTube.
            
        Returns:
            En-tête formaté avec les métadonnées.
        """
        # Obtenir la date du jour au format YYYY-MM-DD
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        # Récupérer le titre de la vidéo depuis les métadonnées
        metadata = self.transcript_downloader.get_video_metadata(video_id)
        video_title = metadata.get('title', f"Video {video_id}")
        safe_title = re.sub(r"[^\w\s\-']", "", video_title).strip()
        video_author = metadata.get('author', "")
        channel_id = metadata.get('channel_id', "")
        # channel_id = 
        
        # Construire l'en-tête
        header = f"""---
création : {current_date}
titre : {safe_title}
theme :
type-source : vidéo/youtube
auteur : {video_author}
tags :
---

# Liens
https://www.youtube.com/watch?v={video_id}
"""
        
        logger.debug(f"En-tête de métadonnées généré pour {video_id}")
        logger.trace(f"Contenu de l'en-tête: {header}")
        
        return header


    def process_video(self, video_id: str) -> Dict[str, str]:
        """Traite une vidéo YouTube."""
        logger.info(f"Traitement de la vidéo {video_id}")
        results = {}
        
        try:
            # Étape 1: Télécharger la transcription
            logger.debug(f"Étape 1: Téléchargement de la transcription")
            transcript_file = self.transcript_downloader.save_transcript(video_id, self.output_dir)
            results["transcript_file"] = transcript_file

            # Lire la transcription
            with open(transcript_file, 'r', encoding='utf-8') as file:
                transcript = file.read()
                logger.trace(f"Transcription chargée, longueur: {len(transcript)} caractères")
            
            # Étape 2: Générer les contenus
            
            # Synthèse avec contenu enrichi
            logger.debug(f"Étape 2.1: Génération de la synthèse")
            summary = self.ai_generator.generate_summary(
                self.config.get_prompt("SYNTHESE"),
                transcript
            )
            summary_file = Path(self.output_dir) / f"{video_id}.md"
            with open(summary_file, 'w', encoding='utf-8') as file:
                file.write(summary)
            results["summary_file"] = str(summary_file)
            logger.trace(f"Résumé sauvegardé dans {summary_file}")
            
            # Notes Zettelkasten
            logger.debug(f"Étape 2.2: Génération des notes Zettelkasten")
            zettelkasten = self.ai_generator.generate_zettelkasten(
                self.config.get_prompt("ZETTELKASTEN"),
                transcript
            )
            zettelkasten_file = Path(self.output_dir) / f"{video_id}.zettelkasten.md"
            with open(zettelkasten_file, 'w', encoding='utf-8') as file:
                file.write(zettelkasten)
            results["zettelkasten_file"] = str(zettelkasten_file)
            logger.trace(f"Notes Zettelkasten sauvegardées dans {zettelkasten_file}")
            
            # Notes enrichies
            logger.debug(f"Étape 2.3: Génération des notes enrichies")
            notes = self.ai_generator.generate_notes(
                self.config.get_prompt("NOTES"),
                transcript,
                video_id,
                self.input_dir
            )
            notes_file = Path(self.output_dir) / f"{video_id}.notes.md"
            with open(notes_file, 'w', encoding='utf-8') as file:
                file.write(notes)
            results["notes_file"] = str(notes_file)
            logger.trace(f"Notes enrichies sauvegardées dans {notes_file}")
            
            # Étape 3: Extraire les concepts
            logger.debug(f"Étape 3: Extraction des concepts")
            concepts = self.concept_extractor.extract_concepts(str(zettelkasten_file))
            concept_files = self.concept_extractor.save_concepts(video_id, concepts)
            results["concept_files"] = concept_files
            logger.trace(f"Concepts extraits et sauvegardés: {len(concept_files)} fichiers")
            
            # Fichier de liaison
            liaison_file = self.concept_extractor.create_liaison_file(video_id, concept_files)
            results["liaison_file"] = liaison_file
            logger.trace(f"Fichier de liaison créé: {liaison_file}")
            
            # Étape 4: Consolider les résultats
            logger.debug(f"Étape 4: Consolidation des résultats")
            master_file = self._consolidate_results(video_id, results)
            results["master_file"] = master_file
            logger.trace(f"Fichier maître créé: {master_file}")
            
            # Nettoyer les fichiers temporaires (post-traitement obligatoire)
            self.cleanup_temp_files(video_id)
            
            logger.info(f"Traitement terminé pour {video_id}")
            return results
            
        except Exception as e:
            logger.error(f"Erreur lors du traitement de {video_id}: {e}")
            raise Exception(f"Échec du traitement de {video_id}: {e}")
    
    def _consolidate_results(self, video_id: str, results: Dict[str, str]) -> str:
        """Consolide tous les résultats dans un fichier maître."""
        logger.info(f"Consolidation des résultats pour {video_id}")
        
        try:

            # Récupérer toutes les métadonnées
            metadata = self.transcript_downloader.get_video_metadata(video_id)

            # Extraire uniquement le titre et le sécurise 
            video_title = metadata['title']
            #JMT BLOBI
            safe_video_title = re.sub(r"[^\w\s\-']", "", video_title).strip()
            #safe_video_title = re.sub(r'[^\w\s-]', '', video_title).strip()

            # Créer le fichier maître
            master_file = Path(self.output_dir) / f"_{safe_video_title}.md"
            
            with open(master_file, 'w', encoding='utf-8') as master:
                # Titre
                #JMT master.write(f"# Document maître pour la vidéo {video_id}\n\n")
                
                master_header = self.generate_metadata_header(video_id)
                master.write(master_header)
                master.write("\n\n")

                # Notes enrichies
                #JMT master.write("## Notes enrichies\n\n")
                with open(results["notes_file"], 'r', encoding='utf-8') as file:
                    notes_content = file.read()
                    master.write(notes_content)
                    logger.trace(f"Contenu notes enrichies ajouté: {len(notes_content)} caractères")
                master.write("\n\n")
                
                # Synthèse
                master.write("# Contenu développé par IA\n")
                with open(results["summary_file"], 'r', encoding='utf-8') as file:
                    summary_content = file.read()
                    master.write(summary_content)
                    logger.trace(f"Contenu synthèse ajouté: {len(summary_content)} caractères")
                master.write("\n\n")

                # Liens avec les concepts
                master.write("# Liens entre concepts\n")
                with open(results["liaison_file"], 'r', encoding='utf-8') as file:
                    # Ignorer la première ligne (titre)
                    file.readline()
                    liaison_content = file.read()
                    master.write(liaison_content)
                    logger.trace(f"Contenu liaison ajouté: {len(liaison_content)} caractères")
                
            logger.info(f"Fichier maître créé: {master_file}")
            return str(master_file)
            
        except Exception as e:
            logger.error(f"Erreur lors de la consolidation: {e}")
            raise Exception(f"Échec de la consolidation: {e}")
    
    def process_all_videos(self) -> Dict[str, Dict[str, str]]:
        """Traite toutes les vidéos YouTube."""
        logger.info("Traitement de toutes les vidéos")
        results = {}
        
        # Obtenir les IDs de vidéos
        video_ids = self.get_video_ids()
        
        if not video_ids:
            logger.warning("Aucune vidéo à traiter")
            return results
        
        # Traiter chaque vidéo
        for video_id in video_ids:
            try:
                video_results = self.process_video(video_id)
                results[video_id] = video_results
                logger.trace(f"Résultats pour {video_id}: {list(video_results.keys())}")
            except Exception as e:
                logger.error(f"Échec du traitement de {video_id}: {e}")
                results[video_id] = {"error": str(e)}
        
        logger.info(f"{len(video_ids)} vidéos traitées")
        return results
    
    def cleanup_temp_files(self, video_id: str) -> None:
        """Supprime les fichiers temporaires générés pour une vidéo."""
        logger.info(f"Nettoyage des fichiers temporaires pour {video_id}")
        
        # Liste des fichiers temporaires à format fixe (sans notes.md qui peut être multiple)
        files_to_remove = [
            f"{video_id}.transcript.txt",
            f"{video_id}.md",
            f"{video_id}.notes.md",
            f"{video_id}.zettelkasten.md",
            f"{video_id}.liaison.md"
        ]
        
        # Supprimer les fichiers à format fixe
        for file_name in files_to_remove:
            file_path = Path(self.output_dir) / file_name
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.debug(f"Fichier supprimé: {file_path}")
            except Exception as e:
                logger.warning(f"Erreur lors de la suppression de {file_path}: {e}")
        
        """
        # Supprimer les fichiers de concept
        concept_pattern = f"{video_id}.*.concept.md"
        concept_files = list(Path(self.output_dir).glob(concept_pattern))
        for concept_file in concept_files:
            try:
                concept_file.unlink()
                logger.debug(f"Fichier de concept supprimé: {concept_file}")
            except Exception as e:
                logger.warning(f"Erreur lors de la suppression de {concept_file}: {e}")"""
        
        logger.info(f"Nettoyage terminé pour {video_id}")
    

def ensure_directory(path: str) -> Path:
    """S'assure qu'un répertoire existe."""
    dir_path = Path(path)
    
    if not dir_path.exists():
        dir_path.mkdir(parents=True)
        logger.info(f"Répertoire créé: {dir_path}")
        
    return dir_path

def main():
    """Point d'entrée principal du script."""
    # Analyser les arguments
    parser = argparse.ArgumentParser(description="byteou-netzwerkstatt")
    parser.add_argument("input_dir", help="Répertoire contenant les fichiers d'entrée")
    parser.add_argument("output_dir", help="Répertoire pour les fichiers de sortie")
    parser.add_argument("-c", "--config", default="config.yaml", help="Chemin du fichier de configuration")
    parser.add_argument("-v", "--verbose", action="count", default=0, help="Niveau de verbosité (peut être utilisé plusieurs fois)")
    args = parser.parse_args()
    
    # Configurer le logger
    setup_logger(args.verbose)
    
    try:
        # Message de bienvenue
        console = Console()
        console.print(Panel.fit(
            "[bold blue]byteou-netzwerkstatt[/bold blue]\n"
            "[italic]L'effet papillon de la connaissance[/italic]",
            border_style="blue"
        ))
        
        logger.info(f"Démarrage avec input_dir={args.input_dir}, output_dir={args.output_dir}, config_path={args.config}")
        
        # S'assurer que les répertoires existent
        input_path = ensure_directory(args.input_dir)
        output_path = ensure_directory(args.output_dir)
        
        # Charger la configuration et les prompts
        config = Config(args.config)
        config.load_prompts()
        
        # Créer le processeur
        processor = Processor(config, str(input_path), str(output_path))
        
        # Obtenir les IDs de vidéos
        video_ids = processor.get_video_ids()
        
        if not video_ids:
            logger.warning("Aucune vidéo trouvée dans le répertoire d'entrée")
            console.print("[yellow]Aucune vidéo trouvée dans le répertoire d'entrée[/yellow]")
            return
        
        # Afficher le nombre de vidéos
        console.print(f"[green]Traitement de {len(video_ids)} vidéos...[/green]")

        # Traiter les vidéos avec ou sans barre de progression selon le niveau de verbosité
        if args.verbose >= 1:
            # Mode verbeux, pas de barre de progression
            for video_id in video_ids:
                logger.info(f"Traitement de {video_id}...")
                try:
                    processor.process_video(video_id)
                except Exception as e:
                    logger.error(f"Erreur lors du traitement de {video_id}: {e}")
                    console.print(f"[red]Erreur lors du traitement de {video_id}: {e}[/red]")
        else:
            # Mode non verbeux, afficher la barre de progression
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Traitement des vidéos...", total=len(video_ids))
                
                for video_id in video_ids:
                    progress.update(task, description=f"[cyan]Traitement de {video_id}...")
                    try:
                        processor.process_video(video_id)
                    except Exception as e:
                        console.print(f"[red]Erreur lors du traitement de {video_id}: {e}[/red]")
                    progress.advance(task)
        


        # Afficher un résumé
        console.print("[bold green]Traitement terminé avec succès ![/bold green]")
        console.print(f"Résultats sauvegardés dans: [blue]{output_path}[/blue]")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution: {e}")
        print(f"Erreur: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()