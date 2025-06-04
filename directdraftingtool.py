from crewai.tools import BaseTool
from openai import OpenAI
import os
import logging
import sys
from typing import List, Tuple
from fileprocessingtool import FileProcessingTool

# Configuration explicite du logging
logging.getLogger('').handlers = []

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.propagate = False

logger.handlers = []

# Utiliser StreamHandler sans sys.stdout pour Colab
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("global_history.log", mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Fonction pour logger sans duplication
def log_and_print(message, level="info"):
    getattr(logger, level)(message)  # Utilise uniquement logger

logger.info("Configuration du logging en cours...")
log_and_print("Logging configuré avec succès. Si ce message n'apparaît pas, il y a un problème avec le logging.")
logger.info("Configuration du logging terminée.")

class DirectDraftingTool(BaseTool):
    name: str = "direct_drafting_tool"
    description: str = "Outil pour rédiger directement des travaux à partir de chunks spécifiés par l'utilisateur."
    llm_provider: str = "The LLM provider"
    
    def __init__(self, llm_provider: str = "xai"):
        super().__init__()
        self.llm_provider = llm_provider.lower()
        log_and_print(f"Outil initialisé avec llm_provider : {self.llm_provider}")

    def _run(self, file_paths: List[str], file_infos: List[str], project_synthesis: str, user_chunks_to_draft: List[Tuple[str, int]]) -> str:
        log_and_print(f"Début de l'exécution avec provider : {self.llm_provider}")

        # Étape 1 : Configurer le LLM
        if self.llm_provider == "xai":
            api_key = os.getenv("XAI_API_KEY")
            base_url = "https://api.x.ai/v1"
            model_id = "grok-3-beta"
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None
            model_id = "gpt-4o"
        if not api_key:
            log_and_print(f"Clé API pour {self.llm_provider.upper()}_API_KEY non définie.", "error")
            return "Erreur : Clé API non définie."

        client = OpenAI(api_key=api_key, base_url=base_url)
        log_and_print(f"LLM configuré avec provider : {self.llm_provider}")

        # Étape 2 : Générer une synthèse succincte pour guider la rédaction
        synthesis_prompt = (
            f"Vous êtes un expert en synthèse de projets. À partir de cette synthèse détaillée : '{project_synthesis}', "
            f"créez une version succincte et percutante (maximum 50 mots) pour guider la rédaction des travaux. "
            f"Cette synthèse doit refléter les objectifs clés sans détails superflus."
        )
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=60,
                temperature=0.3
            )
            drafting_synthesis = response.choices[0].message.content.strip()
            log_and_print(f"Synthèse succincte générée : {drafting_synthesis}")
        except Exception as e:
            log_and_print(f"Erreur lors de la génération de la synthèse succincte : {str(e)}", "error")
            drafting_synthesis = "Projet axé sur des objectifs stratégiques et techniques, nécessitant une analyse ciblée."

        # Étape 3 : Extraire et découper tous les fichiers avec FileProcessingTool
        file_processor = FileProcessingTool()
        all_chunks = file_processor._run(file_paths)
        log_and_print(f"Extraction des chunks terminée. Nombre total de chunks : {len(all_chunks)}")

        if not all_chunks or all(chunk.get("error") for chunk in all_chunks):
            log_and_print("Aucun fichier n’a pu être traité.", "error")
            return "Erreur : Aucun fichier n’a pu être traité."

        # Étape 4 : Organiser les morceaux par fichier
        chunks_by_file = {}
        for chunk in all_chunks:
            if "error" in chunk:
                log_and_print(f"Erreur dans chunk : {chunk['error']}", "warning")
                continue
            file_name = chunk["source"]
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk)
        log_and_print(f"Organisation des chunks par fichier terminée. Fichiers trouvés : {list(chunks_by_file.keys())}")

        # Étape 5 : Initialiser les travaux
        works_text = []
        previous_travaux = ""  # Pour assurer la continuité entre les chunks
        file_info_dict = {os.path.basename(file_path): info for file_path, info in zip(file_paths, file_infos)}  # Associer file_infos par nom de fichier

        # Étape 6 : Rédiger les chunks spécifiés par l'utilisateur
        for file_name, part_id in user_chunks_to_draft:
            if file_name not in chunks_by_file:
                log_and_print(f"Aucun contenu trouvé pour le fichier : {file_name}", "warning")
                works_text.append(f"--- Aucun contenu pour {file_name} ---")
                continue

            file_chunks = chunks_by_file[file_name]
            # Trouver le chunk correspondant à part_id
            current_chunk = next((chunk for chunk in file_chunks if chunk['part_id'] == part_id), None)
            if not current_chunk:
                log_and_print(f"Morceau {part_id} non trouvé pour {file_name}", "warning")
                works_text.append(f"--- Morceau {part_id} non trouvé pour {file_name} ---")
                continue

            chunk_text = current_chunk['text'].strip() if current_chunk['text'] else "Morceau vide"
            log_and_print(f"Rédaction du morceau {part_id} pour {file_name}.", "debug")

            # Récupérer les informations du fichier
            file_info = file_info_dict.get(file_name, "Informations non disponibles")

            # Préparer les derniers mots du chunk précédent pour la continuité
            if previous_travaux and len(previous_travaux.split()) >= 20:
                last_words = " ".join(previous_travaux.split()[-30:])  # 30 derniers mots
            else:
                last_words = ""

            # Prompt pour rédiger directement les travaux
            prompt = (
                f"Vous êtes un rédacteur de travaux scientifiques.\n"
                f"La synthèse succincte du projet est : {drafting_synthesis}.\n"
                f"Vous travaillez sur le fichier '{file_name}'.\n"
                f"Informations sur le fichier : {file_info}.\n"
                f"Morceau {part_id} : {chunk_text}\n\n"
                f"Continuez à partir de : {last_words} avec le contenu de ce morceau. Rédigez directement les travaux en utilisant 'nous' comme sujet, avec très peu de puces, et mentionnez les difficultés rencontrées (s'il y en a). La rédaction doit être complète, logique et claire.\n"
                f"Ne décrivez pas le contenu du morceau, transformez-le en travaux réalisés. Évitez les introductions ou commentaires généraux."
            )
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.5
                )
                works = response.choices[0].message.content.strip()
                log_and_print(f"Travaux générés pour morceau {part_id} : {works}")
                if works:
                    works_text.append(f"Travaux (source : {file_name}, partie {part_id}) : {works}")
                else:
                    log_and_print(f"Aucun travaux généré pour morceau {part_id} de {file_name}", "warning")
                    works_text.append(f"--- Aucun travaux pour {file_name}, partie {part_id} ---")

                # Mettre à jour previous_travaux pour la continuité
                previous_travaux = works

            except Exception as e:
                log_and_print(f"Erreur lors de la rédaction du morceau {part_id} pour {file_name} : {str(e)}", "error")
                works_text.append(f"Erreur lors de la rédaction du morceau {part_id} pour {file_name} : {str(e)}")

        # Vérifier si works_text est vide avant écriture
        if not works_text:
            log_and_print("Aucun travaux rédigé pour aucun fichier.", "warning")
            works_text.append("Aucun travaux rédigé pour les fichiers traités.")

        # Écrire les travaux dans un fichier unique
        try:
            with open("works_output.txt", "w", encoding="utf-8") as f:
                f.write("\n\n".join(works_text))
            log_and_print("Travaux écrits dans works_output.txt")
        except Exception as e:
            log_and_print(f"Erreur lors de l'écriture dans works_output.txt : {str(e)}", "error")

        return "\n\n".join(works_text)

# Exemple d’utilisation
# tool = DirectDraftingTool(llm_provider="xai")
# file_paths = ["Fonctionnalités.pdf", "Révision 2023_Premium UX-UI.pdf"]
# file_infos = [
#     "Pour le fichier Fonctionnalités.pdf, la position est dossier source. Contenu : fonctionnalités de la solution.",
#     "Pour le fichier Révision 2023_Premium UX-UI.pdf, la position est dossier source. Contenu : cahier de charge avec images."
# ]
# user_chunks_to_draft = [("Fonctionnalités.pdf", 1), ("Révision 2023_Premium UX-UI.pdf", 2)]
# with open("synthesis.txt", "r", encoding="utf-8") as f:
#     project_synthesis = f.read().strip()
# result = tool._run(file_paths, file_infos, project_synthesis, user_chunks_to_draft)
# print(result)