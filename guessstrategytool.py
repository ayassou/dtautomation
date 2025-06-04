from crewai.tools import BaseTool
from openai import OpenAI
import os
import logging
import sys
import re
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

class GuessStrategyTool(BaseTool):
    name: str = "guess_strategy_tool"
    description: str = "Outil pour évaluer la pertinence des chunks et élaborer une stratégie de parcours des documents."
    llm_provider: str = "The LLM provider"
    
    def __init__(self, llm_provider: str = "xai"):
        super().__init__()
        self.llm_provider = llm_provider.lower()
        log_and_print(f"Outil initialisé avec llm_provider : {self.llm_provider}")

    def _run(self, file_paths: List[str], file_infos: List[str], project_synthesis: str) -> List[Tuple[str, int]]:
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
            return []

        client = OpenAI(api_key=api_key, base_url=base_url)
        log_and_print(f"LLM configuré avec provider : {self.llm_provider}")

        # Étape 2 : Générer une synthèse succincte pour guider l'évaluation
        synthesis_prompt = (
            f"Vous êtes un expert en synthèse de projets. À partir de cette synthèse détaillée : '{project_synthesis}', "
            f"créez une version succincte et percutante (maximum 50 mots) pour guider l'évaluation de pertinence des fichiers. "
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
            return []

        # Étape 4 : Organiser les morceaux par fichier
        chunks_by_file = {}
        for chunk in all_chunks:
            if "error" in chunk:
                log_and_print(f"Erreur dans chunk : {chunk['error']}", "warning")
                continue
            file_name = os.path.basename(chunk["source"])  # Utiliser le nom original du fichier
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk)
        log_and_print(f"Organisation des chunks par fichier terminée. Fichiers trouvés : {list(chunks_by_file.keys())}")

        # Étape 5 : Initialiser les structures
        file_guesses = {}  # Stocke les guesses pour chaque fichier
        processed_chunks = {}  # Stocke les chunks déjà analysés
        chunks_to_draft = []  # Liste des (file_name, part_id) à envoyer pour rédaction

        # Étape 6 : Boucle sur les fichiers avec zip(file_paths, file_infos)
        for file_path, file_info in zip(file_paths, file_infos):
            file_name = os.path.basename(file_path)  # Utiliser le nom original du fichier
            log_and_print(f"Nom de fichier extrait de file_path : {file_name}", "debug")
            if file_name not in chunks_by_file:
                log_and_print(f"Aucun contenu trouvé pour le fichier : {file_name}", "warning")
                continue

            file_chunks = chunks_by_file[file_name]
            total_chunks = len(file_chunks)
            if file_name not in processed_chunks:
                processed_chunks[file_name] = []

            # Guess initial pour le fichier
            initial_prompt = (
                f"Vous êtes un analyste de projet. À partir de la synthèse du projet : '{project_synthesis}', "
                f"du nom du fichier : '{file_name}', et des informations suivantes : {file_info}, "
                f"estimez la pertinence de ce fichier pour rédiger des travaux réalisés dans le projet. "
                f"Fournissez une estimation (max 90 mots) en expliquant si le fichier contient des éléments "
                f"techniques, stratégiques ou directement liés aux travaux réalisés, ou s'il est non pertinent (ex. documentation d'une librarie python ou autre solution, cahier de charge non lié, etc. tout ce qui ne renseigne pas sur ce qui a été fait dans le projet)."
            )
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": initial_prompt}],
                    max_tokens=200,
                    temperature=0.5
                )
                current_guess = response.choices[0].message.content.strip()
                file_guesses[file_name] = current_guess
                log_and_print(f"Guess initial pour {file_name} : {current_guess}")
            except Exception as e:
                log_and_print(f"Erreur guess initial pour {file_name} : {str(e)}", "error")
                continue

            # Boucle pour analyser les morceaux et élaborer une stratégie
            next_part_id = 1
            while next_part_id and next_part_id != "fin" and (isinstance(next_part_id, int) or next_part_id.isdigit()) and 1 <= int(next_part_id) <= total_chunks:
                if next_part_id in processed_chunks[file_name]:
                    log_and_print(f"Morceau {next_part_id} déjà analysé pour {file_name}", "warning")
                    break

                current_chunk = file_chunks[next_part_id - 1]
                processed_chunks[file_name].append(next_part_id)
                log_and_print(f"Traitement du morceau {next_part_id} pour {file_name}.", "debug")

                # Évaluer la pertinence du chunk avec une stratégie efficace
                prompt = (
                    f"Vous êtes un analyste de projet. La synthèse succincte du projet est : {drafting_synthesis}.\n"
                    f"Vous travaillez sur le fichier '{file_name}' (total chunks : {total_chunks}).\n"
                    f"Informations sur le fichier : {file_info}.\n"
                    f"Guess actuel sur le fichier : {file_guesses[file_name]}.\n"
                    f"Morceaux déjà analysés : {processed_chunks[file_name]}.\n"
                    f"Voici le morceau à évaluer :\n"
                    f"Morceau {current_chunk['part_id']} : {current_chunk['text'].strip() if current_chunk['text'] else 'Morceau vide'}\n\n"
                    f"**Objectif** : Tri des chunks pour identifier ceux qui contiennent des informations sur les travaux réalisés dans le projet. "
                    f"Ignorez les contenus non pertinents comme les documentations externes, ou cahiers de charge ou tout autre fichier ou chunk non liés aux travaux du projet.\n"
                    f"**Stratégie de parcours** : Élaborez une stratégie efficace, surtout pour les fichiers longs (>10 chunks). Vous pouvez :\n"
                    f"- Sauter des chunks selon le guess.\n"
                    f"- Revenir en arrière selon le guess.\n"
                    f"- Arrêter rapidement ('fin') si le fichier semble non pertinent.\n"
                    f"**Instructions importantes** : Retournez une réponse dans ce format exact :\n"
                    f"Pertinent: oui ou non\n"
                    f"Explication: raison de la décision (max 40 mots)\n"
                    f"Guess: nouveau guess (max 80 mots)\n"
                    f"Prochain morceau: numéro entre 1 et {total_chunks} ou 'fin'\n"
                    f"Étape 1 : Déterminez si ce chunk contient des éléments liés aux travaux réalisés.\n"
                    f"Étape 2 : Expliquez brièvement.\n"
                    f"Étape 3 : Mettez à jour le guess.\n"
                    f"Étape 4 : Choisissez le prochain morceau stratégiquement."
                )
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=400,
                        temperature=0.5
                    )
                    result = response.choices[0].message.content.strip()
                    log_and_print(f"Réponse LLM pour morceau {next_part_id} : {result}")

                    # Parser la réponse
                    lines = result.split("\n")
                    pertinent = "non"
                    new_guess = file_guesses[file_name]
                    next_part = None

                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith("pertinent:"):
                            pertinent = line.replace("Pertinent:", "").replace("Pertinent :", "").strip().lower()
                        elif re.match(r"Guess\s*:\s*.+", line):
                            match = re.match(r"Guess\s*:\s*(.+)", line)
                            if match:
                                new_guess = match.group(1).strip()
                                file_guesses[file_name] = new_guess
                        elif re.match(r"Prochain morceau\s*:\s*\w+", line):
                            match = re.match(r"Prochain morceau\s*:\s*(\w+)", line)
                            if match:
                                next_part_str = match.group(1).strip()
                                next_part = int(next_part_str) if next_part_str.isdigit() else next_part_str
                            else:
                                log_and_print(f"Format inattendu pour 'Prochain morceau' : {line}", "warning")
                                next_part = "fin"  # Par défaut si mal formaté

                    if next_part is None:
                        log_and_print(f"Échec du parsing de 'Prochain morceau' pour {file_name}, morceau {next_part_id}. Réponse : {result}", "error")
                        next_part = "fin"

                    # Si pertinent, ajouter le chunk à rédiger
                    if pertinent == "oui":
                        chunks_to_draft.append((file_name, next_part_id))
                        log_and_print(f"Morceau {next_part_id} de {file_name} marqué comme pertinent pour rédaction.")

                    next_part_id = next_part

                except Exception as e:
                    log_and_print(f"Erreur lors de l'évaluation du morceau {next_part_id} pour {file_name} : {str(e)}", "error")
                    next_part_id = "fin"

        # Étape 7 : Sauvegarder les chunks pertinents dans un fichier pour l'utilisateur
        if chunks_to_draft:
            try:
                with open("chunks_to_draft.txt", "w", encoding="utf-8") as f:
                    for file_name, part_id in chunks_to_draft:
                        f.write(f"{file_name},{part_id}\n")
                log_and_print(f"Chunks pertinents sauvegardés dans chunks_to_draft.txt : {chunks_to_draft}")
            except Exception as e:
                log_and_print(f"Erreur lors de la sauvegarde dans chunks_to_draft.txt : {str(e)}", "error")

        log_and_print(f"Chunks à rédiger : {chunks_to_draft}")
        return chunks_to_draft

# Exemple d’utilisation
# tool = GuessStrategyTool(llm_provider="xai")
# file_paths = ["Fonctionnalités.pdf", "Révision 2023_Premium UX-UI.pdf", "User guide Teamnews.pdf"]
# file_infos = [
#     "Pour le fichier Fonctionnalités.pdf, la position est dossier source. Concernant le fichier, pour le type de contenu ce sont des fonctionnalités de la solution et d'autres infos. Concernant la description du contenu, il y a application, QRcode, canal par défaut, backoffice, web, ECRANS en gras avec des bullets qui présenter certaines infos (dont je ne sais pas trop la nature)",
#     "Pour le fichier Révision 2023_Premium UX-UI.pdf, la position est dossier source. Concernant le fichier, pour le type de contenu c'est un cachier de charge en gros avec beaucoup d'image de l'interface de l'application. Concernant la description du contenu, il y a des titres, avec des descriptions (bullet points), et des images. On a par exemple comme titres Nom de l'application, Personnalisation des fiches stores, Icônes, Palette de couleur, Ecran d'accueil, Message d'accueil, et autres",
#     "Pour le fichier User guide Teamnews.pdf, la position est dossier source. Concernant le fichier, le type de contenu est un guide d'utilisateur. Concernant la description du contenu, on explique les avantages de la solution et comment accéder à certaines fonctionnalités"
# ]
# with open("synthesis.txt", "r", encoding="utf-8") as f:
#     project_synthesis = f.read().strip()
# result = tool._run(file_paths, file_infos, project_synthesis)
# print(result)