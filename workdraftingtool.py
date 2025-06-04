from crewai.tools import BaseTool
from openai import OpenAI
import os
import logging
import sys
import re
from typing import List
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

# Fonction pour logger et afficher immédiatement dans Colab
def log_and_print(message, level="info"):
    getattr(logger, level)(message)
    print(f"[{level.upper()}] {message}")
    sys.stdout.flush()

print("Configuration du logging en cours...")
log_and_print("Logging configuré avec succès. Si ce message n'apparaît pas, il y a un problème avec le logging.")
print("Configuration du logging terminée.")

class WorkDraftingTool(BaseTool):
    name: str = "work_drafting_tool"
    description: str = "Outil pour rédiger des travaux à partir de fichiers découpés, en traitant morceau par morceau avec un guess adaptatif et un historique global."
    llm_provider: str = "The LLM provider"
    
    def __init__(self, llm_provider: str = "xai"):
        super().__init__()
        self.llm_provider = llm_provider.lower()
        log_and_print(f"Outil initialisé avec llm_provider : {self.llm_provider}")

    def _run(self, file_paths: List[str], file_infos: List[str], project_synthesis: str) -> str:
        print("Début de l'exécution de _run...")
        log_and_print(f"Début de l'exécution avec provider : {self.llm_provider}")
        print("Après le premier log dans _run.")

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
            return f"Erreur : Clé API pour {self.llm_provider.upper()}_API_KEY non définie."

        client = OpenAI(api_key=api_key, base_url=base_url)
        log_and_print(f"LLM configuré avec provider : {self.llm_provider}")
        print("LLM configuré.")

        # Étape 2 : Générer une synthèse succincte pour les rédactions
        synthesis_prompt = (
            f"Vous êtes un expert en synthèse de projets. À partir de cette synthèse détaillée : '{project_synthesis}', "
            f"créez une version succincte et percutante (maximum 50 mots) pour guider la rédaction des travaux. "
            f"Cette synthèse doit refléter les objectifs clés et la pertinence des fichiers sans détails superflus."
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
            print(f"Synthèse succincte pour rédactions : {drafting_synthesis}")
        except Exception as e:
            log_and_print(f"Erreur lors de la génération de la synthèse succincte : {str(e)}", "error")
            drafting_synthesis = "Projet axé sur des objectifs stratégiques et techniques, nécessitant une analyse ciblée des données."

        # Étape 3 : Extraire et découper tous les fichiers avec FileProcessingTool
        file_processor = FileProcessingTool()
        all_chunks = file_processor._run(file_paths)
        log_and_print(f"Extraction des chunks terminée. Nombre total de chunks : {len(all_chunks)}")
        print("Extraction des chunks terminée.")

        if not all_chunks or all(chunk.get("error") for chunk in all_chunks):
            log_and_print("Aucun fichier n’a pu être traité.", "error")
            return "Erreur : Aucun fichier n’a pu être traité."

        # Étape 4 : Organiser les morceaux par fichier
        chunks_by_file = {}
        for chunk in all_chunks:
            if "error" in chunk:
                log_and_print(f"Erreur dans chunk : {chunk['error']}", "warning")
                continue
            file_name = chunk["source"]  # Utilisation directe du nom sans normalisation
            if file_name not in chunks_by_file:
                chunks_by_file[file_name] = []
            chunks_by_file[file_name].append(chunk)
        log_and_print(f"Organisation des chunks par fichier terminée. Fichiers trouvés : {list(chunks_by_file.keys())}")
        print("Organisation des chunks par fichier terminée.")

        # Étape 5 : Initialiser les guesses et les morceaux traités
        file_guesses = {}
        processed_chunks = {}
        works_text = []

        # Variable pour gérer la continuité entre les chunks
        # previous_travaux = ""  # Stocke les travaux du chunk précédent pour extraire last_words

        # Étape 6 : Boucle sur les fichiers avec zip(file_paths, file_infos)
        for file_path, file_info in zip(file_paths, file_infos):
            file_name = os.path.basename(file_path)  # Récupération directe du nom du fichier
            log_and_print(f"Nom de fichier extrait de file_path : {file_name}", "debug")
            if file_name not in chunks_by_file:
                works_text.append(f"--- Aucun contenu pour {file_name} ---")
                log_and_print(f"Aucun contenu trouvé pour le fichier : {file_name}", "warning")
                continue

            file_chunks = chunks_by_file[file_name]
            total_chunks = len(file_chunks)
            if file_name not in processed_chunks:
                processed_chunks[file_name] = []

            # Guess initial
            initial_prompt = (
                f"Vous êtes un analyste de projet. À partir de la synthèse du projet : '{project_synthesis}', "
                f"du nom du fichier : '{file_name}', et des informations suivantes : {file_info}, "
                f"estimez la pertinence de ce fichier pour rédiger des travaux. Retournez uniquement votre estimation (guess) sous forme de texte."
                f"Tu fais au maximum 120 mots"  
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
                print(f"Guess initial pour {file_name} calculé.")
            except Exception as e:
                works_text.append(f"Erreur lors du guess initial pour {file_name} : {str(e)}")
                log_and_print(f"Erreur guess initial pour {file_name} : {str(e)}", "error")
                continue

            # Boucle sur les morceaux
            next_part_id = 1
            print(f"Début de la boucle pour {file_name}. next_part_id : {next_part_id}")
            while next_part_id and next_part_id != "fin" and (isinstance(next_part_id, int) or next_part_id.isdigit()) and 1 <= int(next_part_id) <= total_chunks:
                if next_part_id in processed_chunks[file_name]:
                    works_text.append(f"Erreur : Morceau {next_part_id} déjà traité pour {file_name}. Passage à fin.")
                    log_and_print(f"Morceau {next_part_id} déjà traité pour {file_name}", "warning")
                    break

                current_chunk = file_chunks[next_part_id - 1]
                processed_chunks[file_name].append(next_part_id)
                log_and_print(f"Traitement du morceau {next_part_id} pour {file_name}.", "debug")
                print(f"Traitement du morceau {next_part_id} pour {file_name}.")

                # Vérifier et logger le contenu du morceau
                # chunk_text = current_chunk['text'].strip() if current_chunk['text'] else "Morceau vide"
                # log_and_print(f"Contenu du morceau {next_part_id} pour {file_name} : {chunk_text}", "debug")
                
                # Préparer last_words avant de construire le prompt
                # if next_part_id > 1 and previous_travaux and previous_travaux != "pas rédigé" and len(previous_travaux.split()) >= 20:
                #     last_words = " ".join(previous_travaux.split()[-60:])  # Prend les 30 derniers mots du chunk précédent
                # else:
                #     last_words = ""  # Vide pour le premier chunk ou si le précédent n'est pas rédigé

                # Construire le prompt avec instructions renforcées
                prompt = (
                    f"Vous êtes un rédacteur de travaux réalisés dans le cadre d'un projet.\n"
                    f"Votre rôle est de rédiger du fichier (décomposés en chunks) en travaux en lien avec le projet.\n"
                    # f"Il faut savoir que les informations des fichiers peuvent ne pas sembler interssantes des fois, mais il faut les utiliser pour rédiger les travaux. Parce qu'on n'a pas toujours mieux.\n"
                    # f"Tu rejettes uniquement un morceau s'il n'a vraiment pas de lien avec le projet ou si le contenu n'est pas interessant pour rédiger en travaux.\n"
                    # f"Et vous avez plusieurs fichiers à traiter dont les contenus ne sont pas forcément intéressants pour rédiger les travaux. C'est pour cela qu'on fait les guess pour savoir comment on parcours le fichier, surtout s'il est long.\n"
                    f"L'objectif est de faire des guess sur le fichier à l'aide des morceaux et de la synthèse du projet pour décider comment pour parcourir les chunks.\n"
                    # f"Par exemple, il y a des fichiers qui relèvent d'une fiche de prise en main utilisateur, d'autres qui sont juste des documentations sur une librairie ou une solution non pertinente, des fichiers marketing ou des cahiers de charge non liés aux travaux réalisés.\n"
                    f"Par exemple, pour les fichiers longs (chunks >30),tu sautes des morceaux selon ton guess sur le fichier et le contenu probable (que tu dévines) des morceaux. Tu ne vas pas parcourir tous les morceaux, tu décides sur la base de ton guess sur le fichier et si un morceau serait intéressant ou pas à regarder.\n"
                    f"Supposons que t'es au morceau 2, tu peux par exemple donner numéro 4 directement puisque t'auras jugé que ça valait pas la peine de traiter le morceau 3, parce que tu supposes que ça n'apporterait pas nouvelle information (basé sur ton guess) pour rédiger les travaux du projet.\n"
                    f"Vous vous basez sur le guess actuel pour guider le parcours du fichier.\n"
                    # f"Il est possible que l'utilisateur donne un fichier qui ne colle pas trop avec le projet. Il faut pouvoir les détecter et ne pas les traiter.\n"
                    # f"Tu peux vraiment sauter aux différents morceaux comme tu veux. Tu n'es pas obligé de suivre un ordre linéaire. Surtout si tu as des doutes sur un fichier décider le plus tôt.\n"
                    # f"Dès qu'un fichier ne semble plus pertinent pour toi, tu peux passer directement à fin\n"
                    f"La synthèse succincte du projet est : {drafting_synthesis}.\n"
                    f"Vous travaillez sur le fichier '{file_name}'.\n"
                    f"Informations sur le fichier : {file_info}.\n"
                    f"Nombre total de morceaux : {total_chunks}.\n"
                    f"Morceaux déjà traités : {processed_chunks[file_name]}.\n"
                    f"Guess actuel sur le fichier : {file_guesses[file_name]}.\n"
                    f"Voici le morceau à traiter :\n"
                    f"Morceau {current_chunk['part_id']} : {current_chunk['text']}\n\n"
                    f"**Instructions importantes** : Vous DEVEZ retourner une réponse dans ce format exact :\n"
                    f"Travaux: texte rédigé ou 'pas rédigé'\n"
                    f"Guess: nouveau guess\n"
                    f"Prochain morceau: numéro ou 'fin'\n"
                    f"--- Métadonnées ---\n"
                    f"Source: {file_name}\n"
                    f"Part_id: {current_chunk['part_id']}\n"
                    f"Étape 1 : Évaluez le contenu du chunk et son lien avec le projet pour le rédiger en travaux.\n"
                    f"Étape 2 : S'il y a un lien, rédigez les travaux en utilisant comme sujet 'nous' avec très peu de puces et mentionnez les difficultés rencontrées (s'il y en a). Tu présentes aussi l'objectif des travaux.\n"
                    f"Suis l'instruction de l'Etape 2. Ne mets pas des commentaires, des introductions. L'objectif est de rédiger sous forme de travaux.\n"
                    f"Vous n'êtes pas censé dire des choses du genre : Nous avons analysé en détail ... décrites dans le document (faisant référence avec le morceau analysé), etc.\n"
                    f"Ce n'est pas ça l'objectif. Ce n'est pas de décrire le contenu des fichiers. NON! Vous NE DECRIVEZ pas les morceaux. Vous rédigez le contenu directement comme des travaux réalisés. Vous transformes le morceau pour que ça sonne travaux réalisés PAS UNE DESCRIPTION!! TRES IMPORTANT\n"
                    f"Notez que les 100 premiers mots du chunk sont identiques à la fin du morceau précédent. Rédigez en tenant compte de ce chevauchement pour qu'on n'ait pas de répétition.\n\n"
                    # f"Pour qu'il n'y ait pas de coupure dans la rédaction, tu utilises Last words (s'il existe) pour avoir une idée sur sur la fin du texte du chunk précédent rédigé dont sera la suite la rédaction que tu vas faire.\n"
                    # f"Last words: {last_words}.\n"
                    f"Pour donner l'impression de fluidité, s'il y a déjà eu un chunk déjà traité (il faut faire introdution pour le premier chunk), REDIGE DIRECTEMENT EN TRAVAUX, sans vouloir faire une transition ou une introduction. Tu vas commence directement par le corps du sujet, les travaux. N'UTILISE AUCUNE PHRASE DE TRANSITION, NON!\n"
                    # f"Étape 3 : IL FAUT TOUJOURS REDIGER MEME S'IL N'Y A PAS DE LIEN AVEC LE PROJET.\n"
                    f"Étape 3 : Si le chunk n'a pas de lien avec le projet ou le contenu n'est pas intéressant, indiquez 'pas rédigé'.\n"
                    f"Étape 4 : Mettez à jour votre guess sur le fichier en fonction de ce morceau.\n"
                    f"Étape 5 : Choisissez le prochain morceau (numéro entre 1 et {total_chunks}, ou 'fin' si rien à traiter).\n\n"
                 )
                try:
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1000,
                        temperature=0.5
                    )
                    result = response.choices[0].message.content.strip()
                    log_and_print(f"Réponse LLM reçue pour morceau {next_part_id} : {result}")
                    # print(f"Réponse LLM pour morceau {next_part_id} : {result}")

                    # Parser la réponse
                    lines = result.split("\n")
                    works = ""
                    new_guess = file_guesses[file_name]
                    next_part = None

                    for line in lines:
                        line = line.strip()
                        # log_and_print(f"Ligne parsée : '{line}'", "debug")
                        if line.lower().startswith("travaux:"):
                            works = line.replace("Travaux:", "").replace("Travaux :", "").strip()
                            # log_and_print(f"Travaux extraits : {works}", "debug")
                        elif re.match(r"Guess\s*:\s*.+", line):
                            match = re.match(r"Guess\s*:\s*(.+)", line)
                            if match:
                                new_guess = match.group(1).strip()
                                file_guesses[file_name] = new_guess
                                # log_and_print(f"Nouveau guess extrait avec regex : {new_guess}", "debug")
                            else:
                                log_and_print(f"Format inattendu pour 'Guess' : {line}", "warning")
                        elif re.match(r"Prochain morceau\s*:\s*\w+", line):
                            match = re.match(r"Prochain morceau\s*:\s*(\w+)", line)
                            if match:
                                next_part_str = match.group(1).strip()
                                next_part = int(next_part_str) if next_part_str.isdigit() else next_part_str
                                log_and_print(f"Prochain morceau extrait avec regex : {next_part}", "debug")
                            else:
                                log_and_print(f"Format inattendu pour 'Prochain morceau' : {line}", "warning")

                    if next_part is None:
                        log_and_print(f"Échec du parsing de 'Prochain morceau' pour {file_name}, morceau {next_part_id}. Réponse : {result}", "error")
                        next_part = "fin"

                    # Ajouter les travaux au fichier de sortie
                    if works:
                        log_and_print(f"Valeur de works après parsing : {works}", "debug")
                        if works != "pas rédigé":
                            works_text.append(f"Travaux (source : {file_name}) : {works}")
                        else:
                            log_and_print(f"Travaux non ajoutés car marqués comme 'pas rédigé' pour {file_name}, morceau {next_part_id}")
                    else:
                        log_and_print(f"Aucun travaux extrait pour {file_name}, morceau {next_part_id}", "warning")

                    # Mise à jour de previous_travaux pour le prochain chunk
                    # previous_travaux = works
                    
                    next_part_id = next_part
                    log_and_print(f"Fin de l'itération. next_part_id : {next_part_id}", "debug")
                    # print(f"Fin de l'itération. next_part_id : {next_part_id}")

                except Exception as e:
                    log_and_print(f"Erreur lors du traitement du morceau {next_part_id} pour {file_name} : {str(e)}", "error")
                    works_text.append(f"Erreur lors du traitement du morceau {next_part_id} pour {file_name} : {str(e)}")
                    next_part = "fin"
                    next_part_id = next_part

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
# tool = WorkDraftingTool(llm_provider="openai")
# file_paths = ["Fonctionnalités.pdf", "Révision 2023_Premium UX-UI.pdf", "User guide Teamnews.pdf"]
# file_infos = [
#     "Pour le fichier Fonctionnalités.pdf, la position est dossier source. Concernant le fichier, pour le type de contenu ce sont des fonctionnalités de la solution et d'autres infos. Concernant la description du contenu, il y a application, QRcode, canal par défaut, backoffice, web, ECRANS en gras avec des bullets qui présenter certaines infos (dont je ne sais pas trop la nature)",
#     "Pour le fichier Révision 2023_Premium UX-UI.pdf, la position est dossier source. Concernant le fichier, pour le type de contenu c'est un cachier de charge en gros avec beaucoup d'image de l'interface de l'application. Concernant la description du contenu, il y a des titres, avec des descriptions (bullet points), et des images. On a par exemple comme titres Nom de l'application, Personnalisation des fiches stores, Icônes, Palette de couleur, Ecran d'accueil, Message d'accueil, et autres",
#     "Pour le fichier User guide Teamnews.pdf, la position est dossier source. Concernant le fichier, le type de contenu est un guide d'utilisateur. Concernant la description du contenu, on explique les avantages de la solution et comment accéder à certaines fonctionnalités"
#     ]

# with open("synthesis.txt", "r", encoding="utf-8") as f:
#     project_synthesis = f.read().strip()
 
# result = tool._run(file_paths, file_infos, project_synthesis)
# print(result)