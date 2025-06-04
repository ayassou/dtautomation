import streamlit as st
import os
from workdraftingtool import WorkDraftingTool
from synthesis_tool import SynthesisTool
from innovationanalysistool import InnovationAnalysisTool
from marketstudytool import MarketStudyTool
from draftingtool import DraftingTool
from directdraftingtool import DirectDraftingTool
from fileprocessingtool import FileProcessingTool
from guessstrategytool import GuessStrategyTool
# Importer les modules nécessaires pour gérer les répertoires temporaires
import shutil  # Pour supprimer le répertoire temp_dir après usage

# Configuration de l'interface
st.title("Agent Consultant IA - Interface Utilisateur")
st.write("Cette application permet d'utiliser plusieurs outils pour rédiger des travaux, générer des synthèses, analyser des innovations, réaliser des études de marché, et rédiger des sections de rapport.")

# Section 0 : Configuration globale
st.header("Configuration Globale")
llm_provider = st.selectbox("Choisir le fournisseur LLM", ["xai", "openai"], index=0)
st.write("Assurez-vous que la clé API correspondante (`XAI_API_KEY` ou `OPENAI_API_KEY`) est définie dans votre environnement.")


# Section 1 : SynthesisTool
st.header("Génération de Synthèse Structurée avec SynthesisTool")
word_file = st.file_uploader("Charger un fichier Word (.docx)", type=["docx"], key="synthesis_file")

if st.button("Générer la synthèse", key="synthesis_button"):
    if word_file:
        file_path = word_file.name
        with open(file_path, "wb") as f:
            f.write(word_file.getbuffer())

        try:
            tool = SynthesisTool(llm_provider=llm_provider)
            result = tool._run(file_path)
            st.subheader("Synthèse Structurée")
            st.write(result)

            with open("structured_synthesis.txt", "r", encoding="utf-8") as f:
                st.download_button("Télécharger structured_synthesis.txt", f.read(), file_name="structured_synthesis.txt", key="download_synthesis")
        except Exception as e:
            st.error(f"Erreur lors de la génération de la synthèse : {str(e)}")
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
    else:
        st.warning("Veuillez charger un fichier Word.")

# Section 3 : InnovationAnalysisTool
st.header("Analyse des Innovations avec InnovationAnalysisTool")
synthesis_input = st.text_area("Entrez la synthèse de la solution", "La solution offre une approche innovante...", key="innovation_synthesis")
solution_name = st.text_input("Nom de la solution", "Ekonsilio Chat", key="innovation_solution")
company_name = st.text_input("Nom de l'entreprise", "Ekonsilio", key="innovation_company")
website_url = st.text_input("URL du site web (optionnel)", "www.ekonsilio.com", key="innovation_url")

if st.button("Analyser les innovations", key="innovation_button"):
    if synthesis_input and solution_name and company_name:
        try:
            tool = InnovationAnalysisTool()
            website = website_url if website_url else None
            result = tool._run(synthesis_input, solution_name, company_name, website, llm_provider=llm_provider)
            st.subheader("Analyse des Innovations")
            st.write(result)
        except Exception as e:
            st.error(f"Erreur lors de l'analyse des innovations : {str(e)}")
    else:
        st.warning("Veuillez remplir tous les champs obligatoires.")

# Section 4 : MarketStudyTool
st.header("Étude de Marché avec MarketStudyTool")
market_synthesis = st.text_area("Entrez la synthèse de la solution", "Citykomi détient un brevet concernant cette innovation...", key="market_synthesis")
web_info = st.text_area("Informations du site web (optionnel)", "Citykomi est une application mobile qui notifie...", key="market_web_info")
innovation_analysis = st.text_area("Analyse d'innovation (optionnel)", "L’innovation de Citykomi réside dans le fait qu’aucune donnée...", key="market_innovation")
market_solution_name = st.text_input("Nom de la solution", "Citykomi", key="market_solution")
market_company_name = st.text_input("Nom de l'entreprise", "Citykomi", key="market_company")

if st.button("Réaliser l'étude de marché", key="market_button"):
    if market_synthesis and market_solution_name and market_company_name:
        try:
            tool = MarketStudyTool()
            result = tool._run(
                synthesis=market_synthesis,
                web_info=web_info,
                innovation_analysis=innovation_analysis,
                solution_name=market_solution_name,
                company_name=market_company_name,
                llm_provider=llm_provider
            )
            st.subheader("Étude de Marché")
            st.write(result)
        except Exception as e:
            st.error(f"Erreur lors de l'étude de marché : {str(e)}")
    else:
        st.warning("Veuillez remplir tous les champs obligatoires.")

# Section 5 : GuessStrategyTool
st.header("Génération de Stratégie de Rédaction avec GuessStrategyTool")
uploaded_files_guess = st.file_uploader("Uploader les fichiers pour la stratégie", accept_multiple_files=True, type=["pdf", "txt", "docx", "xlsx", "ppt"], key="guess_files")
# Champ obligatoire pour la synthèse du projet
project_synthesis_guess = st.text_area("Synthèse du projet (obligatoire, guidant la stratégie de rédaction)", "Entrez la synthèse ici", key="guess_synthesis")

if uploaded_files_guess and project_synthesis_guess:
    # Créer un répertoire temporaire temp_dir pour stocker les fichiers
    temp_dir = "temp_dir_guess"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Créer le répertoire s'il n'existe pas

    file_paths_guess = []
    # Sauvegarder les fichiers dans temp_dir avec leurs noms originaux
    for uploaded_file in uploaded_files_guess:
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths_guess.append(temp_file_path)

    # Champ pour les informations des fichiers
    file_infos_guess = []
    for file_path in file_paths_guess:
        file_info = st.text_area(f"Informations pour {os.path.basename(file_path)} (ex. type de contenu, description)", key=f"file_info_{os.path.basename(file_path)}")
        file_infos_guess.append(file_info if file_info else f"Pour le fichier {os.path.basename(file_path)}, la position est dossier source. Contenu : contenu générique.")

    if st.button("Générer la stratégie de rédaction", key="guess_button"):
        if file_paths_guess:
            try:
                tool = GuessStrategyTool(llm_provider=llm_provider)
                result = tool._run(file_paths_guess, file_infos_guess, project_synthesis_guess)
                st.subheader("Stratégie de Rédaction Suggerée")
                for file_name, parts in result.items():
                    st.write(f"{file_name}: {', '.join(str(p) for p in parts)}")

                # Proposer le téléchargement
                output = "\n".join([f"{file_name},{','.join(str(p) for p in parts)}" for file_name, parts in result.items()])
                st.download_button("Télécharger chunks_to_draft.txt", output, file_name="chunks_to_draft.txt", key="download_guess")
            except Exception as e:
                st.error(f"Erreur lors de la génération de la stratégie : {str(e)}")
        else:
            st.warning("Veuillez uploader au moins un fichier.")

        # Nettoyer le répertoire temporaire
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Supprimer temp_dir et tous ses fichiers
elif not project_synthesis_guess:
    st.warning("Veuillez entrer la synthèse du projet (obligatoire).")


# Section 6 : DirectDraftingTool
st.header("Générer des Travaux avec DirectDraftingTool")
# Élargir les formats supportés pour inclure PPT, Excel, PDF, Word
uploaded_files = st.file_uploader("Uploader les fichiers", accept_multiple_files=True, type=["pdf", "txt", "docx", "xlsx", "ppt"])
# Champ obligatoire pour la synthèse du projet
project_synthesis = st.text_area("Synthèse du projet (obligatoire)", "Entrez la synthèse ici pour guider la rédaction des travaux", key="directdraft_synthesis")

if uploaded_files and project_synthesis:
    # Créer un répertoire temporaire temp_dir pour stocker les fichiers
    temp_dir = "temp_dir"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)  # Créer le répertoire s'il n'existe pas

    file_paths = []
    # Sauvegarder les fichiers dans temp_dir avec leurs noms originaux
    for uploaded_file in uploaded_files:
        # Utiliser le nom original dans temp_dir (ex. temp_dir/Fonctionnalités.pdf)
        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)

    # Charger les informations des fichiers (à adapter selon ton interface)
    file_infos = [f"Pour le fichier {os.path.basename(f)}, la position est dossier source. Contenu : contenu générique." for f in file_paths]

    # Charger les suggestions de chunks_to_draft.txt (optionnel)
    chunks_to_draft_content = st.file_uploader("Uploader chunks_to_draft.txt (optionnel)", type=["txt"])
    suggested_chunks = {}
    if chunks_to_draft_content:
        # Lire le contenu du fichier uploadé avec read() puis decode()
        content = chunks_to_draft_content.read().decode("utf-8")
        for line in content.split("\n"):
            if line.strip():
                file_name, part_id = line.strip().split(",")
                # Conserver le nom original avec accents
                if file_name not in suggested_chunks:
                    suggested_chunks[file_name] = []
                suggested_chunks[file_name].append(int(part_id))
        st.write("Chunks suggérés par GuessStrategyTool :")
        for file_name, parts in suggested_chunks.items():
            st.write(f"{file_name}: {', '.join(str(p) for p in parts)}")

    # Interface pour spécifier les chunks à rédiger par fichier
    user_chunks_to_draft = []
    all_chunks_to_draft = {}  # Pour stocker les choix "Tout rédiger"
    st.write("Spécifiez les numéros des chunks à rédiger pour chaque fichier :")
    chunk_inputs = {}
    for file_path in file_paths:
        file_name = os.path.basename(file_path)  # Récupérer le nom original sans temp_dir
        default_value = ", ".join(str(p) for p in suggested_chunks.get(file_name, [])) if file_name in suggested_chunks else ""
        chunk_inputs[file_name] = st.text_input(f"Chunks pour {file_name} (ex. 1, 3, 5)", value=default_value)
        all_chunks_to_draft[file_name] = st.checkbox(f"Rédiger tous les chunks pour {file_name}")

    if st.button("Générer les travaux"):
        # Parser les entrées utilisateur
        for file_name, input_text in chunk_inputs.items():
            if all_chunks_to_draft[file_name]:
                # Si "Tout rédiger" est coché, ajouter tous les chunks disponibles
                file_processor = FileProcessingTool()
                all_chunks = file_processor._run([f for f in file_paths if os.path.basename(f) == file_name])
                if all_chunks:
                    chunks_by_file = {chunk["source"]: [chunk["part_id"] for chunk in all_chunks if chunk["source"] == file_name] for chunk in all_chunks}
                    user_chunks_to_draft.extend((file_name, part_id) for part_id in chunks_by_file.get(file_name, []))
            elif input_text:
                part_ids = [int(p.strip()) for p in input_text.split(",") if p.strip().isdigit()]
                user_chunks_to_draft.extend((file_name, part_id) for part_id in part_ids)
        
        if user_chunks_to_draft:
            # Lancer DirectDraftingTool
            try:
                tool = DirectDraftingTool(llm_provider="xai")
                result = tool._run(file_paths, file_infos, project_synthesis, user_chunks_to_draft)
                st.write("Résultat :")
                st.text(result)

                # Proposer le téléchargement
                st.download_button("Télécharger les travaux", result, file_name="works_output.txt")
            except Exception as e:
                st.error(f"Erreur lors de la génération des travaux : {str(e)}")
        else:
            st.write("Veuillez spécifier au moins un chunk à rédiger ou cocher 'Rédiger tous les chunks' pour un fichier.")

        # Nettoyer le répertoire temporaire
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)  # Supprimer temp_dir et tous ses fichiers
elif not project_synthesis:
    st.warning("Veuillez entrer la synthèse du projet (obligatoire).")


# Section 7 : DraftingTool avec champs dynamiques
st.header("Rédaction de Sections avec DraftingTool")

# Définir un dictionnaire pour associer les numéros de sections à leurs noms
section_names = {
    "general": "Rédiger en suivant un exemple",
    "1.1": "Contexte et besoin objectif",
    "1.2": "Étude de marché",
    "1.3": "Innovation Produit et Progrès",
    "1.5": "Indicateurs ou conclusion sur l'innovation",
    "1.6": "Présentation de l'entreprise",
    "1.7": "Présentation des activités d'innovation"
}

# Utiliser un selectbox avec les numéros et noms des sections
section = st.selectbox(
    "Choisir la section à rédiger",
    options=[f"{key} - {section_names[key]}" for key in section_names.keys()],
    format_func=lambda x: x,  # Afficher le texte complet dans le selectbox
    key="draft_section"
)

# Extraire uniquement le numéro de la section pour le passer à DraftingTool
selected_section = section.split(" - ")[0] if " - " in section else section

# Définir les champs dynamiques selon la section
content_to_draft = st.text_area(
    "Contenu à rédiger (optionnel pour 1.1, 1.2, 1.3, 1.5, 1.7)",
    "",
    key="draft_content"
) if selected_section in ["general", "1.1", "1.2", "1.3", "1.5", "1.7"] else None
synthesis = st.text_area(
    "Synthèse de la solution (requis pour 1.1, 1.7)",
    "",
    key="draft_synthesis"
) if selected_section in ["1.1", "1.7"] else None
solution_name = st.text_input(
    "Nom de la solution (requis pour 1.2, 1.3)",
    "",
    key="draft_solution"
) if selected_section in ["1.2", "1.3"] else None
company_name = st.text_input(
    "Nom de l'entreprise (requis pour 1.6)",
    "",
    key="draft_company"
) if selected_section == "1.6" else None
example_text = st.text_area(
    "Exemple de texte à imiter (requis pour general)",
    "",
    key="draft_example"
) if selected_section == "general" else st.text_area(
    "Exemple de texte à imiter (optionnel)",
    "",
    key="draft_example"
) if selected_section != "general" else None

if st.button("Rédiger la section", key="draft_button"):
    if selected_section == "general" and not example_text:
        st.warning("Un exemple de texte est requis pour le mode général.")
    elif selected_section == "1.6" and not company_name:
        st.warning("Le nom de l'entreprise est requis pour la section 1.6.")
    elif selected_section in ["1.1", "1.7"] and not synthesis:
        st.warning("Une synthèse est requise pour les sections 1.1 et 1.7.")
    elif selected_section in ["1.2", "1.3"] and not solution_name:
        st.warning("Le nom de la solution est requis pour les sections 1.2 et 1.3.")
    else:
        try:
            tool = DraftingTool()
            result = tool._run(
                content_to_draft=content_to_draft if content_to_draft else "",
                synthesis=synthesis if synthesis else "",
                section=selected_section,
                solution_name=solution_name if solution_name else "",
                company_name=company_name if company_name else "",
                example_text=example_text if example_text else None,
                llm_provider=llm_provider
            )
            st.subheader(f"Texte Rédigé - Section {selected_section} - {section_names[selected_section]}")
            st.write(result)
        except Exception as e:
            st.error(f"Erreur lors de la rédaction : {str(e)}")
            if "SearchTool" in str(e):
                st.warning("La section 1.6 nécessite SearchTool, qui n'est pas implémenté. Résultat limité à la synthèse fournie.")

# Instructions pour exécuter
st.write("**Instructions** : Assurez-vous d'avoir installé les dépendances (`streamlit`, `python-docx`, `PyPDF2`, `openai`, `crewai`, `pandas`, `python-pptx`) et défini les clés API (`XAI_API_KEY` ou `OPENAI_API_KEY`).")
st.write("Pour exécuter localement : `streamlit run app.py`.")

