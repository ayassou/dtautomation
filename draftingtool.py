from crewai.tools import BaseTool
from openai import OpenAI
from typing import Optional
import os
from searchtool import SearchTool

class DraftingTool(BaseTool):
    name: str = "drafting_tool"  # Nom de l'outil avec annotation de type
    description: str = "Outil pour rédiger des sections spécifiques d'un rapport (1.1, 1.2, 1.3, 1.5, 1.6, 1.7) ou un texte général en suivant un style donné."  # Description mise à jour

    def _run(self, content_to_draft: str = "", synthesis: str = "", section: str = "general", solution_name: str = "", company_name: str = "", example_text: Optional[str] = None, llm_provider: str = "xai") -> str:
        """
        Rédige une section spécifique d'un rapport ou un texte général en suivant un style donné.
        Args:
            content_to_draft (str, optional): Contenu principal à rédiger (ex. sortie de MarketStudyTool ou InnovationAnalysisTool).
            synthesis (str, optional): Synthèse de la solution (pour 1.1, 1.7).
            section (str): Section à rédiger ("general", "1.1", "1.2", "1.3", "1.5", "1.6", "1.7"). Par défaut "general".
            solution_name (str, optional): Nom de la solution (pour 1.2, 1.3).
            company_name (str, optional): Nom de l'entreprise (pour 1.6).
            example_text (str, optional): Exemple de rédaction pour imiter le style (si fourni).
            llm_provider (str): Fournisseur du LLM ("xai" ou "openai"). Par défaut "xai".
        Returns:
            str: Texte rédigé pour la section demandée ou texte général.
        """
        llm_provider = llm_provider.lower()
        # Configure le client selon le fournisseur
        if llm_provider == "xai":
            api_key = os.getenv("XAI_API_KEY")
            base_url = "https://api.x.ai/v1"
            model_id = "grok-3-beta"  # Modèle générique pour xAI
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None  # OpenAI utilise l'URL par défaut
            model_id = "gpt-4o"  # Modèle valide pour OpenAI
        if not api_key:
            return f"Erreur : Clé API pour {llm_provider.upper()}_API_KEY non définie dans les variables d'environnement"

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Étape 1 : Recherche web pour la section 1.6 si nécessaire
        web_data = ""
        if section == "1.6":
            if not company_name:
                return "Erreur : Nom de l'entreprise requis pour la section 1.6."
            search_tool = SearchTool()
            web_data = search_tool._run(query=f"presentation of the company {company_name} including history, values, projects, and solutions")
            if "Erreur" in web_data or "Aucun résultat" in web_data:
                web_data = "Aucune donnée disponible sur l'entreprise."

        # Étape 2 : Construction du prompt selon la section
        base_prompt = f"Vous êtes un rédacteur professionnel spécialisé dans les rapports stratégiques."
        if section != "general":
            base_prompt += f" Rédigez la section {section} d'un rapport."
        if example_text:
            base_prompt += (
                f" Adaptez le style, la structure et le ton de l'exemple suivant : \n{example_text}\n\n"
                "Assurez-vous que le texte rédigé respecte ces caractéristiques tout en suivant les instructions spécifiques ci-dessous."
            )

        if section == "general":
            # Mode général : Rédaction d'un contenu quelconque en suivant un exemple
            if not example_text:
                return "Erreur : Un exemple de rédaction est requis pour le mode général."
            drafting_prompt = (
                f"{base_prompt} "
                "Votre tâche est de rédiger un texte en suivant le style, la structure et le ton de l'exemple fourni. "
                f"Voici le contenu à rédiger : \n{content_to_draft}\n\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.1":
            # Section 1.1 - Contexte et besoin objectif
            drafting_prompt = (
                f"{base_prompt} "
                "Parlez du pourquoi du projet, du besoin auquel il cherche à répondre, "
                "ce qui manquait et que le projet cherche à combler, de ce en quoi le projet consiste, "
                "et de comment le projet cherche à répondre au besoin. Mentionnez également "
                "ce qui en fait la principale innovation. "
                f"Voici la synthèse de la solution : \n{synthesis}\n\n"
                f"Voici les informations supplémentaires (sortie de MarketStudyTool) : \n{content_to_draft}\n\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.2":
            # Section 1.2 - Étude de marché
            if not solution_name:
                return "Erreur : Nom de la solution requis pour la section 1.2."
            drafting_prompt = (
                f"{base_prompt} "
                f"Rédigez une étude de marché pour la solution {solution_name}. "
                "Présentez 4 solutions concurrentes sur le marché (de préférence français). "
                "Pour chaque solution, décrivez sur plusieurs lignes : le besoin qu'elle adresse, ses fonctionnalités principales, "
                "et montrez en quoi elle est inférieure à {solution_name}. "
                f"Concluez en présentant {solution_name} comme une solution innovante qui comble ces lacunes. "
                f"Voici les informations sur les concurrents (sortie de MarketStudyTool) : \n{content_to_draft}\n\n"
                "Structurez le texte comme suit : \n"
                "- Introduction : Indiquez qu'il n'existe pas de solution identique à {solution_name}, mais mentionnez des références pertinentes.\n"
                "- Pour chaque concurrent (4 au total) :\n"
                "  - [Nom de la solution] : [Description, besoin adressé, fonctionnalités principales].\n"
                "  - Infériorité par rapport à {solution_name} : [Explication].\n"
                "- Conclusion : Présentez {solution_name} comme innovante, en expliquant comment elle surpasse les concurrents.\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.3":
            # Section 1.3 - Détail des innovations
            if not solution_name:
                return "Erreur : Nom de la solution requis pour la section 1.3."
            drafting_prompt = (
                f"{base_prompt} "
                f"Présentez tous les éléments innovants qui distinguent la solution {solution_name} de la concurrence. "
                f"Décrivez les objectifs de {solution_name} et expliquez comment ses innovations la rendent unique sur le marché. "
                "Mettez en avant les technologies, fonctionnalités ou approches qui la démarquent. "
                f"Voici les détails sur l'innovation (sortie de InnovationAnalysisTool) : \n{content_to_draft}\n\n"
                "Structurez le texte comme suit : \n"
                f"- Introduction : Présentez l'objectif principal de {solution_name} et son positionnement innovant.\n"
                "- Points d'innovation : Listez et détaillez chaque élément innovant (ex. technologies, fonctionnalités spécifiques).\n"
                "- Conclusion : Expliquez comment ces innovations redéfinissent les normes du secteur.\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.5":
            # Section 1.5 - Indicateurs ou conclusion sur l'innovation
            drafting_prompt = (
                f"{base_prompt} "
                "Faites une conclusion en utilisant les informations sur l'innovation de la solution. "
                f"Voici les informations sur l'innovation (sortie de MarketStudyTool) : \n{content_to_draft}\n\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.6":
            # Section 1.6 - Présentation de l'entreprise
            drafting_prompt = (
                f"{base_prompt} "
                f"Présentez l'entreprise {company_name}, son historique, ses valeurs, ses projets, et ses solutions. "
                f"Voici les informations récupérées sur le web : \n{web_data}\n\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        elif section == "1.7":
            # Section 1.7 - Présentation des activités d’innovation
            drafting_prompt = (
                f"{base_prompt} "
                "Présentez la solution innovante (présentation, objectif et innovations). "
                f"Voici la synthèse de la solution : \n{synthesis}\n\n"
                f"Voici les informations supplémentaires (sortie de MarketStudyTool) : \n{content_to_draft}\n\n"
                "Retourne uniquement le texte rédigé, sans commentaire ni introduction."
            )
        else:
            return "Erreur : Section non reconnue. Utilisez 'general', '1.1', '1.2', '1.3', '1.5', '1.6' ou '1.7'."

        # Étape 3 : Appel au LLM pour rédiger
        try:
            drafting_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Vous êtes un rédacteur professionnel spécialisé dans les rapports stratégiques."},
                    {"role": "user", "content": drafting_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            drafted_text = drafting_response.choices[0].message.content.strip()
            return drafted_text
        except Exception as e:
            return f"Erreur lors de la rédaction : {str(e)}"

# Exemple d'utilisation (commenté pour ne pas exécuter)
# tool = DraftingTool()
# synthesis = "Citykomi détient un brevet concernant cette innovation de non collecte de données utilisateur. De plus, Citykomi offre plus de flexibilité à l’utilisateur en lui permettant de choisir les informations dont il souhaite être alerté."
# content_to_draft = """Confirmation de l'innovation : L’innovation de Citykomi réside dans l’absence de collecte de données personnelles, protégée par un brevet.
# Comparaison avec les concurrents :
# - Solution 1 : AlertMe
#   - Fonctionnalités : Notifications en temps réel, personnalisation des alertes.
#   - Points forts : Interface intuitive, large couverture géographique.
#   - Points faibles : Collecte de données personnelles, manque d’intégration multimédia.
#   - Supériorité de Citykomi : Respect de la vie privée, intégration de documents et vidéos.
# """
# innovation_analysis = """Élément différenciant et innovant : L’innovation de Citykomi réside dans l’absence de collecte de données personnelles, protégée par un brevet.
# Fonctionnalités innovantes :
# - Diffusion d'informations en temps réel : Permet une communication instantanée.
# - Respect de la vie privée : Aucune donnée personnelle n’est collectée, ce qui est unique sur le marché.
# """
# example_text = """**Analyse stratégique**  
# La solution se distingue par une approche novatrice qui répond aux attentes actuelles du marché. Voici les points clés :  
# - **Innovation principale** : La solution garantit une confidentialité totale des utilisateurs, un atout majeur dans un secteur où la protection des données est cruciale.  
# - **Comparaison concurrentielle** : Par rapport aux alternatives, elle offre des avantages uniques. Par exemple, la solution concurrente AlertMe, bien que performante sur les notifications, ne protège pas suffisamment les données des utilisateurs, ce qui constitue une faiblesse majeure face à notre solution."""
# # Mode général
# result_general = tool._run(content_to_draft=content_to_draft, section="general", example_text=example_text, llm_provider="openai")
# print("Mode général:", result_general)
# # Section 1.1
# result_1_1 = tool._run(content_to_draft=content_to_draft, synthesis=synthesis, section="1.1", example_text=example_text, llm_provider="openai")
# print("Section 1.1:", result_1_1)
# # Section 1.2
# result_1_2 = tool._run(content_to_draft=content_to_draft, section="1.2", solution_name="Citykomi", example_text=example_text, llm_provider="openai")
# print("Section 1.2:", result_1_2)
# # Section 1.3
# result_1_3 = tool._run(content_to_draft=innovation_analysis, section="1.3", solution_name="Citykomi", example_text=example_text, llm_provider="openai")
# print("Section 1.3:", result_1_3)
# # Section 1.5
# result_1_5 = tool._run(content_to_draft=content_to_draft, section="1.5", example_text=example_text, llm_provider="openai")
# print("Section 1.5:", result_1_5)
# # Section 1.6
# result_1_6 = tool._run(section="1.6", company_name="Citykomi", example_text=example_text, llm_provider="openai")
# print("Section 1.6:", result_1_6)
# # Section 1.7
# result_1_7 = tool._run(content_to_draft=content_to_draft, synthesis=synthesis, section="1.7", example_text=example_text, llm_provider="openai")
# print("Section 1.7:", result_1_7)
