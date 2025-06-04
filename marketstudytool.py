from crewai.tools import BaseTool
from openai import OpenAI
from typing import Optional
import os
from searchtool import SearchTool

class MarketStudyTool(BaseTool):
    name: str = "market_study_tool"  # Nom de l'outil avec annotation de type
    description: str = "Outil pour réaliser une étude de marché en comparant une solution à ses concurrents."  # Description

    def _run(self, synthesis: str, web_info: str = "", innovation_analysis: str = "", solution_name: str = "Citykomi", company_name: str = "", llm_provider: str = "xai") -> str:
        """
        Réalise une étude de marché en comparant la solution à ses concurrents.
        Args:
            synthesis (str): Synthèse de la solution.
            web_info (str, optional): Informations du site web de la solution. Par défaut vide.
            innovation_analysis (str, optional): Sortie de l'analyse d'innovation. Par défaut vide.
            solution_name (str): Nom de la solution (ex. "Citykomi"). Par défaut "Citykomi".
            company_name (str): Nom de l'entreprise (ex. "Citykomi Inc"). Par défaut vide.
            llm_provider (str): Fournisseur du LLM ("xai" ou "openai"). Par défaut "xai".
        Returns:
            str: Analyse de marché structurée.
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

        # Étape 1 : Recherche de solutions similaires sur le marché
        search_tool = SearchTool()
        competitor_query = f"Voici notre solution : https://www.citykomi.com/.\n"
        f"Cherche 4 solutions similaires à {solution_name} sur le marché, la présentation, leurs fonctionnalités, points forts et faibles\n"
        "Pour chaque solution, tu lies le lien web"
        competitor_data = search_tool._run(query=competitor_query)
        if "Erreur" in competitor_data or "Aucun résultat" in competitor_data:
            competitor_data = "Aucune donnée disponible sur les concurrents."
        print(competitor_data)
        # Prompt pour l'analyse de marché
        analysis_prompt = (
            f"Voici la synthèse de notre solution {solution_name} : \n{synthesis}\n\n"
            f"Les informations sur le siteweb de la solution : \n{web_info}\n\n"
            f"L'analyse de l'innovation donne ceci : \n{innovation_analysis}\n\n"
            "Tu te bases dessus pour montrer que la solution est innovante par rapport aux autres.\n\n"
            "Peux-tu chercher les solutions qui ressemblent à notre solution sur le marché ? Et :\n"
            "1- Confirmer qu'il est innovant ?\n"
            "2- Offrir une comparaison avec 04 solutions concurrentes :\n"
            "   - Pour chaque solution: Tu la présentes\n"
                "   - Tu listes ses fonctionnalités\n"
                "   - Tu listes les points forts\n"
                "   - Tu listes les points faibles\n"
                "   - Tu montres en quoi {solution_name} est différent et supérieur/meilleur\n\n"
            "3- A la fin, tu rajoutes les noms des solutions et leur référence (lien web).\n"
            "Voici le modèle :\n"
            "Références :\n"
            "[1]	[Solution 1]: lien 1\n"
            "[2]	[Solution 2]: lien 2\n"
            "...\n"
            f"Les données sur les concurrents sont : \n{competitor_data}\n\n"
            "Retourne le résultat sous forme de texte structuré comme suit :\n"
            "Confirmation de l'innovation : [Description basée sur l'analyse d'innovation]\n"
            "Comparaison avec les concurrents :\n"
            "- [Concurrent 1] : [Présentation]\n"
            "  - Fonctionnalités : [Liste]\n"
            "  - Points forts : [Liste]\n"
            "  - Points faibles : [Liste]\n"
            "  - Supériorité de {solution_name} : [Explication]\n"
            # "- [Concurrent 2] : [Présentation]\n"
            # "  - Fonctionnalités : [Liste]\n"
            # "  - Points forts : [Liste]\n"
            # "  - Points faibles : [Liste]\n"
            # "  - Supériorité de {solution_name} : [Explication]\n"
            # "...\n"
            "Si aucune donnée sur les concurrents n'est disponible, indique : \n"
            "'Aucune donnée disponible sur les concurrents pour une comparaison.'"
        )

        try:
            analysis_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Vous êtes un analyste de marché spécialisé dans les technologies conversationnelles."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,
                temperature=0.7
            )
            analysis_text = analysis_response.choices[0].message.content.strip()
            return analysis_text
        except Exception as e:
            return f"Erreur lors de l'analyse de marché : {str(e)}"

# Exemple d'utilisation (commenté pour ne pas exécuter)
# tool = MarketStudyTool()
# synthesis = "Citykomi détient un brevet concernant cette innovation de non collecte de données utilisateur. De plus, Citykomi offre plus de flexibilité à l’utilisateur en lui permettant de choisir les informations dont il souhaite être alerté. "
# web_info = "Citykomi est une application mobile qui notifie l’utilisateur en temps réel l’information locale des collectivités, services et entreprises de sa localité. L’utilisateur est libre de choisir le type d’information qu’il reçoit."
# innovation_analysis = """L’innovation de Citykomi réside dans le fait qu’aucune donnée personnelle de l’utilisateur n’est collectée. Citykomi a obtenu en effet un brevet à ce sujet, ce qui garantit le complet anonymat des utilisateurs et la collecte d’aucune donnée personnelle. L’objectif de Citykomi est de fournir une solution de communication complète et efficace pour les collectivités et leurs populations. 
# Les fonctionnalités de Citykomi sont les suivantes :
# •	Diffusion d'informations en temps réel
# •	Service de notifications illimité
# •	Choix d'informations personnalisées
# •	Possibilité de faire des brouillons
# •	Possibilité d'intégration de documents et de liens vidéo dans les messages avec la récupération automatique des vignettes
# •	Respect de la vie privée : aucune donnée personnelle des utilisateurs n'est collectée, ce qui garantit leur vie privée
# """
# result = tool._run(synthesis, web_info=web_info, innovation_analysis=innovation_analysis, solution_name="Citykomi", company_name="Citykomi", llm_provider="openai")
# print(result)
