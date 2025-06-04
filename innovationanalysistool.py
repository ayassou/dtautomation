from crewai.tools import BaseTool
from openai import OpenAI
from typing import Optional
import os
from searchtool import SearchTool

class InnovationAnalysisTool(BaseTool):
    name: str = "innovation_analysis_tool"  # Nom de l'outil avec annotation de type
    description: str = "Outil pour analyser si une solution est innovante par rapport au marché en utilisant des données du site de la solution si fourni."  # Description mise à jour

    def _run(self, synthesis: str, solution_name: str, company_name: str, website_url: Optional[str] = None, llm_provider: str = "xai") -> str:
        """
        Analyse si la solution décrite dans la synthèse est innovante par rapport au marché.
        Args:
            synthesis (str): Contenu de la synthèse (ex. synthese.txt).
            solution_name (str): Nom de la solution (ex. "Ekonsilio Chat").
            company_name (str): Nom de l'entreprise (ex. "Ekonsilio").
            website_url (str, optional): URL du site de la solution (ex. "www.ekonsilio.com"). Si None, aucune recherche n'est effectuée.
            llm_provider (str): Fournisseur du LLM ("xai" ou "openai"). Par défaut "xai".
        Returns:
            str: Analyse sous forme de texte structuré.
        """
        llm_provider = llm_provider.lower()
        # Configure le client selon le fournisseur
        if llm_provider == "xai":
            api_key = os.getenv("XAI_API_KEY")
            base_url = "https://api.x.ai/v1"
            model_id = "grok-3-beta"  # Modèle générique pour xAI, à ajuster si nécessaire
        else:  # openai
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = None  # OpenAI utilise l'URL par défaut
            model_id = "gpt-4o"  # Modèle valide pour OpenAI
        if not api_key:
            return f"Erreur : Clé API pour {llm_provider.upper()}_API_KEY non définie dans les variables d'environnement"

        client = OpenAI(api_key=api_key, base_url=base_url)

        # Étape 1 : Recherche sur le site de la solution si un URL est fourni
        website_info = ""
        if website_url:
            search_tool = SearchTool()
            website_info = search_tool._run(query=f"fonctionnalités et caractéristiques de la solution de {company_name} sur {website_url}")
            if "Erreur" in website_info or "Aucun résultat" in website_info:
                website_info = "NA"

        # Étape 2 : Construction du prompt avec ou sans les données du site
        prompt_base = f"Voici une synthèse de la solution {solution_name} de l'entreprise {company_name} : \n{synthesis}\n\n"
        if website_info == 'NA':
          prompt_base
        else :  
            prompt_base += f"Voici les informations recueillies sur le siteweb : \n{website_info}\n\n"

        # Prompt pour analyser directement la synthèse et comparer au marché
        analysis_prompt = (
            prompt_base +
            f"Vous êtes un analyste stratégique spécialisé dans l'innovation technologique."
            # "Vous êtes dans le cadre du crédit d'impôt innovation où il faut analyser les capacités d'innovation des solutions"
            "Analysez si la solution est innovante par rapport au marché et aux pratiques standards dans le secteur de la solution. "
            "Soyez très critique et franc : si rien n'est innovant, dites-le clairement, sois franc. "
            "Identifiez un élément différenciant et innovant, s'il existe, et expliquez pourquoi il est innovant. "
            "Ensuite, identifiez toutes les fonctionnalités ou approches spécifiques de la solution qui sont innovantes, "
            "et fournissez une explication pour chaque sur pourquoi elles sont considérées comme innovantes. "
            "Si une piste d'innovation potentielle existe (même si elle n'est pas pleinement développée), mentionnez-la. "
            "Retournez le résultat sous forme de texte structuré comme suit : \n"
            "Élément différenciant et innovant : [Description et justification]\n"
            "Fonctionnalités innovantes :\n- [Fonctionnalité 1] : [Explication]\n- [Fonctionnalité 2] : [Explication]\n..."
            "Si aucun élément innovant n'est trouvé, indiquez simplement : \n"
            "'Aucun élément différenciant ou innovant identifié.\n"
            "Dans le cas où aucune innovation claire n'a été identifiée, tu refais une analyse plus profonde pour proposer une piste d'innovation.\n" 
            "Par piste d'innovation, j'entend aller encore plus en profondeur pour voir si leur approche n'est pas particulier, s'il n'y a rien côté technicité qui se distingue, s'il n'y a rien niveau concept. S'il n'y a pas une combinaison que propose la solution.\n"
            "Concernant la piste d'innovation, ATTENTION à ne rien inventer. NE PROPOSE UN MOYEN POUR RENDRE LA SOLUTION INNOVANTE, NON. TU ANALYSES JUSTE LA SOLUTION A L'ETAT ACTUEL POUR VOIR CE QUI POURRAIT ETRE INNOVANT DEDANS"
            "Tu refouilles bien dans les informations pour voir quel est l'élément qui peut être considéré comme une innovation (pas forcément innovant, mais peut l'être) si on le regarde sous un autre angle différent"
            # "Il faut que tu trouves un élément qui peut être considérer comme innovant (si on change de perspective)."
            # "Tu mentionnes cette perspective là aussi"
            "Si une telle innovation (probable) est trouvée, tu réinterprètes la solution ET TU REPROPOSES FORCEMENT 02-03 éléments comme innovants de la solution MAIS que tu représentes sous la NOUVELLE perspective.\n"
            "Attention à ne rien inventer malgré tout. Si après tout, il n'y a toujours pas d'élement innovant, alors tu le dis."
            "Piste d'Innovation : [Description]'"
        )

        try:
            analysis_response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "Vous êtes un analyste stratégique spécialisé dans l'innovation technologique."},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            analysis_text = analysis_response.choices[0].message.content.strip()
            return analysis_text
        except Exception as e:
            return f"Erreur lors de l'analyse d'innovation : {str(e)}"


# Exemple d'utilisation (commenté pour ne pas exécuter)
# tool = InnovationAnalysisTool()
# with open("synthese.txt", "r", encoding="utf-8") as f:
#     synthesis = f.read().strip()
# result = tool._run(synthesis, solution_name="Ekonsilio Chat", 
#                    company_name="Ekonsilio", 
#                   #  website_url="www.ekonsilio.com", 
#                    llm_provider="xai")
# print(result)
