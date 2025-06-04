import os
import logging
from docx import Document
from openai import OpenAI
from crewai.tools import BaseTool

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SynthesisTool(BaseTool):
    name: str = "synthesis_tool"
    description: str = "Outil pour générer une synthèse structurée à partir d'un fichier Word contenant des informations dispersées."
    llm_provider: str = "LLm provider"  # Fournisseur LLM par défaut

    def __init__(self, llm_provider: str = "xai"):
        """
        Initialise l'outil avec un fournisseur LLM.

        Args:
            llm_provider (str): Fournisseur du LLM ("xai" ou "openai"). Par défaut "xai".
        """
        super().__init__()
        self.llm_provider = llm_provider.lower()
        logger.info(f"SynthesisTool initialisé avec llm_provider : {self.llm_provider}")

    def _run(self, file_path: str) -> str:
        """
        Génère une synthèse structurée à partir d'un fichier Word.

        Args:
            file_path (str): Chemin vers le fichier Word (.docx).

        Returns:
            str: Synthèse structurée sous forme de texte.
        """
        logger.info(f"Début de la synthèse pour le fichier : {file_path}")

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
            logger.error(f"Clé API pour {self.llm_provider.upper()}_API_KEY non définie.")
            return f"Erreur : Clé API pour {self.llm_provider.upper()}_API_KEY non définie."

        client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info(f"LLM configuré avec provider : {self.llm_provider}")

        # Étape 2 : Lire le fichier Word
        try:
            doc = Document(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():  # Ignorer les paragraphes vides
                    full_text.append(para.text.strip())
            raw_content = "\n".join(full_text)
            logger.info(f"Contenu extrait du fichier Word : {raw_content[:200]}... (tronqué)")
        except Exception as e:
            logger.error(f"Erreur lors de la lecture du fichier {file_path} : {str(e)}")
            return f"Erreur lors de la lecture du fichier : {str(e)}"

        if not raw_content:
            logger.warning("Le fichier Word est vide.")
            return "Erreur : Le fichier Word est vide."

        # Étape 3 : Générer une synthèse structurée avec le LLM
        synthesis_prompt = (
            "Vous êtes un expert en synthèse de documents. À partir du contenu suivant extrait d'un fichier Word :\n"
            f"'{raw_content}'\n\n"
            "Générez une synthèse structurée en trois parties :\n"
            "1. **Introduction** : Résumez le contexte général et l'objectif principal des informations.\n"
            "2. **Points Clés** : Identifiez et listez les informations essentielles sous forme de puces.\n"
            # "3. **Conclusion** : Fournissez une conclusion concise sur les implications.\n"
            "La synthèse doit être concise (maximum 300 mots) et claire."
        )

        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": synthesis_prompt}],
                max_tokens=400,
                temperature=0.4
            )
            structured_synthesis = response.choices[0].message.content.strip()
            logger.info(f"Synthèse générée avec succès : {structured_synthesis[:200]}... (tronqué)")
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la synthèse : {str(e)}")
            return f"Erreur lors de la génération de la synthèse : {str(e)}"

        # Étape 4 : Sauvegarder la synthèse dans un fichier
        output_file = "structured_synthesis.txt"
        try:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(structured_synthesis)
            logger.info(f"Synthèse sauvegardée dans {output_file}")
        except Exception as e:
            logger.error(f"Erreur lors de l'écriture de la synthèse : {str(e)}")
            return f"Erreur lors de l'écriture de la synthèse : {str(e)}"

        return structured_synthesis

# Exemple d'utilisation
if __name__ == "__main__":
    tool = SynthesisTool(llm_provider="xai")
    result = tool._run("example.docx")
    print(result)
