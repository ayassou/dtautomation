from crewai.tools import BaseTool
from openai import OpenAI
from typing import Optional
import os

class SearchTool(BaseTool):
    name: str = "web_search_tool"  # Annotation de type pour name
    description: str = "Outil pour effectuer une recherche web et récupérer des informations pertinentes."  # Annotation de type pour description

    def _run(self, query: str, api_key: Optional[str] = None) -> str:
        """
        Exécute une recherche web via l'API OpenAI avec web_search_preview.
        Args:
            query (str): La requête à rechercher (ex. "solutions de gestion équestre en France").
            api_key (str, optional): Clé API OpenAI. Utilise l'environnement si None.
        Returns:
            str: Résultats de la recherche sous forme de texte brut.
        """
        # Utilise la clé API passée ou celle de l'environnement
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Erreur : OPENAI_API_KEY non définie dans les variables d'environnement ou en paramètre."

        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model="gpt-4o",
                tools=[{"type": "web_search_preview"}],
                input=query
            )
            # Extrait le texte de sortie (output_text) de la réponse
            output_text = response.output_text
            if not output_text:
                return "Aucun résultat trouvé pour la requête : " + query
            return output_text
        except Exception as e:
            return f"Erreur lors de la recherche : {str(e)}"

# Exemple d'utilisation (commenté pour ne pas exécuter)
# tool = SearchTool()

# result = tool._run(
#     """
#       Voici un texte qui présente une solution en gros :

#     EKONSILIO 

# C’est quoi ? 
# On fait du marketing conversationel : chat en ligne à nos client. Pas seulement une solution technique mais aussi une solution humaine. Conseiller en ligne qui répondront en ligne aux questions des utilisateurs de nos clients.
# A la base secteur automobile, maintenant immobiliers. 
# Les deux cofondateurs avaient des connexions sur le secteur auto. On s’est étendu sur la France et à l’étranger. 
# Changement de secteurs par opportunités. Pour les conseillers, on a des équipes dédiées par secteur. Ils sont formés sur le secteur et sur le client. 
# A la création d’ekonsilio on utilisait des solutions du marché, on avait des problématiques pour respecter nos process. On fait de l’assistance, de l’accompagnement mais aussi l’objectif c’est d’établir un projet d’achat et de définir le projet avec le visiteurs pour faire un lead. 
# A l’origine on avait une solution de chat, on a développé notre solution en interne pour améliorer le travail de nos conseillers pour converser et envoyer les contacts du prospects à nos clients. 
# Suite à ça, on fait évoluer tous les jours cette solution pour améliorer le travail de nos conseillers, on rajoute de l’automatisation, de nouveaux canaux de communication. On étend nos service sur les RS. Solution multicanal. Connecter tous les canaux de nos client. 
# Solutions concurrentes 
# On a des concurrents mais pas nécessairement sur le même type de service proposé. Un seul sur le service humain mais lui utilise une solution du marché. 
# Demander le nom du concurrent par mail 
# Zendesk 
# iAdvize
# Crisp
# Userlike
# Mais ils ne proposent que la solution technique. 

# Travaux 2022 

# On a des outils de gestion sur la partie fonctionnement. Quelles sont les fonctionnalités qui ont été développées ? 
# Eléments documentaires 
# Demander un export des plus gros ticket
# Multicanal 2022 : on a fait en grande partie Facebook. On a commencé à faire des tests pour d’autres canaux comme Google Business message. 
# On a mis en place toute la partie « Stat ». On énormément de données chez nous et on a développé le tableau de bord de ces statistiques. Ce tdb sert à améliorer la qualité des échanges. 
# Répondre bien, répondre rapidement. 

#         Est-ce que la solution là est innovante ? Il y a t-il des éléments qui peuvent être considérés tel ? Si oui pourquoi ? Si non pourquoi ?
#         Peux-tu faire une analyse poussée et me dire ce qu'on peut retirer comme innovation de tout ça ? S'il y a aussi une approche, une/des fonctionnalités/techniques innovants, il faut les relever aussi.
#         Attention à ne pas forcer quelque chose. Soit franc. S'il n'y a rien d'innovant ne cache pas.
#     """
# )


# result = tool._run(
#     """
#     Ceci est un texte sur une solution : 
#     A ce jour, aucune plateforme ne propose les services de markéting conversationnel qu’offre eKonsilio au travers de son équipe formé spécifiquement pour les domaines de l’automobile et de l’immobilier et sa solution digitale qui a pour objectif de rassembler tous les outils nécessaires à l’engagement des clients sur ces secteurs spécifiques. eKonsilio se distingue de ses concurrents par les innovations suivantes qui améliorent les performances par rapport au marché :
# •	Fonctionnalités : 
# o	Prise en charge des processus de vente complexes : Les cycles de vente dans l'automobile et l'immobilier peuvent être longs et impliquer de multiples étapes. La plateforme eKonsilio est conçue pour accompagner les clients tout au long de ces processus, en offrant une assistance personnalisée à chaque étape.
# o	Intégration CRM spécialisée : eKonsilio offre une intégration CRM spécifique aux secteurs de l'automobile et de l'immobilier, permettant aux professionnels de gérer efficacement les informations clients, les leads, les biens immobiliers et les transactions, tout en simplifiant les tâches administratives.
# o	Interaction multicanale : En reconnaissant que les clients peuvent préférer divers canaux de communication, eKonsilio prend en charge une gamme complète de canaux, y compris WhatsApp, Messenger, SMS, et d'autres, pour atteindre les clients là où ils se trouvent.
# o	Personnalisation des interactions : La plateforme permet une personnalisation avancée des interactions avec les clients en fonction de leurs besoins spécifiques dans le domaine de l'automobile ou de l'immobilier, renforçant ainsi l'engagement et la satisfaction client.
# eKonsilio offre aussi une équipe d'experts spécialement formés aux particularités de l'automobile et de l'immobilier. Ces experts comprennent les réglementations, les normes de l'industrie, les tendances du marché et sont capables de fournir des réponses précises aux questions des clients. C’est cette expertise aussi qui leur permet de développer une solution digitale qui cadre bien avec les besoins de ces domaines-là.

# L’innovation d’eKonsilio concerne donc sa plateforme spécialisée qui prend en compte les spécificités des secteurs de l'automobile et de l'immobilier. Cette spécialisation permet aux professionnels de ces industries de bénéficier d'une solution conçue sur mesure pour leurs besoins uniques, offrant une expérience client exceptionnelle et renforçant leur efficacité opérationnelle. C'est cette spécificité qui distingue eKonsilio des solutions plus génériques du marché.

# Penses-tu que vraiment les éléments cités là sont innovants? 
#     """               
# )

# result = tool._run(
#     """
#     Voici une synthèse qu'on a pour une solution : 
# Ekonsilio est une entreprise spécialisée dans le marketing conversationnel, active dans les secteurs automobile et immobilier, avec une expansion en France et à l’international. Elle se distingue de concurrents comme Zendesk, iAdvize, Crisp et Userlike, qui offrent des solutions techniques, ainsi que d’un concurrent direct sur le service humain. Son innovation réside dans une solution de chat développée en interne, intégrant automatisation, multicanaux et connexion des canaux clients pour optimiser le travail des conseillers. 
# L’objectif est de définir les projets d’achat des visiteurs pour générer des leads. En 2022, Ekonsilio a enrichi ses outils avec des fonctionnalités de gestion, l’intégration de canaux comme Facebook et Google Business Messages, et un tableau de bord statistique pour améliorer la qualité et la rapidité des échanges. Elle propose également des équipes de conseillers dédiées par secteur, formées pour accompagner les clients dans leurs projets d’achat.
# Les travaux de l’année se sont concentrés sur plusieurs axes de développement et d’amélioration des fonctionnalités. En matière de gestion des utilisateurs et profils, des avancées ont été réalisées dans l’édition des profils et l’affichage des détails, avec une transition vers React-Query pour optimiser les requêtes API, la mise en place de tests unitaires et la résolution de bugs. 
# Concernant l’affichage et la gestion des leads, les efforts ont porté sur la structuration et la présentation des informations via des requêtes API, en surmontant des contraintes d’interface pour un rendu clair et opérationnel. Pour la configuration et gestion des conteneurs et champs, des fonctionnalités d’ajout, de modification et de suppression ont été développées avec des systèmes de drag & drop et des interfaces intuitives, malgré des défis techniques résolus par des ajustements ciblés. 
# En termes de navigation et expérience utilisateur, des améliorations ont été apportées via l’ajout de breadcrumbs, la gestion des états de chargement et d’erreur, et des optimisations visuelles pour une interaction plus fluide. La gestion des entités et interfaces a vu des refontes de listes et des correctifs pour une meilleure maintenance et expérience utilisateur. 
# Les travaux sur la correction de bugs et gestion des erreurs ont permis de renforcer la robustesse des interfaces grâce à des messages d’erreur clairs et des ajustements de code. 
# Enfin, des évolutions dans les modèles de leads et expéditeurs ont facilité la sélection et la gestion des expéditeurs, avec des corrections de bugs et des validations réussies. Ces efforts ont globalement abouti à des résultats fonctionnels et concluants, malgré des défis techniques et des besoins d’ajustements continus.
    
#     Peux-tu analyser ceci et me dire s'il y a un élément ou des éléments innovants de la solution ? Si oui pourquoi ? Si non aussi tu expliques pourquoi ?
    
#     """
# )

# result = tool._run(
#     """
#     Voici la description d'une solution : 
#     "element_diff": "Ekonsilio se distingue par le développement interne d'une solution de chat innovante qui intègre l'automatisation, les fonctionnalités multicanaux, et connecte les différents canaux clients pour optimiser le travail des conseillers. Cette solution vise à améliorer la définition des projets d’achat des visiteurs, et à générer efficacement des leads dans les secteurs automobile et immobilier.",
#     "elements_innovants": 
#       "Développement interne d’une solution de chat intégrant automatisation et multicanaux.",
#       "Intégration de nouveaux canaux de communication comme Facebook et Google Business Messages.",
#       "Mise en place d’un tableau de bord statistique pour améliorer la qualité et la rapidité des échanges.",
#       "Ajout de fonctionnalités de gestion des utilisateurs avec optimisation des requêtes API via React-Query.",
#       "Création d’interfaces utilisateur intuitives avec drag & drop pour la configuration des conteneurs et champs.",
#       "Améliorations UX/UI incluant des breadcrumbs et gestion des états de chargement.",
#       "Équipes de conseillers dédiées et formées par secteur pour accompagner les clients."

#     A l'égard des autres solutions du marché, penses-tu que l'élément différenciant soit pertinent ?
#     A l'égard du marché, et du domaine, ces fonctionnalités innovantes sont-elles vraiment innovantes ?
#     Fais une analyse un par un
#     Si tu trouves un point qui n'est pas innovant, dis le franchement avec les raisons.
#     """
# )

# result = tool._run(
#     """
# ceci est notre solution : https://www.citykomi.com/
# 1- Analyse si la solution est innovante à l'égard des solutions similaires sur le marché
# 2- Si elle est innovante, fourni une comparaison avec les 3 solutions concurrentes :
#    -Pour chaque solution, tu la présentes
#    - Tu listes ses fonctionnalités
#    - Tu montres en quoi citykomi est différent 

# 2- Si elle est innovante, Montre exactement là où est innovante, ce qui la rend innovante
# 3- Si la solution possède des éléments innovants comme des fonctionnalités, ou sur le plan technique, Liste-les!
# """)
# result = tool._run("""
# Voici notre solution Citikomi : https://www.citykomi.com/ 
# Son point innovant est ceci : L’innovation de Citykomi réside dans le fait qu’aucune donnée personnelle de l’utilisateur n’est collectée. Citykomi a obtenu en effet un brevet à ce sujet, ce qui garantit le complet anonymat des utilisateurs et la collecte d’aucune donnée personnelle. 
# Ses fonctionnalités innovantes sont ceci : o	Plateforme unique pour tous les diffuseurs : pas besoin de téléchargements supplémentaires pour s’abonner à plusieurs flux. La solution permettra aux utilisateurs de s’abonner à plusieurs mairies, collectivités et services via une seule interface
# o	Personnalisable : l’utilisateur ne s’abonne qu’aux flux d’informations qui l’intéressent. Il choisit les collectivités ou les thèmes pour lesquels il souhaite être alerté.

# Peux-tu chercher les solutions qui ressemblent à notre solution sur le marché ? Et :
# 1- Nous confirmer s'il est vraiment le seul qui possède une telle innovation ?
# 2- Offrir une comparaison avec les 3 solutions concurrentes :
#   -Pour chaque solution, tu la présentes
#   - Tu listes ses fonctionnalités
#   - Tu montres en quoi citykomi est différent 
# """)
# print(result)
