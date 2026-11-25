"""
Étape 7a : choisir le(s) formulaire(s) à proposer à l'utilisateur,
à partir de la réponse du LLM et des extraits trouvés.
"""
import csv                  # pour lire formulaires.csv (le tableau fiche -> formulaire)
import re                   # expressions régulières (trouver les citations dans la réponse)
from pathlib import Path    # pour manipuler les chemins de fichiers sous Windows comme sous Linux

# __file__ = ce fichier (src/choix_formulaire.py)
# .resolve() = chemin complet ; .parent = src/ ; .parent.parent = le dossier du projet
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_FORMULAIRES = DOSSIER_PROJET / "formulaires"      # « / » colle les morceaux du chemin
FICHIER_CSV = DOSSIER_FORMULAIRES / "formulaires.csv"
# Score minimum pour proposer un formulaire quand le LLM n'a cité aucune fiche.
# Mesuré sur l'évaluation : la meilleure question hors sujet avait 0,838,
# donc avec 0,84 une question hors sujet ne reçoit jamais de formulaire.
SEUIL = 0.84   # juste au-dessus du meilleur score des questions hors sujet (0,838)


def charger_correspondances():
    """Lit formulaires.csv : fiche de procédure -> liste de (fichier, nom affiché)."""
    correspondances = {}          # dictionnaire vide : { "nom_de_la_fiche": [(fichier, nom), ...] }
    # encoding="utf-8-sig" : lit les accents ET enlève le caractère invisible (BOM)
    # qu'Excel ajoute au début des CSV
    with open(FICHIER_CSV, encoding="utf-8-sig") as f:
        # DictReader : chaque ligne devient un dictionnaire avec les noms de colonnes
        # (ligne["procedure"], ligne["formulaire"], ligne["nom_affiche"])
        for ligne in csv.DictReader(f):
            # .strip() enlève les espaces en trop ; .stem = le nom sans « .pdf »
            fiche = Path(ligne["procedure"].strip()).stem
            # un formulaire = un couple (nom du fichier PDF, nom à afficher dans l'app)
            formulaire = (ligne["formulaire"].strip(), ligne["nom_affiche"].strip())
            # setdefault : crée une liste vide si la fiche est nouvelle, puis ajoute le formulaire
            # (une même fiche peut avoir plusieurs formulaires)
            correspondances.setdefault(fiche, []).append(formulaire)
    return correspondances


def fiches_citees(reponse):
    """Récupère les noms des PDF cités entre crochets dans la réponse."""
    # La réponse contient des citations comme [CIN_Perte.pdf, p. 1].
    # \[            : un crochet ouvrant
    # ( ... )       : la partie qu'on garde (le nom du fichier)
    # [^\[\],]+?    : des caractères qui ne sont ni crochet ni virgule (le « ? » = le moins possible)
    # \.pdf         : qui finissent par « .pdf »
    noms = re.findall(r"\[([^\[\],]+?\.pdf)", reponse)
    return [Path(nom).stem for nom in noms]       # on enlève « .pdf » pour comparer avec le CSV


def trouver_formulaires(reponse, resultats, correspondances):
    """Renvoie la liste des formulaires à proposer (sans doublons)."""
    # 0) Si le chatbot n'a pas trouvé la réponse : aucun formulaire
    # (reponse or "") : évite une erreur si reponse vaut None ; .lower() = en minuscules
    if "je ne trouve pas" in (reponse or "").lower():
        return []

    # 1) Les fiches citées par le modèle qui ont un formulaire
    # (liste en compréhension : on garde seulement les fiches présentes dans le CSV)
    fiches = [f for f in fiches_citees(reponse) if f in correspondances]

    # 2) Sinon : seulement le 1er résultat, s'il a un formulaire et un score assez haut
    if not fiches and resultats:                       # aucune fiche citée, mais il y a des résultats
        premier = Path(resultats[0]["source"]).stem    # le meilleur extrait trouvé par la recherche
        # les deux conditions : la fiche a un formulaire ET le score dépasse le seuil
        if premier in correspondances and resultats[0]["score"] >= SEUIL:
            fiches = [premier]

    # On rassemble les formulaires de toutes les fiches retenues, sans doublons
    # (deux fiches peuvent avoir le même formulaire)
    formulaires = []
    for fiche in fiches:
        for formulaire in correspondances[fiche]:
            if formulaire not in formulaires:          # pas encore dans la liste -> on l'ajoute
                formulaires.append(formulaire)
    return formulaires


# S'exécute seulement quand on lance « python src/choix_formulaire.py » : 3 questions de test
if __name__ == "__main__":
    # Importés ici seulement : app.py importe ce fichier sans recharger tout ça deux fois
    from embeddings import charger_modele
    from generation import ouvrir_collection, repondre

    correspondances = charger_correspondances()
    print(f"Fiches ayant un formulaire : {len(correspondances)}")

    collection = ouvrir_collection()
    modele = charger_modele()
    # (question, zone) : une question qui doit donner un formulaire, une à l'étranger, une hors sujet
    tests = [("J'ai perdu ma carte d'identité, que dois-je faire ?", None),
             ("J'ai perdu mon passeport, que dois-je faire ?", "etranger"),
             ("Quel temps fera-t-il demain à Tunis ?", None)]

    for question, public in tests:
        # toute la chaîne RAG, puis le choix du formulaire
        reponse, resultats = repondre(question, collection, modele, public=public)
        formulaires = trouver_formulaires(reponse, resultats, correspondances)
        print(f"\nQuestion : {question}")
        if not formulaires:
            print("  Aucun formulaire proposé")
        for fichier, nom in formulaires:               # chaque formulaire est un couple (fichier, nom)
            # on vérifie que le PDF existe vraiment dans le dossier formulaires/
            existe = (DOSSIER_FORMULAIRES / fichier).exists()
            print(f"  📄 {nom}  ->  {fichier}  (fichier trouvé : {existe})")