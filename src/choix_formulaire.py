"""
Étape 7a : choisir le(s) formulaire(s) à proposer à l'utilisateur,
à partir de la réponse du LLM et des extraits trouvés.
"""
import csv
import re
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_FORMULAIRES = DOSSIER_PROJET / "formulaires"
FICHIER_CSV = DOSSIER_FORMULAIRES / "formulaires.csv"
SEUIL = 0.83


def charger_correspondances():
    """Lit formulaires.csv : fiche de procédure -> liste de (fichier, nom affiché)."""
    correspondances = {}
    with open(FICHIER_CSV, encoding="utf-8-sig") as f:
        for ligne in csv.DictReader(f):
            fiche = Path(ligne["procedure"].strip()).stem
            formulaire = (ligne["formulaire"].strip(), ligne["nom_affiche"].strip())
            correspondances.setdefault(fiche, []).append(formulaire)
    return correspondances


def fiches_citees(reponse):
    """Récupère les noms des PDF cités entre crochets dans la réponse."""
    noms = re.findall(r"\[([^\[\],]+?\.pdf)", reponse)
    return [Path(nom).stem for nom in noms]


def trouver_formulaires(reponse, resultats, correspondances):
    """Renvoie la liste des formulaires à proposer (sans doublons)."""
    # 0) Si le chatbot n'a pas trouvé la réponse : aucun formulaire
    if "je ne trouve pas" in (reponse or "").lower():
        return []

    # 1) Les fiches citées par le modèle qui ont un formulaire
    fiches = [f for f in fiches_citees(reponse) if f in correspondances]

    # 2) Sinon : seulement le 1er résultat, s'il a un formulaire et un score assez haut
    if not fiches and resultats:
        premier = Path(resultats[0]["source"]).stem
        if premier in correspondances and resultats[0]["score"] >= SEUIL:
            fiches = [premier]

    formulaires = []
    for fiche in fiches:
        for formulaire in correspondances[fiche]:
            if formulaire not in formulaires:
                formulaires.append(formulaire)
    return formulaires


if __name__ == "__main__":
    from embeddings import charger_modele
    from generation import ouvrir_collection, repondre

    correspondances = charger_correspondances()
    print(f"Fiches ayant un formulaire : {len(correspondances)}")

    collection = ouvrir_collection()
    modele = charger_modele()
    tests = [("J'ai perdu ma carte d'identité, que dois-je faire ?", None),
             ("J'ai perdu mon passeport, que dois-je faire ?", "etranger"),
             ("Quel temps fera-t-il demain à Tunis ?", None)]

    for question, public in tests:
        reponse, resultats = repondre(question, collection, modele, public=public)
        formulaires = trouver_formulaires(reponse, resultats, correspondances)
        print(f"\nQuestion : {question}")
        if not formulaires:
            print("  Aucun formulaire proposé")
        for fichier, nom in formulaires:
            existe = (DOSSIER_FORMULAIRES / fichier).exists()
            print(f"  📄 {nom}  ->  {fichier}  (fichier trouvé : {existe})")