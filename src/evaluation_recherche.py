"""
Étape 8a : évaluation automatique de la RECHERCHE (sans le LLM).

Pour chaque question de evaluation/questions_test.csv, on regarde à quelle
position arrive le premier bon document, puis on calcule :
- rappel@3 : le bon document est-il dans les 3 premiers résultats ?
- rappel@5 : le bon document est-il dans les 5 premiers résultats ?
- MRR      : moyenne de 1 / position du premier bon document.
On compare la recherche SIMPLE (étape 5) et la recherche AMÉLIORÉE.
"""
import csv                  # lire les questions et écrire les résultats en CSV
from pathlib import Path    # manipuler les chemins de fichiers

# Nos propres fonctions : le modèle e5, l'ouverture de la base et la recherche améliorée
from embeddings import charger_modele, vectoriser_question
from generation import ouvrir_collection
from indexation import rechercher

DOSSIER_PROJET = Path(__file__).resolve().parent.parent              # src/ -> dossier du projet
FICHIER_QUESTIONS = DOSSIER_PROJET / "evaluation" / "questions_test.csv"    # entrée
FICHIER_RESULTATS = DOSSIER_PROJET / "evaluation" / "resultats_recherche.csv"  # sortie
K_MAX = 5                   # on regarde les 5 premiers résultats (assez pour rappel@3 et rappel@5)


def charger_questions():
    """Lit le fichier des questions de test."""
    # utf-8-sig : accents + enlève le caractère invisible (BOM) ajouté par Excel
    with open(FICHIER_QUESTIONS, encoding="utf-8-sig") as f:
        # DictReader : chaque ligne -> dictionnaire (q["id"], q["question"], q["fichiers_attendus"], q["public"])
        return list(csv.DictReader(f))


def rechercher_simple(collection, modele, question, k, public):
    """Recherche de l'étape 5 : les k plus proches, sans aucune amélioration."""
    # Pas d'expansion des abréviations, pas de bonus « procédure », pas de limite par document :
    # c'est la version de base, qui sert de point de comparaison.
    vecteur = vectoriser_question(modele, question)
    # Filtre Tunisie / étranger seulement si la question en a un (expression « ... if ... else ... »)
    filtre = {"public": {"$in": [public, "tous"]}} if public else None
    # .tolist() : ChromaDB attend une liste Python, pas un tableau numpy
    res = collection.query(query_embeddings=[vecteur.tolist()], n_results=k, where=filtre)
    # [0] : résultats de la 1re (et seule) question ; score = 1 - distance cosinus
    return [{"source": meta["source"], "score": 1 - distance}
            for meta, distance in zip(res["metadatas"][0], res["distances"][0])]


def position_bon_document(resultats, attendus):
    """Position (1, 2, 3...) du premier bon document, ou None s'il n'est pas trouvé."""
    # enumerate(..., start=1) : la 1re position vaut 1 (et pas 0)
    for position, r in enumerate(resultats, start=1):
        # .stem = nom sans « .pdf », pour comparer avec les noms écrits dans le CSV
        if Path(r["source"]).stem in attendus:
            return position         # on s'arrête au PREMIER bon document
    return None                     # aucun bon document dans les 5 résultats


def evaluer(nom, fonction, collection, modele, questions):
    """Calcule rappel@3, rappel@5 et MRR pour une méthode de recherche."""
    # « fonction » = la méthode de recherche à tester (simple ou améliorée) :
    # on passe une fonction en paramètre, ce qui permet d'évaluer les deux avec le même code
    lignes = []
    for q in questions:
        # Plusieurs bons documents possibles, séparés par « ; » dans le CSV
        attendus = q["fichiers_attendus"].split(";")
        public = q["public"] or None            # case vide "" -> None (pas de filtre)
        resultats = fonction(collection, modele, q["question"], K_MAX, public)
        position = position_bon_document(resultats, attendus)
        lignes.append({"methode": nom, "id": q["id"], "question": q["question"],
                       "position": position if position else "non trouvé",
                       # int(True) = 1, int(False) = 0 : 1 si trouvé dans les 3 (ou 5) premiers
                       "rappel@3": int(position is not None and position <= 3),
                       "rappel@5": int(position is not None and position <= 5),
                       # rang réciproque : 1re place -> 1, 2e -> 0,5, 3e -> 0,33... ; non trouvé -> 0
                       "rr": 1 / position if position else 0,
                       "premier_resultat": resultats[0]["source"]})
    n = len(lignes)
    print(f"\n===== {nom} ({n} questions) =====")
    # Moyenne = somme / nombre ; :.2f = 2 chiffres après la virgule
    print(f"Rappel@3 : {sum(l['rappel@3'] for l in lignes) / n:.2f}")
    print(f"Rappel@5 : {sum(l['rappel@5'] for l in lignes) / n:.2f}")
    print(f"MRR      : {sum(l['rr'] for l in lignes) / n:.2f}")   # MRR = moyenne des rangs réciproques
    # On affiche les questions ratées (bon document absent du top 3) pour les analyser
    for l in lignes:
        if not l["rappel@3"]:
            print(f"  ⚠ Q{l['id']} (position : {l['position']}) {l['question']}")
    return lignes


def scores_hors_sujet(collection, modele, questions):
    """Affiche le meilleur score obtenu par les questions hors sujet."""
    # Ces scores ont servi à fixer SEUIL = 0,84 dans choix_formulaire.py
    # (le plus haut était 0,838 : avec 0,84, aucune question hors sujet n'a de formulaire)
    print("\n===== Questions hors sujet : meilleur score =====")
    for q in questions:
        resultats = rechercher(collection, modele, q["question"], k=1)   # k=1 : seulement le meilleur
        print(f"  {resultats[0]['score']:.3f}  {q['question']}")


# S'exécute seulement avec « python src/evaluation_recherche.py »
if __name__ == "__main__":
    questions = charger_questions()
    # Une question hors sujet n'a pas de fichier attendu (case vide) : on les sépare
    dans_le_sujet = [q for q in questions if q["fichiers_attendus"]]
    hors_sujet = [q for q in questions if not q["fichiers_attendus"]]
    print(f"Questions : {len(dans_le_sujet)} dans le sujet, {len(hors_sujet)} hors sujet")

    collection = ouvrir_collection()
    modele = charger_modele()

    # On évalue les deux méthodes sur les mêmes questions
    lignes = evaluer("Recherche simple", rechercher_simple, collection, modele, dans_le_sujet)
    # lambda = petite fonction sans nom : elle adapte rechercher() aux 5 paramètres
    # attendus par evaluer() (c = collection, m = modèle, q = question, k, p = public)
    lignes += evaluer("Recherche améliorée",
                      lambda c, m, q, k, p: rechercher(c, m, q, k=k, public=p),
                      collection, modele, dans_le_sujet)
    scores_hors_sujet(collection, modele, hors_sujet)

    # Enregistrement du détail, question par question, dans resultats_recherche.csv
    # newline="" : évite les lignes vides en trop sous Windows ; delimiter=";" : pour Excel en français
    with open(FICHIER_RESULTATS, "w", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=lignes[0].keys(), delimiter=";")   # colonnes = clés
        ecrivain.writeheader()          # 1re ligne : les noms des colonnes
        ecrivain.writerows(lignes)      # puis une ligne par question et par méthode
    # relative_to : affiche le chemin à partir du dossier du projet (plus court)
    print(f"\nDétail enregistré dans : {FICHIER_RESULTATS.relative_to(DOSSIER_PROJET)}")