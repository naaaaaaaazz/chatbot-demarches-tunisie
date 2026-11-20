"""
Étape 8a : évaluation automatique de la RECHERCHE (sans le LLM).

Pour chaque question de evaluation/questions_test.csv, on regarde à quelle
position arrive le premier bon document, puis on calcule :
- rappel@3 : le bon document est-il dans les 3 premiers résultats ?
- rappel@5 : le bon document est-il dans les 5 premiers résultats ?
- MRR      : moyenne de 1 / position du premier bon document.
On compare la recherche SIMPLE (étape 5) et la recherche AMÉLIORÉE.
"""
import csv
from pathlib import Path

from embeddings import charger_modele, vectoriser_question
from generation import ouvrir_collection
from indexation import rechercher

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_QUESTIONS = DOSSIER_PROJET / "evaluation" / "questions_test.csv"
FICHIER_RESULTATS = DOSSIER_PROJET / "evaluation" / "resultats_recherche.csv"
K_MAX = 5


def charger_questions():
    """Lit le fichier des questions de test."""
    with open(FICHIER_QUESTIONS, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def rechercher_simple(collection, modele, question, k, public):
    """Recherche de l'étape 5 : les k plus proches, sans aucune amélioration."""
    vecteur = vectoriser_question(modele, question)
    filtre = {"public": {"$in": [public, "tous"]}} if public else None
    res = collection.query(query_embeddings=[vecteur.tolist()], n_results=k, where=filtre)
    return [{"source": meta["source"], "score": 1 - distance}
            for meta, distance in zip(res["metadatas"][0], res["distances"][0])]


def position_bon_document(resultats, attendus):
    """Position (1, 2, 3...) du premier bon document, ou None s'il n'est pas trouvé."""
    for position, r in enumerate(resultats, start=1):
        if Path(r["source"]).stem in attendus:
            return position
    return None


def evaluer(nom, fonction, collection, modele, questions):
    """Calcule rappel@3, rappel@5 et MRR pour une méthode de recherche."""
    lignes = []
    for q in questions:
        attendus = q["fichiers_attendus"].split(";")
        public = q["public"] or None
        resultats = fonction(collection, modele, q["question"], K_MAX, public)
        position = position_bon_document(resultats, attendus)
        lignes.append({"methode": nom, "id": q["id"], "question": q["question"],
                       "position": position if position else "non trouvé",
                       "rappel@3": int(position is not None and position <= 3),
                       "rappel@5": int(position is not None and position <= 5),
                       "rr": 1 / position if position else 0,
                       "premier_resultat": resultats[0]["source"]})
    n = len(lignes)
    print(f"\n===== {nom} ({n} questions) =====")
    print(f"Rappel@3 : {sum(l['rappel@3'] for l in lignes) / n:.2f}")
    print(f"Rappel@5 : {sum(l['rappel@5'] for l in lignes) / n:.2f}")
    print(f"MRR      : {sum(l['rr'] for l in lignes) / n:.2f}")
    for l in lignes:
        if not l["rappel@3"]:
            print(f"  ⚠ Q{l['id']} (position : {l['position']}) {l['question']}")
    return lignes


def scores_hors_sujet(collection, modele, questions):
    """Affiche le meilleur score obtenu par les questions hors sujet."""
    print("\n===== Questions hors sujet : meilleur score =====")
    for q in questions:
        resultats = rechercher(collection, modele, q["question"], k=1)
        print(f"  {resultats[0]['score']:.3f}  {q['question']}")


if __name__ == "__main__":
    questions = charger_questions()
    dans_le_sujet = [q for q in questions if q["fichiers_attendus"]]
    hors_sujet = [q for q in questions if not q["fichiers_attendus"]]
    print(f"Questions : {len(dans_le_sujet)} dans le sujet, {len(hors_sujet)} hors sujet")

    collection = ouvrir_collection()
    modele = charger_modele()

    lignes = evaluer("Recherche simple", rechercher_simple, collection, modele, dans_le_sujet)
    lignes += evaluer("Recherche améliorée",
                      lambda c, m, q, k, p: rechercher(c, m, q, k=k, public=p),
                      collection, modele, dans_le_sujet)
    scores_hors_sujet(collection, modele, hors_sujet)

    with open(FICHIER_RESULTATS, "w", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=lignes[0].keys(), delimiter=";")
        ecrivain.writeheader()
        ecrivain.writerows(lignes)
    print(f"\nDétail enregistré dans : {FICHIER_RESULTATS.relative_to(DOSSIER_PROJET)}")