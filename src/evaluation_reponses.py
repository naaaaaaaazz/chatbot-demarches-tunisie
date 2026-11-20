"""
Étape 8c : générer les réponses du chatbot pour les 35 questions de test,
afin de les noter À LA MAIN (exactitude et fidélité).

Les réponses sont enregistrées une par une dans evaluation/reponses_llm.csv.
Si le programme s'arrête, on le relance : les questions déjà faites sont sautées.
"""
import csv
import time
from pathlib import Path

from embeddings import charger_modele
from generation import ouvrir_collection, repondre
from choix_formulaire import charger_correspondances, trouver_formulaires
from evaluation_recherche import charger_questions

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_REPONSES = DOSSIER_PROJET / "evaluation" / "reponses_llm.csv"
COLONNES = ["id", "question", "reponse_attendue", "reponse_chatbot", "sources",
            "formulaires", "temps_s", "exactitude", "fidelite", "commentaire"]


def ids_deja_faits():
    """Renvoie les numéros des questions déjà présentes dans le fichier."""
    if not FICHIER_REPONSES.exists():
        return set()
    with open(FICHIER_REPONSES, encoding="utf-8-sig") as f:
        return {ligne["id"] for ligne in csv.DictReader(f, delimiter=";")}


if __name__ == "__main__":
    questions = charger_questions()
    faits = ids_deja_faits()
    a_faire = [q for q in questions if q["id"] not in faits]
    print(f"{len(faits)} question(s) déjà faite(s), {len(a_faire)} à faire.")

    collection = ouvrir_collection()
    modele = charger_modele()
    correspondances = charger_correspondances()

    nouveau_fichier = not FICHIER_REPONSES.exists()
    with open(FICHIER_REPONSES, "a", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=COLONNES, delimiter=";")
        if nouveau_fichier:
            ecrivain.writeheader()

        for numero, q in enumerate(a_faire, start=1):
            print(f"\n[{numero}/{len(a_faire)}] Q{q['id']} : {q['question']}")
            debut = time.perf_counter()
            reponse, resultats = repondre(q["question"], collection, modele,
                                          public=q["public"] or None)
            formulaires = trouver_formulaires(reponse, resultats, correspondances)
            duree = time.perf_counter() - debut

            ecrivain.writerow({
                "id": q["id"],
                "question": q["question"],
                "reponse_attendue": q["reponse_attendue"],
                "reponse_chatbot": reponse,
                "sources": " | ".join(f"{r['source']} p.{r['page']}" for r in resultats),
                "formulaires": " | ".join(fichier for fichier, _ in formulaires),
                "temps_s": round(duree, 1),
                "exactitude": "", "fidelite": "", "commentaire": "",
            })
            f.flush()  # écrit tout de suite sur le disque (rien n'est perdu si ça s'arrête)
            print(f"   -> réponse enregistrée ({duree:.0f} s)")

    print(f"\nTerminé. Fichier : {FICHIER_REPONSES.relative_to(DOSSIER_PROJET)}")