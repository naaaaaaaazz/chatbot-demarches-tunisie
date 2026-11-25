"""
Étape 8c : générer les réponses du chatbot pour les 35 questions de test,
afin de les noter À LA MAIN (exactitude et fidélité).

Les réponses sont enregistrées une par une dans evaluation/reponses_llm.csv.
Si le programme s'arrête, on le relance : les questions déjà faites sont sautées.
"""
import csv                  # lire les questions, écrire les réponses en CSV
import time                 # chronomètre (perf_counter) et pause (sleep)
from pathlib import Path    # manipuler les chemins de fichiers

# On réutilise toute la chaîne : modèle e5, RAG complet, choix du formulaire, questions de test
from embeddings import charger_modele
from generation import ouvrir_collection, repondre
from choix_formulaire import charger_correspondances, trouver_formulaires
from evaluation_recherche import charger_questions

DOSSIER_PROJET = Path(__file__).resolve().parent.parent               # src/ -> dossier du projet
FICHIER_REPONSES = DOSSIER_PROJET / "evaluation" / "reponses_llm.csv"   # le fichier produit
# Les colonnes du CSV. Les 3 dernières sont laissées vides : on les remplit à la main dans Excel
# (exactitude et fidélité notées de 0 à 2, plus un commentaire si besoin)
COLONNES = ["id", "question", "reponse_attendue", "reponse_chatbot", "sources",
            "formulaires", "temps_s", "exactitude", "fidelite", "commentaire"]
PAUSE_S = 25   # pause entre deux questions, pour rester dans la limite gratuite de Groq


def ids_deja_faits():
    """Renvoie les numéros des questions déjà présentes dans le fichier."""
    if not FICHIER_REPONSES.exists():
        return set()                        # aucun fichier : rien n'est encore fait
    with open(FICHIER_REPONSES, encoding="utf-8-sig") as f:
        # ensemble en compréhension : { "1", "2", ... } (un set = pas de doublons, recherche rapide)
        return {ligne["id"] for ligne in csv.DictReader(f, delimiter=";")}


# S'exécute seulement avec « python src/evaluation_reponses.py »
if __name__ == "__main__":
    questions = charger_questions()
    faits = ids_deja_faits()
    # On garde seulement les questions pas encore traitées (reprise après un arrêt)
    a_faire = [q for q in questions if q["id"] not in faits]
    print(f"{len(faits)} question(s) déjà faite(s), {len(a_faire)} à faire.")

    collection = ouvrir_collection()
    modele = charger_modele()
    correspondances = charger_correspondances()

    nouveau_fichier = not FICHIER_REPONSES.exists()
    # "a" = append : on AJOUTE à la fin du fichier sans effacer les réponses déjà enregistrées
    # newline="" : évite les lignes vides en trop sous Windows
    with open(FICHIER_REPONSES, "a", encoding="utf-8-sig", newline="") as f:
        ecrivain = csv.DictWriter(f, fieldnames=COLONNES, delimiter=";")   # « ; » pour Excel en français
        if nouveau_fichier:
            ecrivain.writeheader()          # la ligne des noms de colonnes, seulement la 1re fois

        for numero, q in enumerate(a_faire, start=1):
            print(f"\n[{numero}/{len(a_faire)}] Q{q['id']} : {q['question']}")
            debut = time.perf_counter()     # on lance le chronomètre
            # Toute la chaîne RAG, comme dans l'application ; case « public » vide -> None
            reponse, resultats = repondre(q["question"], collection, modele,
                                          public=q["public"] or None)
            formulaires = trouver_formulaires(reponse, resultats, correspondances)
            duree = time.perf_counter() - debut     # temps total en secondes

            # Une ligne du CSV pour cette question
            ecrivain.writerow({
                "id": q["id"],
                "question": q["question"],
                "reponse_attendue": q["reponse_attendue"],      # écrite à l'avance, pour comparer
                "reponse_chatbot": reponse,
                # les 5 extraits trouvés, séparés par « | » : « fichier.pdf p.1 | ... »
                "sources": " | ".join(f"{r['source']} p.{r['page']}" for r in resultats),
                # chaque formulaire est un couple (fichier, nom) : on garde le fichier (« _ » ignore le nom)
                "formulaires": " | ".join(fichier for fichier, _ in formulaires),
                "temps_s": str(round(duree, 1)).replace(".", ","),   # virgule : lisible par Excel en français
                "exactitude": "", "fidelite": "", "commentaire": "",   # à remplir à la main
            })

            f.flush()  # écrit tout de suite sur le disque (rien n'est perdu si ça s'arrête)
            print(f"   -> réponse enregistrée ({duree:.0f} s)")
            # Pause entre deux questions (pas après la dernière) : l'API gratuite de Groq
            # limite le nombre de tokens par minute (sinon erreur 429)
            if numero < len(a_faire):
                time.sleep(PAUSE_S)

    print(f"\nTerminé. Fichier : {FICHIER_REPONSES.relative_to(DOSSIER_PROJET)}")