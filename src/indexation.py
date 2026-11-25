"""
Étape 5 : stockage des chunks et de leurs vecteurs dans ChromaDB,
puis recherche avec filtre sur les métadonnées.

Améliorations de la recherche :
- expansion de requête : les abréviations (ex. « cin ») sont remplacées avant la recherche ;
- au plus 2 extraits par document (le guide ne prend plus toutes les places) ;
- petite priorité pour les fiches de procédure.
(+ détection de la zone Tunisie / étranger, plus bas : detecter_public)
"""
import json                 # pour relire chunks.json
import re                   # expressions régulières (remplacer les abréviations)
from pathlib import Path    # pour manipuler les chemins de fichiers

import numpy as np          # pour relire les vecteurs enregistrés (embeddings.npy)
import chromadb             # la base vectorielle

# On réutilise l'étape 4 : le modèle e5, la fonction qui transforme une question en vecteur,
# et les chemins des fichiers créés à l'étape 4
from embeddings import charger_modele, vectoriser_question, FICHIER_CHUNKS, FICHIER_VECTEURS

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_BASE = DOSSIER_PROJET / "vector_db"     # dossier où ChromaDB enregistre la base sur le disque
NOM_COLLECTION = "chunks_600"                   # une « collection » = une table de la base (chunks de 600 car.)

MAX_PAR_DOCUMENT = 2        # au plus 2 extraits d'un même PDF dans les résultats
BONUS_PROCEDURE = 0.01      # petit bonus pour faire passer les fiches pratiques devant les lois et le guide

# Expansion de requête : motif à chercher -> texte qui le remplace.
# \b = limite de mot : « cin » est remplacé, mais pas « cinéma ». | = « ou ».
ABREVIATIONS = {
    r"\bcin\b": "carte d'identité nationale (CIN)",
    r"\bpasseport\b|\bpasport\b|\bpassport\b": "passeport",   # corrige les fautes d'orthographe courantes
}


def charger_donnees():
    """Recharge les chunks et les vecteurs calculés à l'étape 4."""
    # On ne recalcule pas les embeddings (3 min 30) : on relit ceux enregistrés à l'étape 4
    with open(FICHIER_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)                     # liste des 232 chunks (texte + métadonnées)
    vecteurs = np.load(FICHIER_VECTEURS)          # tableau 232 x 768
    return chunks, vecteurs


def creer_collection(nom=NOM_COLLECTION):
    """Crée une collection vide (supprime l'ancienne si elle existe)."""
    # PersistentClient = base enregistrée sur le disque (dans vector_db/) : elle reste après l'arrêt du programme
    client = chromadb.PersistentClient(path=str(DOSSIER_BASE))
    try:
        client.delete_collection(nom)             # on repart de zéro pour ne pas avoir de doublons
    except Exception:
        pass  # la collection n'existait pas encore
    # ChromaDB range les vecteurs dans un index HNSW (un graphe de voisins) qui retrouve
    # les plus proches très vite ; "hnsw:space": "cosine" = on mesure la distance cosinus
    return client.create_collection(name=nom, metadata={"hnsw:space": "cosine"})


def indexer(collection, chunks, vecteurs, taille_lot=100):
    """Ajoute les chunks dans la collection, par lots de 100."""
    # range(0, 232, 100) donne 0, 100, 200 : trois lots (0-99, 100-199, 200-231)
    for debut in range(0, len(chunks), taille_lot):
        fin = debut + taille_lot
        lot = chunks[debut:fin]                   # [debut:fin] = une tranche de la liste
        collection.add(
            ids=[c["id"] for c in lot],                       # identifiant unique de chaque chunk
            embeddings=vecteurs[debut:fin].tolist(),          # les vecteurs (tolist : tableau NumPy -> liste Python)
            documents=[c["texte"] for c in lot],              # le texte, renvoyé avec les résultats
            metadatas=[{"source": c["source"], "chemin": c["chemin"],   # métadonnées : pour citer
                        "page": c["page"], "type": c["type"],           # et pour filtrer
                        "public": c["public"]} for c in lot],
        )


def enrichir_question(question):
    """Remplace les abréviations et fautes courantes avant la recherche."""
    # Exemple : « je veux renouvller ma cin » -> « ... ma carte d'identité nationale (CIN) »
    # (le modèle e5 comprend mal « cin » seul et trouvait la fiche passeport)
    for motif, remplacement in ABREVIATIONS.items():
        question = re.sub(motif, remplacement, question, flags=re.IGNORECASE)
    return question

# Mots qui indiquent que la personne est à l'étranger, ou en Tunisie
MOTS_ETRANGER = ["étranger", "etranger", "consulat", "ambassade", "france", "italie",
                 "allemagne", "belgique", "canada", "europe", "j'habite à l'", "je vis à l'"]
MOTS_TUNISIE = ["en tunisie"]


def detecter_public(question):
    """Devine la zone à partir des mots de la question (None si rien n'est détecté)."""
    texte = question.lower()                      # en minuscules pour comparer facilement
    if any(mot in texte for mot in MOTS_ETRANGER):
        return "etranger"
    if any(mot in texte for mot in MOTS_TUNISIE):
        return "tunisie"
    return None                                   # rien de détecté : pas de filtre


def rechercher(collection, modele, question, k=3, public=None):
    """Renvoie les k meilleurs chunks, avec au plus 2 chunks par document
    et une petite priorité pour les fiches de procédure."""
    # 1) Zone : si l'utilisateur n'a rien choisi dans la barre latérale, on la devine
    if public is None:
        public = detecter_public(question)
    # 2) On remplace les abréviations, puis on transforme la question en vecteur (préfixe « query: »)
    vecteur = vectoriser_question(modele, enrichir_question(question))
    # 3) Filtre sur les métadonnées : ex. public = etranger -> on cherche dans « etranger » ET « tous ».
    #    $in = « la valeur fait partie de cette liste ». Pas de zone -> pas de filtre (None).
    filtre = {"public": {"$in": [public, "tous"]}} if public else None
    # 4) On demande à ChromaDB 4 fois plus de candidats que nécessaire (20 pour k = 5),
    #    car on va ensuite en écarter (règle des 2 par document)
    res = collection.query(query_embeddings=[vecteur.tolist()],
                           n_results=k * 4, where=filtre)

    # 5) On transforme la réponse de ChromaDB en liste de dictionnaires.
    #    [0] = résultats de notre seule question ; zip parcourt les trois listes en même temps.
    candidats = []
    for texte, meta, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - distance                      # ChromaDB donne une distance : 1 - distance = similarité
        bonus = BONUS_PROCEDURE if meta["type"] == "procedure" else 0
        candidats.append({"texte": texte, "source": meta["source"], "page": meta["page"],
                          "type": meta["type"], "score": score, "priorite": score + bonus})
    # 6) Tri par priorité (score + bonus), du plus grand au plus petit.
    #    Le bonus sert seulement au tri : le score affiché reste le vrai score.
    candidats.sort(key=lambda c: c["priorite"], reverse=True)

    # 7) On garde les meilleurs, mais au plus 2 chunks par document
    resultats = []
    compte = {}                                   # compte = {nom_du_pdf: nombre de chunks déjà gardés}
    for c in candidats:
        if compte.get(c["source"], 0) < MAX_PAR_DOCUMENT:   # .get(..., 0) = 0 si le PDF n'est pas encore compté
            resultats.append(c)
            compte[c["source"]] = compte.get(c["source"], 0) + 1
        if len(resultats) == k:                   # on s'arrête dès qu'on en a k
            break
    return resultats


# S'exécute seulement quand on lance « python src/indexation.py »
if __name__ == "__main__":
    chunks, vecteurs = charger_donnees()          # relire les résultats de l'étape 4
    collection = creer_collection()               # créer une collection vide
    indexer(collection, chunks, vecteurs)         # y ranger les 232 chunks

    print("\n========== BILAN DE L'INDEXATION ==========")
    print(f"Chunks dans la base : {collection.count()}")   # doit afficher 232
    print(f"Base enregistrée dans : {DOSSIER_BASE.relative_to(DOSSIER_PROJET)}")

    # Petit test de la recherche sur 3 questions (avec k = 5, comme dans le chatbot)
    modele = charger_modele()
    tests = [("J'ai perdu ma carte d'identité, que dois-je faire ?", None),
             ("J'ai perdu mon passeport, que dois-je faire ?", "etranger"),
             ("je veux renouvller ma cin je fais quoi", None)]
    for question, public in tests:
        print(f"\nQuestion : « {question} »  (filtre public = {public})")
        print(f"Question enrichie : « {enrichir_question(question)} »")
        for r in rechercher(collection, modele, question, k=5, public=public):
            print(f"  {r['score']:.3f}  {r['source']} (page {r['page']})")