"""
Étape 5 : stockage des chunks et de leurs vecteurs dans ChromaDB,
puis recherche avec filtre sur les métadonnées.

Améliorations de la recherche :
- expansion de requête : les abréviations (ex. « cin ») sont remplacées avant la recherche ;
- au plus 2 extraits par document (le guide ne prend plus toutes les places) ;
- petite priorité pour les fiches de procédure.
"""
import json
import re
from pathlib import Path

import numpy as np
import chromadb

from embeddings import charger_modele, vectoriser_question, FICHIER_CHUNKS, FICHIER_VECTEURS

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_BASE = DOSSIER_PROJET / "vector_db"
NOM_COLLECTION = "chunks_600"

MAX_PAR_DOCUMENT = 2
BONUS_PROCEDURE = 0.01

ABREVIATIONS = {
    r"\bcin\b": "carte d'identité nationale (CIN)",
    r"\bpasseport\b|\bpasport\b|\bpassport\b": "passeport",
}


def charger_donnees():
    """Recharge les chunks et les vecteurs calculés à l'étape 4."""
    with open(FICHIER_CHUNKS, encoding="utf-8") as f:
        chunks = json.load(f)
    vecteurs = np.load(FICHIER_VECTEURS)
    return chunks, vecteurs


def creer_collection(nom=NOM_COLLECTION):
    """Crée une collection vide (supprime l'ancienne si elle existe)."""
    client = chromadb.PersistentClient(path=str(DOSSIER_BASE))
    try:
        client.delete_collection(nom)
    except Exception:
        pass  # la collection n'existait pas encore
    return client.create_collection(name=nom, metadata={"hnsw:space": "cosine"})


def indexer(collection, chunks, vecteurs, taille_lot=100):
    """Ajoute les chunks dans la collection, par lots de 100."""
    for debut in range(0, len(chunks), taille_lot):
        fin = debut + taille_lot
        lot = chunks[debut:fin]
        collection.add(
            ids=[c["id"] for c in lot],
            embeddings=vecteurs[debut:fin].tolist(),
            documents=[c["texte"] for c in lot],
            metadatas=[{"source": c["source"], "chemin": c["chemin"],
                        "page": c["page"], "type": c["type"],
                        "public": c["public"]} for c in lot],
        )


def enrichir_question(question):
    """Remplace les abréviations et fautes courantes avant la recherche."""
    for motif, remplacement in ABREVIATIONS.items():
        question = re.sub(motif, remplacement, question, flags=re.IGNORECASE)
    return question

MOTS_ETRANGER = ["étranger", "etranger", "consulat", "ambassade", "france", "italie",
                 "allemagne", "belgique", "canada", "europe", "j'habite à l'", "je vis à l'"]
MOTS_TUNISIE = ["en tunisie"]


def detecter_public(question):
    """Devine la zone à partir des mots de la question (None si rien n'est détecté)."""
    texte = question.lower()
    if any(mot in texte for mot in MOTS_ETRANGER):
        return "etranger"
    if any(mot in texte for mot in MOTS_TUNISIE):
        return "tunisie"
    return None


def rechercher(collection, modele, question, k=3, public=None):
    """Renvoie les k meilleurs chunks, avec au plus 2 chunks par document
    et une petite priorité pour les fiches de procédure."""
    if public is None:
        public = detecter_public(question)
    vecteur = vectoriser_question(modele, enrichir_question(question))
    filtre = {"public": {"$in": [public, "tous"]}} if public else None
    res = collection.query(query_embeddings=[vecteur.tolist()],
                           n_results=k * 4, where=filtre)
    
    candidats = []
    for texte, meta, distance in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        score = 1 - distance
        bonus = BONUS_PROCEDURE if meta["type"] == "procedure" else 0
        candidats.append({"texte": texte, "source": meta["source"], "page": meta["page"],
                          "type": meta["type"], "score": score, "priorite": score + bonus})
    candidats.sort(key=lambda c: c["priorite"], reverse=True)

    resultats = []
    compte = {}
    for c in candidats:
        if compte.get(c["source"], 0) < MAX_PAR_DOCUMENT:
            resultats.append(c)
            compte[c["source"]] = compte.get(c["source"], 0) + 1
        if len(resultats) == k:
            break
    return resultats


if __name__ == "__main__":
    chunks, vecteurs = charger_donnees()
    collection = creer_collection()
    indexer(collection, chunks, vecteurs)

    print("\n========== BILAN DE L'INDEXATION ==========")
    print(f"Chunks dans la base : {collection.count()}")
    print(f"Base enregistrée dans : {DOSSIER_BASE.relative_to(DOSSIER_PROJET)}")

    modele = charger_modele()
    tests = [("J'ai perdu ma carte d'identité, que dois-je faire ?", None),
             ("J'ai perdu mon passeport, que dois-je faire ?", "etranger"),
             ("je veux renouvller ma cin je fais quoi", None)]
    for question, public in tests:
        print(f"\nQuestion : « {question} »  (filtre public = {public})")
        print(f"Question enrichie : « {enrichir_question(question)} »")
        for r in rechercher(collection, modele, question, k=5, public=public):
            print(f"  {r['score']:.3f}  {r['source']} (page {r['page']})")