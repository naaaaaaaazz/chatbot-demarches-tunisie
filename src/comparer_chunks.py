"""
Étape 8b : comparer deux tailles de chunks (300 et 600 caractères).

Pour chaque taille, on crée (une seule fois) une collection ChromaDB :
chunks_300 et chunks_600. Puis on mesure la recherche améliorée avec
evaluer() de l'étape 8a : rappel@3 (k=3), rappel@5 (k=5) et MRR.
"""
import chromadb

from extraction import extraire_tout
from nettoyage import nettoyer_tout
from decoupage import decouper
from embeddings import charger_modele, vectoriser_chunks
from indexation import DOSSIER_BASE, creer_collection, indexer, rechercher
from evaluation_recherche import charger_questions, evaluer

TAILLES = [(300, 50), (600, 80)]  # (taille du chunk, chevauchement)


def collection_pour(taille, chevauchement, modele):
    """Renvoie la collection chunks_<taille> ; la crée si elle n'existe pas encore."""
    nom = f"chunks_{taille}"
    client = chromadb.PersistentClient(path=str(DOSSIER_BASE))
    try:
        collection = client.get_collection(nom)
        if collection.count() > 0:
            print(f"{nom} : déjà indexée ({collection.count()} chunks)")
            return collection
    except Exception:
        pass  # la collection n'existe pas encore : on la crée

    print(f"\nCréation de {nom} (taille {taille}, chevauchement {chevauchement})...")
    chunks = decouper(nettoyer_tout(extraire_tout()), taille, chevauchement)
    vecteurs = vectoriser_chunks(modele, chunks)
    collection = creer_collection(nom)
    indexer(collection, chunks, vecteurs)
    print(f"{nom} : {collection.count()} chunks indexés")
    return collection


if __name__ == "__main__":
    questions = [q for q in charger_questions() if q["fichiers_attendus"]]
    modele = charger_modele()

    for taille, chevauchement in TAILLES:
        collection = collection_pour(taille, chevauchement, modele)
        evaluer(f"Chunks de {taille} caractères",
                lambda c, m, q, k, p: rechercher(c, m, q, k=k, public=p),
                collection, modele, questions)