"""
Étape 8b : comparer deux tailles de chunks (300 et 600 caractères).

Pour chaque taille, on crée (une seule fois) une collection ChromaDB :
chunks_300 et chunks_600. Puis on mesure la recherche améliorée avec
evaluer() de l'étape 8a : rappel@3 (k=3), rappel@5 (k=5) et MRR.
"""
import chromadb             # la base vectorielle

# On réutilise TOUTE la chaîne des étapes 1 à 5, puis l'évaluation de l'étape 8a
from extraction import extraire_tout                 # 1. PDF -> texte par page
from nettoyage import nettoyer_tout                  # 2. nettoyage du texte
from decoupage import decouper                       # 3. découpage en chunks
from embeddings import charger_modele, vectoriser_chunks        # 4. vecteurs e5
from indexation import DOSSIER_BASE, creer_collection, indexer, rechercher   # 5. ChromaDB
from evaluation_recherche import charger_questions, evaluer     # 8a. rappel@3, rappel@5, MRR

# Les deux réglages à comparer : (taille du chunk en caractères, chevauchement)
# Le chevauchement est d'environ 15 % de la taille dans les deux cas
TAILLES = [(300, 50), (600, 80)]  # (taille du chunk, chevauchement)


def collection_pour(taille, chevauchement, modele):
    """Renvoie la collection chunks_<taille> ; la crée si elle n'existe pas encore."""
    nom = f"chunks_{taille}"                  # « chunks_300 » ou « chunks_600 »
    client = chromadb.PersistentClient(path=str(DOSSIER_BASE))   # la base dans vector_db/
    # On essaie d'ouvrir la collection : si elle existe déjà et n'est pas vide,
    # on la réutilise (inutile de recalculer les vecteurs, ce qui prend plusieurs minutes)
    try:
        collection = client.get_collection(nom)
        if collection.count() > 0:            # count() = nombre de chunks dans la collection
            print(f"{nom} : déjà indexée ({collection.count()} chunks)")
            return collection
    except Exception:
        pass  # la collection n'existe pas encore : on la crée

    # Sinon : on refait toute la chaîne avec cette taille de chunk
    print(f"\nCréation de {nom} (taille {taille}, chevauchement {chevauchement})...")
    # Les appels sont imbriqués : extraire, puis nettoyer, puis découper (lire de l'intérieur vers l'extérieur)
    chunks = decouper(nettoyer_tout(extraire_tout()), taille, chevauchement)
    vecteurs = vectoriser_chunks(modele, chunks)          # l'étape la plus longue
    collection = creer_collection(nom)
    indexer(collection, chunks, vecteurs)                 # range chunks + vecteurs + métadonnées
    print(f"{nom} : {collection.count()} chunks indexés")
    return collection


# S'exécute seulement avec « python src/comparer_chunks.py »
if __name__ == "__main__":
    # Seulement les questions dans le sujet (celles qui ont un fichier attendu)
    questions = [q for q in charger_questions() if q["fichiers_attendus"]]
    modele = charger_modele()

    # Même modèle, mêmes questions, même recherche améliorée : SEULE la taille des chunks change.
    # C'est ce qui rend la comparaison juste.
    for taille, chevauchement in TAILLES:
        collection = collection_pour(taille, chevauchement, modele)
        # lambda : adapte rechercher() aux 5 paramètres attendus par evaluer()
        evaluer(f"Chunks de {taille} caractères",
                lambda c, m, q, k, p: rechercher(c, m, q, k=k, public=p),
                collection, modele, questions)