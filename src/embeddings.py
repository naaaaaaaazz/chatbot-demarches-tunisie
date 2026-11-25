"""
ÉTAPE 4 — LES EMBEDDINGS

Transforme chaque chunk en vecteur (une liste de 768 nombres qui représente son sens)
avec le modèle multilingue intfloat/multilingual-e5-base, puis fait un petit test de
recherche sémantique.
Résultat : data/processed/embeddings.npy

Lancer :  python src/embeddings.py
"""
from pathlib import Path    # pour manipuler les chemins de fichiers

import numpy as np          # calcul sur des tableaux de nombres (ici : les vecteurs)
from sentence_transformers import SentenceTransformer   # bibliothèque qui charge et utilise le modèle e5

# On réutilise les étapes 1, 2 et 3
from extraction import extraire_tout, sauvegarder_json
from nettoyage import nettoyer_tout
from decoupage import decouper


# ==================================================
# 1. CHEMINS ET MODÈLE
# ==================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_VECTEURS = DOSSIER_PROJET / "data" / "processed" / "embeddings.npy"   # .npy = format de NumPy
FICHIER_CHUNKS = DOSSIER_PROJET / "data" / "processed" / "chunks.json"

# Modèle multilingue (bon en français), gratuit, qui tourne sur l'ordinateur.
# Il transforme un texte en 768 nombres : deux textes de sens proche ont des vecteurs proches.
NOM_MODELE = "intfloat/multilingual-e5-base"   # modèle recommandé par le cours


# ==================================================
# 2. CHARGER LE MODÈLE
# ==================================================

def charger_modele():
    """Charge le modèle d'embeddings (téléchargé automatiquement la première fois)."""
    # La 1re fois, le modèle (≈ 1 Go) est téléchargé depuis Hugging Face, puis gardé en cache.
    print(f"Chargement du modèle {NOM_MODELE}...")
    return SentenceTransformer(NOM_MODELE)


# ==================================================
# 3. VECTORISER LES CHUNKS ET LES QUESTIONS
# ==================================================

def vectoriser_chunks(modele, chunks):
    """Transforme chaque chunk en vecteur. Le modèle e5 exige le préfixe « passage: »."""
    # e5 a été entraîné avec ces préfixes : « passage: » pour les documents,
    # « query: » pour les questions. Sans eux, la recherche est moins bonne.
    textes = ["passage: " + chunk["texte"] for chunk in chunks]
    # encode() renvoie un tableau de 232 lignes x 768 colonnes (un vecteur par chunk).
    # normalize_embeddings=True met chaque vecteur à la longueur 1 (voir plus bas pourquoi).
    # show_progress_bar=True affiche la barre de progression (le calcul prend environ 3 min 30).
    return modele.encode(textes, normalize_embeddings=True, show_progress_bar=True)


def vectoriser_question(modele, question):
    """Transforme une question en vecteur. Le modèle e5 exige le préfixe « query: »."""
    # On encode une liste d'une seule question, puis [0] prend son vecteur (768 nombres).
    return modele.encode(["query: " + question], normalize_embeddings=True)[0]


# ==================================================
# 4. PETIT TEST DE RECHERCHE SÉMANTIQUE
# ==================================================

# Cette fonction sert seulement à tester les embeddings ici.
# La vraie recherche du chatbot se fait avec ChromaDB (fonction rechercher() dans indexation.py).
def chercher(question, modele, vecteurs, chunks, k=3):
    """Renvoie les k chunks dont le sens est le plus proche de la question."""
    vecteur_question = vectoriser_question(modele, question)
    # « @ » = produit matriciel : on multiplie le vecteur de la question par celui de chaque chunk.
    # Comme les vecteurs ont une longueur 1, ce produit scalaire EST la similarité cosinus
    # (entre -1 et 1 ; plus c'est proche de 1, plus les sens sont proches). On obtient 232 scores.
    scores = vecteurs @ vecteur_question            # similarité cosinus avec chaque chunk
    # argsort trie les indices du plus petit au plus grand score,
    # [::-1] inverse l'ordre (du plus grand au plus petit), [:k] garde les k premiers.
    meilleurs = np.argsort(scores)[::-1][:k]        # indices des k meilleurs scores
    # float(...) transforme le nombre NumPy en nombre Python normal (pour l'affichage)
    return [(chunks[i], float(scores[i])) for i in meilleurs]


# ==================================================
# 5. PROGRAMME PRINCIPAL
# ==================================================

# S'exécute seulement quand on lance « python src/embeddings.py »
if __name__ == "__main__":
    chunks = decouper(nettoyer_tout(extraire_tout()))   # étapes 1, 2 et 3
    modele = charger_modele()

    vecteurs = vectoriser_chunks(modele, chunks)        # étape 4
    # .shape = dimensions du tableau : (nombre de chunks, nombre de nombres par vecteur)
    print("\n========== BILAN DES EMBEDDINGS ==========")
    print(f"Chunks vectorisés      : {vecteurs.shape[0]}")          # 232
    print(f"Taille d'un vecteur    : {vecteurs.shape[1]} nombres")  # 768

    # On enregistre les vecteurs (embeddings.npy) et les chunks (chunks.json)
    FICHIER_VECTEURS.parent.mkdir(parents=True, exist_ok=True)
    np.save(FICHIER_VECTEURS, vecteurs)
    sauvegarder_json(chunks, FICHIER_CHUNKS)
    print(f"Vecteurs enregistrés dans : {FICHIER_VECTEURS.relative_to(DOSSIER_PROJET)}")

    # Petit test : les 3 chunks les plus proches d'une vraie question
    question = "J'ai perdu ma carte d'identité, que dois-je faire ?"
    print(f"\nTest de recherche : « {question} »")
    for chunk, score in chercher(question, modele, vecteurs, chunks):
        print(f"  {score:.3f}  {chunk['source']} (page {chunk['page']})")   # :.3f = 3 chiffres après la virgule