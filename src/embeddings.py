"""
ÉTAPE 4 — LES EMBEDDINGS

Transforme chaque chunk en vecteur (une liste de 768 nombres qui représente son sens)
avec le modèle multilingue intfloat/multilingual-e5-base, puis fait un petit test de
recherche sémantique.
Résultat : data/processed/embeddings.npy

Lancer :  python src/embeddings.py
"""
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from extraction import extraire_tout, sauvegarder_json
from nettoyage import nettoyer_tout
from decoupage import decouper


# ==================================================
# 1. CHEMINS ET MODÈLE
# ==================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_VECTEURS = DOSSIER_PROJET / "data" / "processed" / "embeddings.npy"
FICHIER_CHUNKS = DOSSIER_PROJET / "data" / "processed" / "chunks.json"

NOM_MODELE = "intfloat/multilingual-e5-base"   # modèle recommandé par le cours


# ==================================================
# 2. CHARGER LE MODÈLE
# ==================================================

def charger_modele():
    """Charge le modèle d'embeddings (téléchargé automatiquement la première fois)."""
    print(f"Chargement du modèle {NOM_MODELE}...")
    return SentenceTransformer(NOM_MODELE)


# ==================================================
# 3. VECTORISER LES CHUNKS ET LES QUESTIONS
# ==================================================

def vectoriser_chunks(modele, chunks):
    """Transforme chaque chunk en vecteur. Le modèle e5 exige le préfixe « passage: »."""
    textes = ["passage: " + chunk["texte"] for chunk in chunks]
    return modele.encode(textes, normalize_embeddings=True, show_progress_bar=True)


def vectoriser_question(modele, question):
    """Transforme une question en vecteur. Le modèle e5 exige le préfixe « query: »."""
    return modele.encode(["query: " + question], normalize_embeddings=True)[0]


# ==================================================
# 4. PETIT TEST DE RECHERCHE SÉMANTIQUE
# ==================================================

def chercher(question, modele, vecteurs, chunks, k=3):
    """Renvoie les k chunks dont le sens est le plus proche de la question."""
    vecteur_question = vectoriser_question(modele, question)
    scores = vecteurs @ vecteur_question            # similarité cosinus avec chaque chunk
    meilleurs = np.argsort(scores)[::-1][:k]        # indices des k meilleurs scores
    return [(chunks[i], float(scores[i])) for i in meilleurs]


# ==================================================
# 5. PROGRAMME PRINCIPAL
# ==================================================

if __name__ == "__main__":
    chunks = decouper(nettoyer_tout(extraire_tout()))
    modele = charger_modele()

    vecteurs = vectoriser_chunks(modele, chunks)
    print("\n========== BILAN DES EMBEDDINGS ==========")
    print(f"Chunks vectorisés      : {vecteurs.shape[0]}")
    print(f"Taille d'un vecteur    : {vecteurs.shape[1]} nombres")

    FICHIER_VECTEURS.parent.mkdir(parents=True, exist_ok=True)
    np.save(FICHIER_VECTEURS, vecteurs)
    sauvegarder_json(chunks, FICHIER_CHUNKS)
    print(f"Vecteurs enregistrés dans : {FICHIER_VECTEURS.relative_to(DOSSIER_PROJET)}")

    question = "J'ai perdu ma carte d'identité, que dois-je faire ?"
    print(f"\nTest de recherche : « {question} »")
    for chunk, score in chercher(question, modele, vecteurs, chunks):
        print(f"  {score:.3f}  {chunk['source']} (page {chunk['page']})")