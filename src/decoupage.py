"""
ÉTAPE 3 — DÉCOUPAGE EN CHUNKS

Découpe le texte nettoyé de chaque page en morceaux (chunks) de 600 caractères au plus,
avec 80 caractères de chevauchement, et ajoute à chaque chunk ses métadonnées :
document, chemin, page, type (procédure / règlement / guide) et public (tunisie / etranger / tous).
Résultat : data/processed/chunks.json

Lancer :  python src/decoupage.py
"""
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from extraction import extraire_tout, sauvegarder_json
from nettoyage import nettoyer_tout

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_SORTIE = DOSSIER_PROJET / "data" / "processed" / "chunks.json"

TAILLE_CHUNK = 600      # nombre maximal de caractères par chunk
CHEVAUCHEMENT = 80      # caractères répétés entre deux chunks qui se suivent


def type_document(nom_fichier):
    """Déduit le type du document à partir de son nom."""
    if nom_fichier.startswith("loi_"):
        return "reglement"
    if nom_fichier.startswith("guide_"):
        return "guide"
    return "procedure"


def public_document(nom_fichier):
    """Déduit à qui s'adresse le document : Tunisie, étranger ou tous."""
    if "etranger" in nom_fichier:
        return "etranger"
    if "tunisie" in nom_fichier:
        return "tunisie"
    return "tous"


def decouper(pages, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT):
    """Découpe chaque page en chunks et ajoute les métadonnées."""
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=taille,
        chunk_overlap=chevauchement,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        morceaux = decoupeur.split_text(page["texte"])
        for numero, morceau in enumerate(morceaux):
            chunks.append({
                "id": f"{page['source']}_p{page['page']}_c{numero}",
                "texte": morceau,
                "source": page["source"],
                "chemin": page["chemin"],
                "page": page["page"],
                "type": type_document(page["source"]),
                "public": public_document(page["source"]),
            })
    return chunks


if __name__ == "__main__":
    pages = nettoyer_tout(extraire_tout())
    chunks = decouper(pages)

    longueurs = [len(c["texte"]) for c in chunks]
    print("\n========== BILAN DU DÉCOUPAGE ==========")
    print(f"Chunks créés           : {len(chunks)}")
    print(f"Longueur moyenne       : {sum(longueurs) // len(longueurs)} caractères")
    print(f"Plus court / plus long : {min(longueurs)} / {max(longueurs)} caractères")

    sauvegarder_json(chunks, FICHIER_SORTIE)