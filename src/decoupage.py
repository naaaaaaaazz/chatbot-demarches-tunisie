"""
ÉTAPE 3 — DÉCOUPAGE EN CHUNKS

Découpe le texte nettoyé de chaque page en morceaux (chunks) de 600 caractères au plus,
avec 80 caractères de chevauchement, et ajoute à chaque chunk ses métadonnées :
document, chemin, page, type (procédure / règlement / guide) et public (tunisie / etranger / tous).
Résultat : data/processed/chunks.json

Lancer :  python src/decoupage.py
"""
from pathlib import Path    # pour manipuler les chemins de fichiers

# Découpeur de texte de LangChain (installé avec pip install langchain-text-splitters)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# On réutilise les étapes 1 (extraction) et 2 (nettoyage)
from extraction import extraire_tout, sauvegarder_json
from nettoyage import nettoyer_tout

# Racine du projet et fichier où l'on enregistre les chunks
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_SORTIE = DOSSIER_PROJET / "data" / "processed" / "chunks.json"

# Valeurs choisies après comparaison : 600 caractères donnent un rappel@5 de 1,00
# (contre 0,97 avec 300), car un article de loi garde son contexte.
TAILLE_CHUNK = 600      # nombre maximal de caractères par chunk
CHEVAUCHEMENT = 80      # caractères répétés entre deux chunks qui se suivent
                        # -> une phrase coupée entre deux chunks reste lisible en entier dans l'un d'eux


def type_document(nom_fichier):
    """Déduit le type du document à partir de son nom."""
    # Convention de nommage de nos PDF : loi_... = loi, guide_... = guide, le reste = fiche pratique.
    # Le type sert plus tard à donner un petit bonus aux fiches de procédure dans la recherche.
    if nom_fichier.startswith("loi_"):
        return "reglement"
    if nom_fichier.startswith("guide_"):
        return "guide"
    return "procedure"


def public_document(nom_fichier):
    """Déduit à qui s'adresse le document : Tunisie, étranger ou tous."""
    # Le public sert à filtrer la recherche : quelqu'un à l'étranger ne reçoit
    # que les documents « etranger » et « tous » (consulat, pas poste de police).
    if "etranger" in nom_fichier:
        return "etranger"
    if "tunisie" in nom_fichier:
        return "tunisie"
    return "tous"


def decouper(pages, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT):
    """Découpe chaque page en chunks et ajoute les métadonnées."""
    # Découpage « récursif » : il essaie de couper d'abord entre deux paragraphes (\n\n),
    # puis à un retour à la ligne (\n), puis à la fin d'une phrase (". "), puis entre deux mots (" "),
    # et en tout dernier recours n'importe où (""). Ainsi on coupe rarement au milieu d'une phrase.
    decoupeur = RecursiveCharacterTextSplitter(
        chunk_size=taille,
        chunk_overlap=chevauchement,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:                                   # on découpe page par page :
        morceaux = decoupeur.split_text(page["texte"])   # un chunk ne mélange jamais deux pages
        for numero, morceau in enumerate(morceaux):      # numero = 0, 1, 2... dans la page
            chunks.append({
                # Identifiant unique, ex. CIN_renouvellement.pdf_p1_c2 (fichier, page 1, chunk 2)
                "id": f"{page['source']}_p{page['page']}_c{numero}",
                "texte": morceau,
                # Métadonnées gardées pour les citations [fichier, p. X] :
                "source": page["source"],
                "chemin": page["chemin"],
                "page": page["page"],
                # Métadonnées utilisées par la recherche (bonus et filtre) :
                "type": type_document(page["source"]),
                "public": public_document(page["source"]),
            })
    return chunks


# S'exécute seulement quand on lance « python src/decoupage.py »
if __name__ == "__main__":
    pages = nettoyer_tout(extraire_tout())   # étapes 1 et 2
    chunks = decouper(pages)                 # étape 3

    # Bilan (chez nous : 232 chunks, 503 caractères en moyenne)
    longueurs = [len(c["texte"]) for c in chunks]
    print("\n========== BILAN DU DÉCOUPAGE ==========")
    print(f"Chunks créés           : {len(chunks)}")
    print(f"Longueur moyenne       : {sum(longueurs) // len(longueurs)} caractères")   # // = division entière
    print(f"Plus court / plus long : {min(longueurs)} / {max(longueurs)} caractères")

    sauvegarder_json(chunks, FICHIER_SORTIE)