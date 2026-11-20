"""
ÉTAPE 1 — EXTRACTION DU TEXTE DES PDF

Lit tous les PDF du dossier data/ (et de ses sous-dossiers), page par page,
et enregistre le texte avec ses métadonnées (document, chemin, page)
dans data/processed/pages_extraites.json.

Lancer :  python src/extraction.py
"""
import json
from pathlib import Path

import pymupdf


# ==================================================
# 1. CHEMINS
# ==================================================

# Dossier du projet : on part de ce fichier (src/extraction.py) et on remonte de 2 niveaux
DOSSIER_PROJET = Path(__file__).resolve().parent.parent

# Dossier qui contient les 20 PDF (procedures/, reglements/, guides/)
DOSSIER_PDF = DOSSIER_PROJET / "data"

# Dossier et fichier où l'on enregistre le résultat
DOSSIER_SORTIE = DOSSIER_PROJET / "data" / "processed"
FICHIER_SORTIE = DOSSIER_SORTIE / "pages_extraites.json"

# En dessous de ce nombre de caractères, le PDF est probablement scanné (une image)
SEUIL_TEXTE_COURT = 100


# ==================================================
# 2. EXTRAIRE UN SEUL PDF
# ==================================================

def extraire_pdf(chemin_pdf):
    """Lit un PDF et renvoie la liste de ses pages, avec leurs métadonnées."""
    pages = []
    with pymupdf.open(chemin_pdf) as document:      # « with » ferme le PDF automatiquement
        for numero, page in enumerate(document, start=1):
            pages.append({
                "source": chemin_pdf.name,                                 # nom du fichier
                "chemin": chemin_pdf.relative_to(DOSSIER_PDF).as_posix(),  # ex. procedures/CIN/...
                "page": numero,                                            # numéro de page (à partir de 1)
                "texte": page.get_text("text"),                            # texte brut de la page
            })
    return pages


# ==================================================
# 3. EXTRAIRE TOUS LES PDF
# ==================================================

def extraire_tout():
    """Parcourt tous les PDF de data/ et renvoie toutes leurs pages."""
    toutes_les_pages = []
    fichiers_pdf = sorted(DOSSIER_PDF.rglob("*.pdf"))
    print(f"Nombre de PDF trouvés : {len(fichiers_pdf)}\n")

    erreurs = 0
    for chemin_pdf in fichiers_pdf:
        try:
            pages = extraire_pdf(chemin_pdf)
            toutes_les_pages.extend(pages)

            nb_caracteres = sum(len(p["texte"].strip()) for p in pages)
            if nb_caracteres < SEUIL_TEXTE_COURT:
                print(f"ATTENTION : texte très court (PDF scanné ?) : {chemin_pdf.name}")
            else:
                print(f"OK : {chemin_pdf.name} ({len(pages)} page(s), {nb_caracteres} caractères)")

        except Exception as erreur:
            erreurs += 1
            print(f"ERREUR avec {chemin_pdf.name} : {erreur}")

    print("\n========== BILAN DE L'EXTRACTION ==========")
    print(f"PDF trouvés      : {len(fichiers_pdf)}")
    print(f"PDF en erreur    : {erreurs}")
    print(f"Pages extraites  : {len(toutes_les_pages)}")
    return toutes_les_pages


# ==================================================
# 4. SAUVEGARDER LE RÉSULTAT
# ==================================================

def sauvegarder_json(pages, fichier=FICHIER_SORTIE):
    """Enregistre les pages dans un fichier JSON lisible (accents conservés)."""
    fichier.parent.mkdir(parents=True, exist_ok=True)   # crée data/processed/ s'il n'existe pas
    with open(fichier, "w", encoding="utf-8") as f:
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\nRésultat enregistré dans : {fichier.relative_to(DOSSIER_PROJET)}")


# ==================================================
# 5. PROGRAMME PRINCIPAL
# ==================================================

if __name__ == "__main__":
    pages = extraire_tout()
    sauvegarder_json(pages)