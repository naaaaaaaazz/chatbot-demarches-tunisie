"""
ÉTAPE 1 — EXTRACTION DU TEXTE DES PDF

Lit tous les PDF du dossier data/ (et de ses sous-dossiers), page par page,
et enregistre le texte avec ses métadonnées (document, chemin, page)
dans data/processed/pages_extraites.json.

Lancer :  python src/extraction.py
"""
import json                 # pour enregistrer le résultat dans un fichier .json
from pathlib import Path    # pour manipuler les chemins de fichiers (marche sous Windows et Linux)

import pymupdf              # bibliothèque qui lit les PDF (installée avec pip install pymupdf)


# ==================================================
# 1. CHEMINS
# ==================================================

# Dossier du projet : on part de ce fichier (src/extraction.py) et on remonte de 2 niveaux
# __file__ = chemin de ce fichier ; .resolve() = chemin complet ; .parent = dossier au-dessus
DOSSIER_PROJET = Path(__file__).resolve().parent.parent

# Dossier qui contient les 20 PDF (procedures/, reglements/, guides/)
# L'opérateur « / » de Path colle deux morceaux de chemin : projet/data
DOSSIER_PDF = DOSSIER_PROJET / "data"

# Dossier et fichier où l'on enregistre le résultat
DOSSIER_SORTIE = DOSSIER_PROJET / "data" / "processed"
FICHIER_SORTIE = DOSSIER_SORTIE / "pages_extraites.json"

# En dessous de ce nombre de caractères, le PDF est probablement scanné (une image)
# Un PDF scanné ne contient pas de texte : pymupdf renverrait presque rien (il faudrait un OCR)
SEUIL_TEXTE_COURT = 100


# ==================================================
# 2. EXTRAIRE UN SEUL PDF
# ==================================================

def extraire_pdf(chemin_pdf):
    """Lit un PDF et renvoie la liste de ses pages, avec leurs métadonnées."""
    pages = []                                       # liste vide qu'on va remplir, une case par page
    with pymupdf.open(chemin_pdf) as document:      # « with » ferme le PDF automatiquement
        # enumerate(..., start=1) donne en même temps le numéro (1, 2, 3...) et la page
        for numero, page in enumerate(document, start=1):
            # Chaque page devient un dictionnaire : c'est ce qui permettra plus tard
            # de citer « [nom_du_fichier.pdf, p. X] » dans les réponses du chatbot
            pages.append({
                "source": chemin_pdf.name,                                 # nom du fichier
                "chemin": chemin_pdf.relative_to(DOSSIER_PDF).as_posix(),  # ex. procedures/CIN/...
                "page": numero,                                            # numéro de page (à partir de 1)
                "texte": page.get_text("text"),                            # texte brut de la page
            })
    return pages                                     # liste de dictionnaires (une par page)


# ==================================================
# 3. EXTRAIRE TOUS LES PDF
# ==================================================

def extraire_tout():
    """Parcourt tous les PDF de data/ et renvoie toutes leurs pages."""
    toutes_les_pages = []                            # toutes les pages de tous les PDF
    # rglob("*.pdf") cherche les PDF dans data/ ET dans tous ses sous-dossiers
    # sorted(...) les range par ordre alphabétique (même ordre à chaque lancement)
    fichiers_pdf = sorted(DOSSIER_PDF.rglob("*.pdf"))
    print(f"Nombre de PDF trouvés : {len(fichiers_pdf)}\n")

    erreurs = 0                                      # compteur de PDF illisibles
    for chemin_pdf in fichiers_pdf:
        # try / except : si un PDF est abîmé, on affiche l'erreur et on passe au suivant,
        # au lieu d'arrêter tout le programme
        try:
            pages = extraire_pdf(chemin_pdf)
            toutes_les_pages.extend(pages)           # extend ajoute toutes les pages à la liste

            # Nombre total de caractères du PDF (strip() enlève les espaces autour)
            nb_caracteres = sum(len(p["texte"].strip()) for p in pages)
            if nb_caracteres < SEUIL_TEXTE_COURT:
                print(f"ATTENTION : texte très court (PDF scanné ?) : {chemin_pdf.name}")
            else:
                print(f"OK : {chemin_pdf.name} ({len(pages)} page(s), {nb_caracteres} caractères)")

        except Exception as erreur:
            erreurs += 1
            print(f"ERREUR avec {chemin_pdf.name} : {erreur}")

    # Résumé affiché à la fin (chez nous : 20 PDF, 0 erreur, 44 pages)
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
    with open(fichier, "w", encoding="utf-8") as f:     # "w" = écriture ; utf-8 pour les accents
        # ensure_ascii=False garde les accents (é au lieu de é) ; indent=2 rend le fichier lisible
        json.dump(pages, f, ensure_ascii=False, indent=2)
    print(f"\nRésultat enregistré dans : {fichier.relative_to(DOSSIER_PROJET)}")


# ==================================================
# 5. PROGRAMME PRINCIPAL
# ==================================================

# Ce bloc ne s'exécute que si on lance CE fichier (python src/extraction.py).
# Si un autre fichier fait « from extraction import ... », il ne s'exécute pas.
if __name__ == "__main__":
    pages = extraire_tout()     # 1) lire les 20 PDF
    sauvegarder_json(pages)     # 2) enregistrer le résultat dans pages_extraites.json