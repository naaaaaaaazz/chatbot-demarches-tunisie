"""
ÉTAPE 2 — NETTOYAGE DU TEXTE

Reprend les pages extraites à l'étape 1 et supprime le bruit repéré dans nos PDF :
numéros de page, en-têtes du site legislation-securite.tn, tirets isolés, espaces en trop.
Les métadonnées (document, chemin, page) sont conservées.
Résultat : data/processed/pages_nettoyees.json

Lancer :  python src/nettoyage.py
"""
import re
from pathlib import Path

from extraction import extraire_tout, sauvegarder_json


# ==================================================
# 1. CHEMINS
# ==================================================

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_SORTIE = DOSSIER_PROJET / "data" / "processed" / "pages_nettoyees.json"


# ==================================================
# 2. LES MOTIFS DE BRUIT (repérés dans NOS PDF)
# ==================================================

# Ils viennent surtout du site legislation-securite.tn (lois 2015-46 et 2024-22).
MOTIFS_BRUIT = [
    r"Page \d+ sur \d+",                     # numéro de page : « Page 2 sur 4 »
    r"Base de données",                      # en-tête du site
    r"LA LEGISLATION DU SECTEUR DE LA SECURITE EN TUNISIE",
    r"www\.legislation-securite\.tn",        # adresse du site (en-tête)
    r"Accès aux lois, décrets et autres textes juridiques consolidés et mis à jour",
    r"[−–-]",                                # tiret tout seul sur une ligne
]
REGEX_BRUIT = [re.compile(motif, re.IGNORECASE) for motif in MOTIFS_BRUIT]


def est_du_bruit(ligne):
    """Renvoie True si la ligne ENTIÈRE correspond à un motif de bruit."""
    return any(regex.fullmatch(ligne) for regex in REGEX_BRUIT)


# ==================================================
# 3. NETTOYER LE TEXTE D'UNE PAGE
# ==================================================

def nettoyer_texte(texte):
    """Nettoie le texte d'une page, ligne par ligne."""
    lignes_propres = []
    for ligne in texte.splitlines():
        ligne = re.sub(r"\s+", " ", ligne).strip()   # espaces multiples -> un seul espace
        if ligne == "":
            # Ligne vide : on en garde UNE seule entre deux paragraphes
            if lignes_propres and lignes_propres[-1] != "":
                lignes_propres.append("")
            continue
        if est_du_bruit(ligne):
            continue                                  # ligne de bruit : on la saute
        lignes_propres.append(ligne)
    return "\n".join(lignes_propres).strip()


# ==================================================
# 4. NETTOYER TOUTES LES PAGES
# ==================================================

def nettoyer_tout(pages):
    """Nettoie toutes les pages en gardant leurs métadonnées."""
    pages_propres = []
    for page in pages:
        texte = nettoyer_texte(page["texte"])
        if texte:                                     # on ignore une page devenue vide
            pages_propres.append({
                "source": page["source"],
                "chemin": page["chemin"],
                "page": page["page"],
                "texte": texte,
            })
    return pages_propres


# ==================================================
# 5. PROGRAMME PRINCIPAL
# ==================================================

if __name__ == "__main__":
    pages = extraire_tout()
    pages_propres = nettoyer_tout(pages)

    avant = sum(len(p["texte"]) for p in pages)
    apres = sum(len(p["texte"]) for p in pages_propres)
    print("\n========== BILAN DU NETTOYAGE ==========")
    print(f"Pages nettoyées        : {len(pages_propres)}")
    print(f"Caractères avant       : {avant}")
    print(f"Caractères après       : {apres}")
    print(f"Caractères supprimés   : {avant - apres}")

    sauvegarder_json(pages_propres, FICHIER_SORTIE)