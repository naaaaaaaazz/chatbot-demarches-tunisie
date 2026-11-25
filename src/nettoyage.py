"""
ÉTAPE 2 — NETTOYAGE DU TEXTE

Reprend les pages extraites à l'étape 1 et supprime le bruit repéré dans nos PDF :
numéros de page, en-têtes du site legislation-securite.tn, tirets isolés, espaces en trop.
Les métadonnées (document, chemin, page) sont conservées.
Résultat : data/processed/pages_nettoyees.json

Lancer :  python src/nettoyage.py
"""
import re                   # « re » = expressions régulières : chercher des motifs dans du texte
from pathlib import Path    # pour manipuler les chemins de fichiers

# On réutilise les fonctions de l'étape 1 au lieu de les réécrire
from extraction import extraire_tout, sauvegarder_json


# ==================================================
# 1. CHEMINS
# ==================================================

# Dossier du projet (on remonte de src/ vers la racine), puis fichier de sortie
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_SORTIE = DOSSIER_PROJET / "data" / "processed" / "pages_nettoyees.json"


# ==================================================
# 2. LES MOTIFS DE BRUIT (repérés dans NOS PDF)
# ==================================================

# Ils viennent surtout du site legislation-securite.tn (lois 2015-46 et 2024-22).
# Le « r » devant les chaînes = « raw string » : les \ sont gardés tels quels pour les regex.
MOTIFS_BRUIT = [
    r"Page \d+ sur \d+",                     # numéro de page : « Page 2 sur 4 » (\d+ = un ou plusieurs chiffres)
    r"Base de données",                      # en-tête du site
    r"LA LEGISLATION DU SECTEUR DE LA SECURITE EN TUNISIE",
    r"www\.legislation-securite\.tn",        # adresse du site (en-tête) ; \. = un vrai point
    r"Accès aux lois, décrets et autres textes juridiques consolidés et mis à jour",
    r"[−–-]",                                # tiret tout seul sur une ligne ([...] = un des 3 tirets)
]
# re.compile « prépare » chaque motif une fois pour toutes (plus rapide quand on l'utilise souvent).
# re.IGNORECASE : majuscules et minuscules sont traitées pareil.
REGEX_BRUIT = [re.compile(motif, re.IGNORECASE) for motif in MOTIFS_BRUIT]


def est_du_bruit(ligne):
    """Renvoie True si la ligne ENTIÈRE correspond à un motif de bruit."""
    # fullmatch = la ligne doit correspondre au motif du début à la fin.
    # Ainsi, une vraie phrase qui CONTIENT « Base de données » n'est jamais supprimée.
    # any(...) = True si au moins un des motifs correspond.
    return any(regex.fullmatch(ligne) for regex in REGEX_BRUIT)


# ==================================================
# 3. NETTOYER LE TEXTE D'UNE PAGE
# ==================================================

def nettoyer_texte(texte):
    """Nettoie le texte d'une page, ligne par ligne."""
    lignes_propres = []
    for ligne in texte.splitlines():                  # splitlines() coupe le texte à chaque retour à la ligne
        ligne = re.sub(r"\s+", " ", ligne).strip()   # espaces multiples -> un seul espace ; strip() enlève ceux des bords
        if ligne == "":
            # Ligne vide : on en garde UNE seule entre deux paragraphes
            # (lignes_propres[-1] = la dernière ligne ajoutée ; on n'ajoute pas deux vides de suite)
            if lignes_propres and lignes_propres[-1] != "":
                lignes_propres.append("")
            continue                                  # continue = passer directement à la ligne suivante
        if est_du_bruit(ligne):
            continue                                  # ligne de bruit : on la saute
        lignes_propres.append(ligne)                  # ligne normale : on la garde
    # On recolle les lignes avec des retours à la ligne ; strip() enlève les lignes vides au début et à la fin
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
            # Même dictionnaire qu'à l'étape 1 : seul le texte change,
            # le nom du fichier et le numéro de page sont gardés pour les citations
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

# S'exécute seulement quand on lance « python src/nettoyage.py »
if __name__ == "__main__":
    pages = extraire_tout()                 # étape 1 : lire les PDF
    pages_propres = nettoyer_tout(pages)    # étape 2 : nettoyer

    # Bilan : combien de caractères ont été supprimés (chez nous : 113 321 -> 112 371)
    avant = sum(len(p["texte"]) for p in pages)
    apres = sum(len(p["texte"]) for p in pages_propres)
    print("\n========== BILAN DU NETTOYAGE ==========")
    print(f"Pages nettoyées        : {len(pages_propres)}")
    print(f"Caractères avant       : {avant}")
    print(f"Caractères après       : {apres}")
    print(f"Caractères supprimés   : {avant - apres}")

    # On réutilise la fonction de l'étape 1, mais avec un autre nom de fichier
    sauvegarder_json(pages_propres, FICHIER_SORTIE)