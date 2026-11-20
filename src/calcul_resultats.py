"""
Étape 8d : calculer les résultats de la notation manuelle
(evaluation/reponses_llm.csv, colonnes exactitude et fidelite remplies de 0 à 2).
"""
import csv
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
FICHIER_REPONSES = DOSSIER_PROJET / "evaluation" / "reponses_llm.csv"
NB_QUESTIONS_DANS_LE_SUJET = 30


def lire_lignes():
    """Lit le fichier noté (Excel peut l'enregistrer avec ; ou ,)."""
    octets = FICHIER_REPONSES.read_bytes()
    try:
        contenu = octets.decode("utf-8-sig")
    except UnicodeDecodeError:
        contenu = octets.decode("cp1252")  # fichier réenregistré par Excel (accents Windows)
    separateur = ";" if contenu.splitlines()[0].count(";") > 0 else ","
    return list(csv.DictReader(contenu.splitlines(), delimiter=separateur))


def moyenne(valeurs):
    return sum(valeurs) / len(valeurs) if valeurs else 0


if __name__ == "__main__":
    lignes = lire_lignes()
    non_notees = [l["id"] for l in lignes if not l["exactitude"].strip() or not l["fidelite"].strip()]
    if non_notees:
        print(f"⚠ Questions pas encore notées : {', '.join(non_notees)}")

    notees = [l for l in lignes if l["id"] not in non_notees]
    dans_le_sujet = [l for l in notees if int(l["id"]) <= NB_QUESTIONS_DANS_LE_SUJET]
    hors_sujet = [l for l in notees if int(l["id"]) > NB_QUESTIONS_DANS_LE_SUJET]

    exactitudes = [int(l["exactitude"]) for l in dans_le_sujet]
    fidelites = [int(l["fidelite"]) for l in dans_le_sujet]
    temps = [float(l["temps_s"].replace(",", ".")) for l in lignes]

    print(f"\n===== Questions dans le sujet ({len(dans_le_sujet)}) =====")
    print(f"Exactitude moyenne : {moyenne(exactitudes):.2f} / 2")
    print(f"Fidélité moyenne   : {moyenne(fidelites):.2f} / 2")
    for note in (2, 1, 0):
        print(f"  note {note} : exactitude {exactitudes.count(note):>2} | fidélité {fidelites.count(note):>2}")

    refus = sum(1 for l in hors_sujet if l["exactitude"].strip() == "2")
    print(f"\n===== Questions hors sujet ({len(hors_sujet)}) =====")
    print(f"Refus corrects : {refus} / {len(hors_sujet)}")

    print(f"\n===== Temps =====")
    print(f"Temps moyen par réponse : {moyenne(temps):.0f} s (de {min(temps):.0f} à {max(temps):.0f} s)")