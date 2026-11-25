"""
Étape 8d : calculer les résultats de la notation manuelle
(colonnes exactitude et fidelite remplies de 0 à 2).

  python src/calcul_resultats.py                      -> evaluation/reponses_llm.csv (Groq)
  python src/calcul_resultats.py reponses_ollama.csv  -> les anciens résultats (Ollama)
"""
import csv                  # lire le CSV noté
import sys                  # sys.argv : les mots écrits après le nom du script dans le terminal
from pathlib import Path    # manipuler les chemins de fichiers

DOSSIER_PROJET = Path(__file__).resolve().parent.parent      # src/ -> dossier du projet
# sys.argv[0] = le nom du script ; sys.argv[1] = le 1er argument (ex. « reponses_ollama.csv »).
# S'il n'y a pas d'argument, on prend le fichier Groq par défaut.
NOM_FICHIER = sys.argv[1] if len(sys.argv) > 1 else "reponses_llm.csv"
FICHIER_REPONSES = DOSSIER_PROJET / "evaluation" / NOM_FICHIER
# Les questions 1 à 30 sont dans le sujet ; les questions 31 à 35 sont hors sujet (refus attendu)
NB_QUESTIONS_DANS_LE_SUJET = 30


def lire_lignes():
    """Lit le fichier noté (Excel peut l'enregistrer avec ; ou ,)."""
    octets = FICHIER_REPONSES.read_bytes()       # on lit le fichier brut, sans choisir l'encodage
    # Le fichier peut être en UTF-8 (écrit par Python) ou en cp1252 (réenregistré par Excel) :
    # on essaie UTF-8, et en cas d'erreur on essaie l'encodage Windows
    try:
        contenu = octets.decode("utf-8-sig")
    except UnicodeDecodeError:
        contenu = octets.decode("cp1252")  # fichier réenregistré par Excel (accents Windows)
    # On regarde la 1re ligne (les noms des colonnes) : si elle contient « ; », c'est le séparateur
    separateur = ";" if contenu.splitlines()[0].count(";") > 0 else ","
    lignes = csv.DictReader(contenu.splitlines(), delimiter=separateur)
    # (l.get("id") or "") : évite une erreur si la case id est absente (None) ; .strip() enlève les espaces
    return [l for l in lignes if (l.get("id") or "").strip()]   # ignore les lignes vides ajoutées par Excel


def moyenne(valeurs):
    """Moyenne d'une liste de nombres (0 si la liste est vide, pour éviter une division par zéro)."""
    return sum(valeurs) / len(valeurs) if valeurs else 0


# S'exécute seulement avec « python src/calcul_resultats.py »
if __name__ == "__main__":
    print(f"Fichier analysé : {NOM_FICHIER}")
    lignes = lire_lignes()
    # Une question est « non notée » si la case exactitude OU la case fidélité est vide
    non_notees = [l["id"] for l in lignes if not l["exactitude"].strip() or not l["fidelite"].strip()]
    if non_notees:
        print(f"⚠ Questions pas encore notées : {', '.join(non_notees)}")   # join : « 3, 7, 12 »

    # On sépare les questions notées en deux groupes, selon leur numéro
    notees = [l for l in lignes if l["id"] not in non_notees]
    dans_le_sujet = [l for l in notees if int(l["id"]) <= NB_QUESTIONS_DANS_LE_SUJET]
    hors_sujet = [l for l in notees if int(l["id"]) > NB_QUESTIONS_DANS_LE_SUJET]

    # Les notes sont du texte dans le CSV : int() les transforme en nombres
    exactitudes = [int(l["exactitude"]) for l in dans_le_sujet]
    fidelites = [int(l["fidelite"]) for l in dans_le_sujet]
    # Temps : « 1,8 » -> « 1.8 » -> 1.8 (Python attend un point, pas une virgule)
    temps = [float(l["temps_s"].replace(",", ".")) for l in lignes]

    print(f"\n===== Questions dans le sujet ({len(dans_le_sujet)}) =====")
    print(f"Exactitude moyenne : {moyenne(exactitudes):.2f} / 2")
    print(f"Fidélité moyenne   : {moyenne(fidelites):.2f} / 2")
    # Répartition des notes : combien de 2, de 1 et de 0 ; :>2 = aligné à droite sur 2 caractères
    for note in (2, 1, 0):
        print(f"  note {note} : exactitude {exactitudes.count(note):>2} | fidélité {fidelites.count(note):>2}")

    # Hors sujet : un refus correct est noté 2 en exactitude ; on compte ces questions
    refus = sum(1 for l in hors_sujet if l["exactitude"].strip() == "2")
    print(f"\n===== Questions hors sujet ({len(hors_sujet)}) =====")
    print(f"Refus corrects : {refus} / {len(hors_sujet)}")

    print(f"\n===== Temps =====")
    # :.0f = arrondi sans chiffre après la virgule ; min / max = réponse la plus rapide / la plus lente
    print(f"Temps moyen par réponse : {moyenne(temps):.0f} s (de {min(temps):.0f} à {max(temps):.0f} s)")