"""
Tests fonctionnels du chatbot : chaque test vérifie qu'une partie du projet fonctionne.

Lancer depuis la racine du projet :  python tests/test_projet.py
Chaque test affiche  (réussi) ou (échoué, avec la raison).
"""
import re                   # expressions régulières (chercher une citation, une clé API)
import sys                  # sys.path (où Python cherche les modules) et sys.exit (code de sortie)
from pathlib import Path    # manipuler les chemins de fichiers

# tests/test_projet.py -> .parent = tests/ -> .parent.parent = le dossier du projet
DOSSIER_PROJET = Path(__file__).resolve().parent.parent
# Ce fichier est dans tests/, pas dans src/ : Python ne trouverait pas embeddings.py, etc.
# On ajoute src/ en tête (position 0) de la liste des dossiers où Python cherche les modules.
sys.path.insert(0, str(DOSSIER_PROJET / "src"))   # pour pouvoir importer les fichiers de src/

# Ces imports viennent APRÈS sys.path.insert, sinon ils échoueraient
from embeddings import charger_modele
from indexation import rechercher
from generation import ouvrir_collection, repondre, nettoyer_citations
from choix_formulaire import charger_correspondances, trouver_formulaires

NB_CHUNKS_ATTENDU = 232          # chunks de 600 caractères (étape 5)


# ---------- Les tests ----------
# Chaque test reçoit la collection et le modèle (chargés une seule fois plus bas).
# « assert condition, message » : si la condition est fausse, Python lève une AssertionError
# avec ce message -> le test est échoué. Sinon, le test renvoie un petit texte de détail.
# La docstring (« 1. La base ChromaDB... ») sert de titre affiché à l'écran.

def test_base(collection, modele):
    """1. La base ChromaDB existe et contient tous les chunks."""
    nb = collection.count()                 # nombre de chunks rangés dans la collection
    assert nb == NB_CHUNKS_ATTENDU, f"{nb} chunks au lieu de {NB_CHUNKS_ATTENDU}"
    return f"{nb} chunks dans la base"


def test_recherche(collection, modele):
    """2. Une question sur la CIN trouve une fiche CIN dans les 3 premiers résultats."""
    resultats = rechercher(collection, modele, "J'ai perdu ma carte d'identité, que dois-je faire ?", k=5)
    trois_premiers = [r["source"] for r in resultats[:3]]      # [:3] = les 3 premiers
    # any(...) = vrai si AU MOINS UN des 3 noms de fichiers commence par « CIN »
    assert any(s.startswith("CIN") for s in trois_premiers), f"trouvé : {trois_premiers}"
    return f"1er résultat : {resultats[0]['source']}"


def test_nettoyage(collection, modele):
    """3. Les citations mal écrites par le LLM sont remises au bon format."""
    # Dictionnaire : texte écrit par le LLM -> texte attendu après nettoyer_citations()
    cas = {
        "15 ans【CIN_premiere_fois.pdf, p. 1】": "15 ans[CIN_premiere_fois.pdf, p. 1]",       # crochets chinois
        "[**passeport_perte_tunisie.pdf**, p. 1]": "[passeport_perte_tunisie.pdf, p. 1]",   # gras dedans
        "*[CIN_renouvellement.pdf, p. 1]*": "[CIN_renouvellement.pdf, p. 1]",               # italique autour
        # le gras HORS citation doit rester intact (on ne casse pas la mise en forme normale)
        "**3 dinars** [CIN_renouvellement.pdf, p. 1]": "**3 dinars** [CIN_renouvellement.pdf, p. 1]",
    }
    for avant, attendu in cas.items():      # .items() = les couples (clé, valeur)
        obtenu = nettoyer_citations(avant)
        assert obtenu == attendu, f"« {avant} » a donné « {obtenu} »"
    return f"{len(cas)} formats corrigés"


def test_reponse_avec_source(collection, modele):
    """4. Une vraie question reçoit une réponse qui cite un PDF."""
    # « _ » : on ignore le 2e résultat (les extraits), seule la réponse nous intéresse
    reponse, _ = repondre("Combien coûte le renouvellement d'une carte d'identité en bon état ?",
                          collection, modele)
    assert "je ne trouve pas" not in reponse.lower(), "le chatbot n'a pas trouvé l'information"
    # On cherche au moins une citation au format exact [fichier.pdf, p. X] (\d+ = un ou plusieurs chiffres)
    assert re.search(r"\[[^\[\]]+\.pdf, p\. \d+\]", reponse), "aucune citation [fichier.pdf, p. X]"
    return "réponse trouvée avec une citation"


def test_refus(collection, modele):
    """5. Une question hors sujet donne « Je ne trouve pas » et aucun formulaire."""
    reponse, resultats = repondre("Quel temps fait-il à Tunis aujourd'hui ?", collection, modele)
    # reponse[:80] : les 80 premiers caractères, pour un message d'erreur court
    assert "je ne trouve pas" in reponse.lower(), f"réponse inattendue : {reponse[:80]}"
    # Vérifie aussi la règle 0 de choix_formulaire.py : pas de formulaire en cas de refus
    formulaires = trouver_formulaires(reponse, resultats, charger_correspondances())
    assert formulaires == [], f"formulaire proposé à tort : {formulaires}"
    return "refus correct, aucun formulaire"


def test_securite(collection, modele):
    """6. Aucune clé API dans le code, et .env est bien ignoré par git."""
    # On parcourt tous les .py de src/ + ce fichier de test
    for fichier in list((DOSSIER_PROJET / "src").glob("*.py")) + [Path(__file__)]:
        texte = fichier.read_text(encoding="utf-8")
        # Une clé Groq commence par « gsk_ » suivie d'au moins 20 lettres ou chiffres ({20,})
        assert not re.search(r"gsk_[A-Za-z0-9]{20,}", texte), f"clé API trouvée dans {fichier.name}"
    gitignore = (DOSSIER_PROJET / ".gitignore").read_text(encoding="utf-8")
    # .split() découpe le fichier en mots : on vérifie que « .env » y est comme une ligne à part entière
    # (et pas seulement à l'intérieur de « .env.example »)
    assert ".env" in gitignore.split(), ".env n'est pas dans .gitignore"
    return "pas de clé dans le code, .env ignoré par git"


# ---------- Lancement ----------

# La liste des fonctions de test (les fonctions elles-mêmes, sans les appeler : pas de parenthèses)
TESTS = [test_base, test_recherche, test_nettoyage,
         test_reponse_avec_source, test_refus, test_securite]

if __name__ == "__main__":
    print("Chargement du modèle et de la base...")
    # Chargés UNE fois et partagés par tous les tests (le modèle e5 met plusieurs secondes à charger)
    collection = ouvrir_collection()
    modele = charger_modele()

    reussis = 0                             # compteur de tests réussis
    for test in TESTS:
        titre = test.__doc__.strip()        # __doc__ = la docstring de la fonction, utilisée comme titre
        try:
            detail = test(collection, modele)
            print(f"✅ {titre}  ->  {detail}")
            reussis += 1
        except AssertionError as erreur:    # un assert a échoué : le test est raté
            print(f"❌ {titre}  ->  {erreur}")
        except Exception as erreur:                     # erreur inattendue (fichier manquant...)
            print(f"❌ {titre}  ->  erreur : {erreur}")
        # Grâce au try/except, un test raté n'arrête pas les autres : on voit tous les résultats

    print(f"\nRésultat : {reussis} / {len(TESTS)} tests réussis")
    # Code de sortie : 0 = tout a réussi, 1 = au moins un échec (convention utilisée par les outils)
    sys.exit(0 if reussis == len(TESTS) else 1)