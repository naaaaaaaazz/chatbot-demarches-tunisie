"""
Tests fonctionnels du chatbot : chaque test vérifie qu'une partie du projet fonctionne.

Lancer depuis la racine du projet :  python tests/test_projet.py
Chaque test affiche ✅ (réussi) ou ❌ (échoué, avec la raison).
"""
import re
import sys
from pathlib import Path

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DOSSIER_PROJET / "src"))   # pour pouvoir importer les fichiers de src/

from embeddings import charger_modele
from indexation import rechercher
from generation import ouvrir_collection, repondre, nettoyer_citations
from choix_formulaire import charger_correspondances, trouver_formulaires

NB_CHUNKS_ATTENDU = 232          # chunks de 600 caractères (étape 5)


# ---------- Les tests ----------

def test_base(collection, modele):
    """1. La base ChromaDB existe et contient tous les chunks."""
    nb = collection.count()
    assert nb == NB_CHUNKS_ATTENDU, f"{nb} chunks au lieu de {NB_CHUNKS_ATTENDU}"
    return f"{nb} chunks dans la base"


def test_recherche(collection, modele):
    """2. Une question sur la CIN trouve une fiche CIN dans les 3 premiers résultats."""
    resultats = rechercher(collection, modele, "J'ai perdu ma carte d'identité, que dois-je faire ?", k=5)
    trois_premiers = [r["source"] for r in resultats[:3]]
    assert any(s.startswith("CIN") for s in trois_premiers), f"trouvé : {trois_premiers}"
    return f"1er résultat : {resultats[0]['source']}"


def test_nettoyage(collection, modele):
    """3. Les citations mal écrites par le LLM sont remises au bon format."""
    cas = {
        "15 ans【CIN_premiere_fois.pdf, p. 1】": "15 ans[CIN_premiere_fois.pdf, p. 1]",
        "[**passeport_perte_tunisie.pdf**, p. 1]": "[passeport_perte_tunisie.pdf, p. 1]",
        "*[CIN_renouvellement.pdf, p. 1]*": "[CIN_renouvellement.pdf, p. 1]",
        "**3 dinars** [CIN_renouvellement.pdf, p. 1]": "**3 dinars** [CIN_renouvellement.pdf, p. 1]",
    }
    for avant, attendu in cas.items():
        obtenu = nettoyer_citations(avant)
        assert obtenu == attendu, f"« {avant} » a donné « {obtenu} »"
    return f"{len(cas)} formats corrigés"


def test_reponse_avec_source(collection, modele):
    """4. Une vraie question reçoit une réponse qui cite un PDF."""
    reponse, _ = repondre("Combien coûte le renouvellement d'une carte d'identité en bon état ?",
                          collection, modele)
    assert "je ne trouve pas" not in reponse.lower(), "le chatbot n'a pas trouvé l'information"
    assert re.search(r"\[[^\[\]]+\.pdf, p\. \d+\]", reponse), "aucune citation [fichier.pdf, p. X]"
    return "réponse trouvée avec une citation"


def test_refus(collection, modele):
    """5. Une question hors sujet donne « Je ne trouve pas » et aucun formulaire."""
    reponse, resultats = repondre("Quel temps fait-il à Tunis aujourd'hui ?", collection, modele)
    assert "je ne trouve pas" in reponse.lower(), f"réponse inattendue : {reponse[:80]}"
    formulaires = trouver_formulaires(reponse, resultats, charger_correspondances())
    assert formulaires == [], f"formulaire proposé à tort : {formulaires}"
    return "refus correct, aucun formulaire"


def test_securite(collection, modele):
    """6. Aucune clé API dans le code, et .env est bien ignoré par git."""
    for fichier in list((DOSSIER_PROJET / "src").glob("*.py")) + [Path(__file__)]:
        texte = fichier.read_text(encoding="utf-8")
        assert not re.search(r"gsk_[A-Za-z0-9]{20,}", texte), f"clé API trouvée dans {fichier.name}"
    gitignore = (DOSSIER_PROJET / ".gitignore").read_text(encoding="utf-8")
    assert ".env" in gitignore.split(), ".env n'est pas dans .gitignore"
    return "pas de clé dans le code, .env ignoré par git"


# ---------- Lancement ----------

TESTS = [test_base, test_recherche, test_nettoyage,
         test_reponse_avec_source, test_refus, test_securite]

if __name__ == "__main__":
    print("Chargement du modèle et de la base...")
    collection = ouvrir_collection()
    modele = charger_modele()

    reussis = 0
    for test in TESTS:
        titre = test.__doc__.strip()
        try:
            detail = test(collection, modele)
            print(f"✅ {titre}  ->  {detail}")
            reussis += 1
        except AssertionError as erreur:
            print(f"❌ {titre}  ->  {erreur}")
        except Exception as erreur:                     # erreur inattendue (fichier manquant...)
            print(f"❌ {titre}  ->  erreur : {erreur}")

    print(f"\nRésultat : {reussis} / {len(TESTS)} tests réussis")
    sys.exit(0 if reussis == len(TESTS) else 1)