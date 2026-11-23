"""
Étape 7b : interface web du chatbot (Streamlit).
Lancer depuis la racine du projet avec : streamlit run src/app.py
"""
import base64
from pathlib import Path

import pymupdf
import streamlit as st

from embeddings import charger_modele
from generation import ouvrir_collection, preparer, generer_reponse_flux
from choix_formulaire import charger_correspondances, trouver_formulaires, DOSSIER_FORMULAIRES

st.set_page_config(page_title="Assistant démarches administratives",
                   page_icon="🏛️", layout="wide")

DOSSIER_PROJET = Path(__file__).resolve().parent.parent
DOSSIER_IMAGES = DOSSIER_PROJET / "images"
DOSSIER_DATA = DOSSIER_PROJET / "data"

PUBLICS = {"Je ne sais pas": None,
           "En Tunisie": "tunisie",
           "À l'étranger (consulat)": "etranger"}

# Les 3 cartes de la page d'accueil : titre, description, image, questions d'exemple
CARTES = [
    ("Carte d'identité", "Première demande, renouvellement, perte, modification",
     "cin.png", ["J'ai perdu ma carte d'identité, que dois-je faire ?",
                 "Quels papiers pour une première carte d'identité ?"]),
    ("Passeport", "En Tunisie ou à l'étranger (consulat)",
     "passeport.png", ["Comment renouveler mon passeport ?",
                       "Combien coûte un duplicata de passeport à l'étranger ?"]),
    ("Services municipaux", "Extrait de naissance, attestation de résidence, légalisation de signature",
     "etat_civil.png", ["Comment obtenir un extrait de naissance ?",
                        "Comment légaliser une signature ?"]),
]

CSS = """
<style>
/* Fond : bandeau bleu marine en haut, gris clair en dessous */
.stApp { background: linear-gradient(180deg, #0b2447 0px, #19376d 230px, #f4f6fb 230px); }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1100px; padding-top: 2rem; }

/* En-tête */
.entete { text-align: center; color: #ffffff; padding: 10px 0 28px; }
.entete h1 { color: #ffffff; font-size: 2.3rem; margin: 0; }
.entete p { color: #dbe3f3; font-size: 1.05rem; margin: 6px 0 0; }
.entete .ligne { width: 90px; height: 5px; background: #c8102e; border-radius: 3px; margin: 14px auto 0; }

/* Cartes de la page d'accueil */
.carte { background: #ffffff; border-radius: 14px; overflow: hidden;
         box-shadow: 0 8px 24px rgba(11, 36, 71, .15); margin-bottom: 10px; }
.carte .visuel { height: 170px; display: flex; align-items: center; justify-content: center;
                 background: linear-gradient(135deg, #eef2fa, #d9e1f1); }
.carte .visuel svg { height: 135px; width: auto; }
.carte .visuel img { width: 100%; height: 100%; object-fit: cover; object-position: top; }
.carte .contenu { border-top: 4px solid #c8102e; padding: 12px 16px 14px; min-height: 100px; }
.carte h3 { color: #0b2447; font-size: 1.15rem; margin: 0; padding: 0; }
.carte p { color: #5b6479; font-size: .9rem; margin: 4px 0 0; }
</style>
"""


@st.cache_resource
def charger_ressources():
    """Charge le modèle, la base et les formulaires une seule fois."""
    return charger_modele(), ouvrir_collection(), charger_correspondances()


def css_photo_de_fond():
    """Si images/fond.jpg existe : photo en fond de page, assombrie par un voile bleu."""
    chemin = DOSSIER_IMAGES / "fond.jpg"
    if not chemin.exists():
        return ""
    code = base64.b64encode(chemin.read_bytes()).decode()
    return f"""<style>
.stApp {{ background:
    linear-gradient(rgba(11, 36, 71, .82), rgba(11, 36, 71, .62) 40%, rgba(11, 36, 71, .88)),
    url("data:image/jpeg;base64,{code}") center / cover fixed; }}
[data-testid="stChatMessage"] {{ background: #ffffff; border-radius: 12px;
    box-shadow: 0 6px 18px rgba(0, 0, 0, .18); }}
[data-testid="stBottom"] > div {{ background: transparent; }}
[data-testid="stSpinner"] p {{ color: #ffffff; }}
</style>"""


def choisir_question(question):
    """Appelée quand on clique sur une question d'exemple."""
    st.session_state.question_choisie = question


def afficher_carte(titre, description, image):
    """Affiche une carte (image + titre + description) en HTML."""
    chemin = DOSSIER_IMAGES / image
    if image.endswith(".svg"):
        visuel = chemin.read_text(encoding="utf-8")
    else:
        type_image = "jpeg" if image.endswith(".jpg") else "png"
        code = base64.b64encode(chemin.read_bytes()).decode()
        visuel = f'<img src="data:image/{type_image};base64,{code}">'
    st.markdown(f"""<div class="carte">
        <div class="visuel">{visuel}</div>
        <div class="contenu"><h3>{titre}</h3><p>{description}</p></div>
    </div>""", unsafe_allow_html=True)


@st.cache_data
def chemins_des_documents():
    """Associe chaque nom de PDF du corpus à son chemin (ex. CIN_renouvellement.pdf -> data/...)."""
    return {chemin.name: str(chemin) for chemin in DOSSIER_DATA.rglob("*.pdf")}


@st.cache_data
def pages_en_images(chemin):
    """Transforme chaque page d'un PDF en image PNG (une seule fois grâce au cache)."""
    with pymupdf.open(chemin) as document:
        return [page.get_pixmap(dpi=100).tobytes("png") for page in document]


def apercu_pdf(titre, chemin, cle, page_citee=None):
    """Un bloc dépliable : voir le PDF dans la page, puis le télécharger si on veut."""
    chemin = Path(chemin)
    with st.expander(titre):
        for numero, image in enumerate(pages_en_images(str(chemin)), start=1):
            legende = f"Page {numero}" + (" — page citée" if numero == page_citee else "")
            st.image(image, caption=legende)
        st.download_button("⬇️ Télécharger le PDF", data=chemin.read_bytes(),
                           file_name=chemin.name, mime="application/pdf", key=cle)


def mettre_en_forme(texte):
    """Transforme les puces « • » en liste Markdown (un point par ligne)."""
    return texte.replace("•", "\n- ")


def afficher_message(i, message):
    """Affiche un message du chat (question ou réponse avec formulaires et sources)."""
    avatar = "🧑" if message["role"] == "user" else "🏛️"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(mettre_en_forme(message["contenu"]))
        if message["role"] == "assistant":
            if message["formulaires"]:
                st.info("📝 Formulaire officiel pour cette démarche : ouvrez-le pour le voir, puis téléchargez-le pour l’imprimer et le remplir.")
            for fichier, nom in message["formulaires"]:
                chemin = DOSSIER_FORMULAIRES / fichier
                if chemin.exists():
                    apercu_pdf(f"📄 Voir le formulaire : {nom}", chemin, f"{i}_form_{fichier}")

            st.markdown("**📚 Sources consultées** (cliquez pour ouvrir le document)")
            documents = chemins_des_documents()
            deja_vus = set()
            for r in message["sources"]:
                if r["source"] in deja_vus or r["source"] not in documents:
                    continue
                deja_vus.add(r["source"])
                apercu_pdf(f"{r['source']} — page {r['page']} (score {r['score']:.3f})",
                           documents[r["source"]], f"{i}_src_{r['source']}", page_citee=r["page"])


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(css_photo_de_fond(), unsafe_allow_html=True)

if "historique" not in st.session_state:
    st.session_state.historique = []

# ----- Barre latérale -----
st.sidebar.header("Votre situation")
choix = st.sidebar.radio("Où faites-vous la démarche ?", list(PUBLICS.keys()))
public = PUBLICS[choix]
if st.sidebar.button("🔄 Nouvelle conversation"):
    st.session_state.historique = []
st.sidebar.caption("Les réponses viennent uniquement des documents du projet. "
                   "Vérifiez toujours auprès de l'administration concernée.")

# ----- En-tête -----
st.markdown("""<div class="entete">
    <h1>🏛️ Assistant des démarches administratives</h1>
    <p>Carte d'identité, passeport, services municipaux : posez votre question,
       la réponse cite les documents officiels.</p>
    <div class="ligne"></div>
</div>""", unsafe_allow_html=True)

with st.spinner("Chargement du modèle et des documents (première fois : environ 30 s)..."):
    modele, collection, correspondances = charger_ressources()

# ----- Page d'accueil : les 3 cartes (seulement si la conversation est vide) -----
if not st.session_state.historique:
    colonnes = st.columns(3)
    for colonne, (titre, description, image, exemples) in zip(colonnes, CARTES):
        with colonne:
            afficher_carte(titre, description, image)
            for exemple in exemples:
                st.button(exemple, key=exemple, on_click=choisir_question,
                          args=(exemple,), width="stretch")

# ----- Conversation -----
for i, message in enumerate(st.session_state.historique):
    afficher_message(i, message)

question = st.chat_input("Exemple : J'ai perdu ma carte d'identité, que dois-je faire ?")
if not question:
    question = st.session_state.pop("question_choisie", None)

if question:
    message_user = {"role": "user", "contenu": question}
    st.session_state.historique.append(message_user)
    afficher_message(len(st.session_state.historique) - 1, message_user)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("Recherche dans les documents..."):
            resultats, contexte = preparer(question, collection, modele, public=public)
        reponse = st.write_stream(generer_reponse_flux(question, contexte))
    formulaires = trouver_formulaires(reponse, resultats, correspondances)

    message_assistant = {"role": "assistant", "contenu": reponse,
                         "sources": resultats, "formulaires": formulaires}
    st.session_state.historique.append(message_assistant)
    st.rerun()