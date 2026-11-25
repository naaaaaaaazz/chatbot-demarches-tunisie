"""
Étape 7b : interface web du chatbot (Streamlit).
Lancer depuis la racine du projet avec : streamlit run src/app.py
"""
import base64               # pour mettre une image directement dans le HTML (texte base64)
import html                 # html.escape() : rend inoffensif un texte avant de l'afficher en HTML
import re                   # expressions régulières (citations, adresses web)
from pathlib import Path    # manipuler les chemins de fichiers
from urllib.parse import urlparse   # découper une adresse web (pour garder seulement le nom du site)

import pymupdf              # lire les PDF : pages -> images, liens écrits dans les PDF
import streamlit as st      # la bibliothèque de l'interface web

# Nos propres fichiers : le modèle e5, la chaîne RAG et le choix des formulaires
from embeddings import charger_modele
from generation import ouvrir_collection, preparer, generer_reponse_flux, nettoyer_citations
from choix_formulaire import charger_correspondances, trouver_formulaires, DOSSIER_FORMULAIRES

# Réglages de l'onglet du navigateur : titre, icône, page en pleine largeur.
# Doit être le premier appel Streamlit du script.
st.set_page_config(page_title="Assistant démarches administratives",
                   page_icon="🏛️", layout="wide")

DOSSIER_PROJET = Path(__file__).resolve().parent.parent   # src/app.py -> dossier du projet
DOSSIER_IMAGES = DOSSIER_PROJET / "images"
DOSSIER_DATA = DOSSIER_PROJET / "data"

# Choix affiché dans la barre latérale -> valeur du filtre « public » envoyée à la recherche
# (None = pas de filtre, la zone est devinée à partir de la question)
PUBLICS = {"Je ne sais pas": None,
           "En Tunisie": "tunisie",
           "À l'étranger (consulat)": "etranger"}

# Les 3 cartes de la page d'accueil : titre, description, image, questions d'exemple
# Chaque carte est un tuple de 4 éléments ; on les « déballe » plus bas avec un for
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

# Le style de la page (couleurs, cartes, bulles...) écrit en CSS.
# Les commentaires /* ... */ à l'intérieur sont des commentaires CSS.
# [data-testid="..."] = on vise les éléments que Streamlit crée lui-même (messages, barre du bas...)
CSS = """
<style>
/* Fond : bandeau bleu marine en haut, gris clair en dessous */
.stApp { background: linear-gradient(180deg, #0b2447 0px, #19376d 330px, #f4f6fb 330px); }
header[data-testid="stHeader"] { background: transparent; }
.stAppDeployButton { display: none; }
.block-container { max-width: 1100px; padding-top: 1.5rem; }

/* En-tête */
.entete { text-align: center; color: #ffffff; padding: 6px 0 34px; }
.entete .logo { width: 74px; height: 74px; margin: 0 auto 12px; border-radius: 50%;
    background: #ffffff; display: flex; align-items: center; justify-content: center;
    font-size: 2.3rem; box-shadow: 0 0 0 6px rgba(255, 255, 255, .15); }
.entete .etiquette { display: inline-block; padding: 3px 14px; border-radius: 999px;
    background: rgba(255, 255, 255, .12); border: 1px solid rgba(255, 255, 255, .25);
    color: #dbe3f3; font-size: .75rem; font-weight: 600; letter-spacing: .12em; }
.entete h1 { font-family: Georgia, serif; color: #ffffff; font-size: 2.5rem;
    margin: 10px 0 0; padding: 0; }
.entete p { color: #dbe3f3; font-size: 1.05rem; margin: 6px 0 0; }
.entete .ligne { width: 90px; height: 5px; background: #c8102e; border-radius: 3px; margin: 16px auto 18px; }
.entete .pastilles { display: flex; gap: 10px; justify-content: center; flex-wrap: wrap; }
.entete .pastilles span { background: #ffffff; color: #0b2447; border-radius: 999px;
    padding: 6px 14px; font-size: .85rem; font-weight: 600;
    box-shadow: 0 4px 12px rgba(0, 0, 0, .15); }

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

/* Question de l'utilisateur : bulle bleue à droite */
[data-testid="stChatMessage"]:has(.bulle) { background: transparent !important;
    box-shadow: none !important; flex-direction: row-reverse; }
.bulle { margin-left: auto; width: fit-content; max-width: 75%; background: #1f3a93; color: #ffffff;
    padding: 10px 18px; border-radius: 18px 18px 4px 18px; box-shadow: 0 4px 12px rgba(11, 36, 71, .25); }

/* Réponses du chatbot : titre de la démarche, rubriques, cases cochées */
[data-testid="stChatMessage"] h2 { font-family: Georgia, serif; color: #0b2447; font-size: 1.45rem;
    background: #ffffff; border-left: 6px solid #c8102e; border-radius: 10px;
    padding: 14px 18px; margin: 14px 0 6px; box-shadow: 0 4px 14px rgba(11, 36, 71, .10); }
[data-testid="stChatMessage"] h3 { color: #19376d; font-size: .95rem; text-transform: uppercase;
    letter-spacing: .05em; margin: 20px 0 6px; padding-bottom: 6px; border-bottom: 2px solid #e3e8f2; }
[data-testid="stChatMessage"] ul { list-style: none; padding-left: 0; }
[data-testid="stChatMessage"] li { position: relative; margin: 0; padding: 9px 0 9px 34px;
    border-bottom: 1px dashed #e3e8f2; }
[data-testid="stChatMessage"] li::before { content: "✓"; position: absolute; left: 4px; top: 9px;
    width: 20px; height: 20px; border: 2px solid #19376d; border-radius: 5px; color: #19376d;
    font-weight: 700; font-size: .8rem; display: flex; align-items: center; justify-content: center; }

/* Cartes des formulaires */
.titre-rubrique { color: #19376d; font-size: .95rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; margin: 18px 0 4px; }
[class*="st-key-formulaire_"] { background: #f7f9fc; border-radius: 12px; }
.icone-pdf { width: 42px; height: 42px; border-radius: 10px; background: #e8eefb;
    display: flex; align-items: center; justify-content: center; font-size: 1.4rem; }
.nom-formulaire { color: #0b2447; font-weight: 600; font-size: 1.02rem; }
.sous-titre { color: #6b7489; font-size: .85rem; }

/* Encadré des sites officiels */
.sites { background: #eef3fb; border: 1px solid #c5d3ef; border-radius: 12px;
    padding: 12px 18px; margin: 14px 0; }
.sites .titre-sites { color: #19376d; font-size: .9rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .05em; margin: 0 0 6px; }
.sites a { display: block; color: #1f3a93; text-decoration: none; padding: 3px 0;
    border-bottom: 1px dotted #9fb3dc; width: fit-content; margin-bottom: 4px; }
.sites a:hover { color: #c8102e; }

/* Réponses dans une carte blanche, zone de saisie alignée sur le contenu */
[data-testid="stChatMessage"] { background: #ffffff; border-radius: 14px;
    box-shadow: 0 6px 20px rgba(11, 36, 71, .10); padding: 18px 22px; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { font-size: 1rem; line-height: 1.6; }
[data-testid="stBottomBlockContainer"] { max-width: 1100px; margin: 0 auto; }
[data-testid="stBottom"] > div { background: transparent; }

/* Numéro de source dans le texte (survoler = voir le fichier) */
.ref { display: inline-flex; align-items: center; justify-content: center; width: 18px; height: 18px;
    margin-left: 4px; border-radius: 50%; background: #19376d; color: #ffffff;
    font-size: .68rem; font-weight: 700; vertical-align: super; cursor: help; }
.refs { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 14px; padding-top: 10px;
    border-top: 1px solid #e3e8f2; }
.refs .source { margin-left: 0; }
.refs .source b { background: #19376d; color: #ffffff; border-radius: 50%; padding: 0 5px; margin-right: 3px; }

/* Étiquette de source après chaque information */
.source { display: inline-block; margin-left: 6px; padding: 1px 9px; border-radius: 999px;
    background: #e8eefb; color: #19376d; border: 1px solid #c5d3ef;
    font-size: .75rem; font-weight: 600; white-space: nowrap; vertical-align: middle; }
</style>
"""


# @st.cache_resource : la fonction n'est exécutée QU'UNE FOIS ; ensuite Streamlit garde le résultat.
# Important car Streamlit relance tout le script à chaque clic : sans cache,
# on rechargerait le modèle e5 (environ 30 s) à chaque question.
@st.cache_resource
def charger_ressources():
    """Charge le modèle, la base et les formulaires une seule fois."""
    return charger_modele(), ouvrir_collection(), charger_correspondances()


def css_photo_de_fond():
    """Si images/fond.jpg existe : photo sur tout le fond de la page, assombrie par un voile bleu."""
    chemin = DOSSIER_IMAGES / "fond.jpg"
    if not chemin.exists():
        return ""                       # pas de photo : on garde le fond bleu/gris du CSS
    # L'image est transformée en texte base64 pour être mise directement dans le CSS
    code = base64.b64encode(chemin.read_bytes()).decode()
    # f-string : les {{ }} doubles donnent de vraies accolades CSS ; {code} insère l'image.
    # Le dégradé rgba(...) est un voile bleu semi-transparent posé SUR la photo (texte lisible).
    return f"""<style>
.stApp {{ background:
    linear-gradient(rgba(11, 36, 71, .82), rgba(11, 36, 71, .62) 40%, rgba(11, 36, 71, .88)),
    url("data:image/jpeg;base64,{code}") center / cover fixed; }}
[data-testid="stSpinner"] p {{ color: #ffffff; }}
</style>"""


def choisir_question(question):
    """Appelée quand on clique sur une question d'exemple."""
    # session_state = la mémoire de la page entre deux relances du script.
    # On y range la question cliquée ; elle sera lue plus bas comme si on l'avait tapée.
    st.session_state.question_choisie = question


def afficher_carte(titre, description, image):
    """Affiche une carte (image + titre + description) en HTML."""
    chemin = DOSSIER_IMAGES / image
    if image.endswith(".svg"):
        visuel = chemin.read_text(encoding="utf-8")      # un SVG est du texte : on l'insère tel quel
    else:
        type_image = "jpeg" if image.endswith(".jpg") else "png"
        code = base64.b64encode(chemin.read_bytes()).decode()
        visuel = f'<img src="data:image/{type_image};base64,{code}">'   # image intégrée au HTML
    # unsafe_allow_html=True : autorise Streamlit à afficher notre HTML (sinon il l'affiche en texte)
    st.markdown(f"""<div class="carte">
        <div class="visuel">{visuel}</div>
        <div class="contenu"><h3>{titre}</h3><p>{description}</p></div>
    </div>""", unsafe_allow_html=True)


# Un dictionnaire gardé en cache : même question + même zone -> réponse immédiate, sans LLM
@st.cache_resource
def memoire_des_reponses():
    """Mémoire partagée : question déjà posée -> (réponse, sources). Vidée quand on relance Streamlit."""
    return {}


# @st.cache_data : garde le résultat pour les mêmes arguments (ici, calculé une seule fois)
@st.cache_data
def chemins_des_documents():
    """Associe chaque nom de PDF du corpus à son chemin (ex. CIN_renouvellement.pdf -> data/...)."""
    # rglob("*.pdf") : cherche les PDF dans data/ ET dans tous ses sous-dossiers
    # dictionnaire en compréhension : { nom_du_fichier: chemin }
    return {chemin.name: str(chemin) for chemin in DOSSIER_DATA.rglob("*.pdf")}


@st.cache_data
def pages_en_images(chemin):
    """Transforme chaque page d'un PDF en image PNG (une seule fois grâce au cache)."""
    # « with » ferme le PDF automatiquement à la fin
    with pymupdf.open(chemin) as document:
        # get_pixmap(dpi=100) = « photo » de la page ; tobytes("png") = l'image en PNG
        return [page.get_pixmap(dpi=100).tobytes("png") for page in document]


def apercu_pdf(titre, chemin, cle, page_citee=None):
    """Un bloc dépliable : voir le PDF dans la page, puis le télécharger si on veut."""
    chemin = Path(chemin)
    with st.expander(titre):                 # expander = bloc qu'on ouvre/ferme en cliquant
        # enumerate(..., start=1) : numérote les pages à partir de 1
        for numero, image in enumerate(pages_en_images(str(chemin)), start=1):
            # on ajoute « — page citée » sous la page utilisée pour la réponse
            legende = f"Page {numero}" + (" — page citée" if numero == page_citee else "")
            st.image(image, caption=legende)
        # key = identifiant unique : Streamlit refuse deux boutons avec la même clé
        st.download_button("⬇️ Télécharger le PDF", data=chemin.read_bytes(),
                           file_name=chemin.name, mime="application/pdf", key=cle)


def basculer(cle):
    """Bouton « Ouvrir / Fermer » : inverse l'état ouvert/fermé d'un formulaire."""
    # .get(cle, False) : False si la clé n'existe pas encore ; « not » inverse True <-> False
    st.session_state[cle] = not st.session_state.get(cle, False)


def carte_formulaire(nom, chemin, cle):
    """Une carte par formulaire : icône, nom, bouton Télécharger et bouton Ouvrir."""
    chemin = Path(chemin)
    pages = pages_en_images(str(chemin))
    cle_ouvert = f"ouvert_{cle}"                       # où on retient si ce formulaire est ouvert
    ouvert = st.session_state.get(cle_ouvert, False)
    # container(border=True) = un cadre ; key -> classe CSS « st-key-formulaire_... » (fond gris clair)
    with st.container(border=True, key=f"formulaire_{cle}"):
        # 4 colonnes, largeurs relatives 0.6 / 5 / 1.6 / 1.6, centrées verticalement
        icone, texte, telecharger, ouvrir = st.columns([0.6, 5, 1.6, 1.6],
                                                        vertical_alignment="center")
        icone.markdown('<div class="icone-pdf">📄</div>', unsafe_allow_html=True)
        # html.escape(nom) : le nom vient du CSV, on l'échappe avant de le mettre dans du HTML
        texte.markdown(f'<div class="nom-formulaire">{html.escape(nom)}</div>'
                       f'<div class="sous-titre">Formulaire (PDF) · {len(pages)} page(s)</div>',
                       unsafe_allow_html=True)
        # type="primary" = bouton coloré ; width="stretch" = toute la largeur de la colonne
        telecharger.download_button("Télécharger", data=chemin.read_bytes(),
                                    file_name=chemin.name, mime="application/pdf",
                                    key=f"tel_{cle}", type="primary", width="stretch")
        # Le texte du bouton change selon l'état ; on_click appelle basculer(cle_ouvert)
        ouvrir.button("Fermer" if ouvert else "Ouvrir", key=f"btn_{cle}",
                      on_click=basculer, args=(cle_ouvert,), width="stretch")
        if ouvert:                                 # aperçu des pages dans la page
            for numero, image in enumerate(pages, start=1):
                st.image(image, caption=f"Page {numero}")


@st.cache_data
def liens_du_pdf(chemin):
    """Trouve les adresses web écrites dans un PDF (liens cliquables + adresses dans le texte)."""
    liens = []
    with pymupdf.open(chemin) as document:
        for page in document:
            # 1) les liens cliquables du PDF (get_links) qui ont une adresse « uri »
            adresses = [lien["uri"] for lien in page.get_links() if lien.get("uri")]
            # 2) les adresses écrites dans le texte : commencent par http(s):// ou www.,
            #    puis tous les caractères qui ne sont ni espace, ni guillemet, ni parenthèse, ni crochet
            adresses += re.findall(r"(?:https?://|www\.)[^\s<>\"'()\[\]]+", page.get_text())
            for adresse in adresses:
                adresse = adresse.rstrip(".,;:")       # enlève la ponctuation collée à la fin
                if not adresse.startswith("http"):
                    adresse = "https://" + adresse     # « www.site.tn » -> « https://www.site.tn »
                if adresse not in liens:               # pas de doublon
                    liens.append(adresse)
    return liens


def pdf_utilises(sources, reponse):
    """Les PDF cités dans la réponse ; si aucun n'est cité, le meilleur PDF trouvé."""
    # on garde les sources dont le nom apparaît dans le texte de la réponse
    cites = [r["source"] for r in sources if r["source"] in reponse]
    if not cites and sources:
        cites = [sources[0]["source"]]                # sinon : le 1er résultat de la recherche
    # dict.fromkeys : un dictionnaire n'a pas de clés en double -> enlève les doublons
    return list(dict.fromkeys(cites))                 # enlève les doublons, garde l'ordre


def encadre_sites(sources, reponse):
    """Encadré « Sites officiels » : les liens écrits dans les PDF utilisés pour la réponse."""
    if "je ne trouve pas" in reponse.lower():
        return                                        # refus : pas d'encadré
    documents = chemins_des_documents()
    liens = []
    for nom in pdf_utilises(sources, reponse):
        if nom in documents:
            for url in liens_du_pdf(documents[nom]):
                if url not in liens:
                    liens.append(url)
    if liens:
        # Un lien <a> par site (4 au maximum). urlparse(url).netloc = le nom du site
        # (ex. « www.idaraty.tn ») ; removeprefix("www.") -> « idaraty.tn ».
        # target="_blank" ouvre le site dans un nouvel onglet.
        html_liens = "".join(
            f'<a href="{html.escape(url)}" target="_blank">'
            f'{html.escape(urlparse(url).netloc.removeprefix("www."))} ↗</a>'
            for url in liens[:4])
        st.markdown('<div class="sites"><p class="titre-sites">🌐 Sites officiels (sources des documents)</p>'
                    + html_liens + "</div>", unsafe_allow_html=True)


def mettre_en_forme(texte):
    """Puces « • » -> liste Markdown. Citations [fichier.pdf, p. X] -> petits numéros ①
    dans le texte, et la liste des références (numéro -> fichier, page) à la fin."""
    texte = texte.replace("<", "&lt;")             # sécurité : pas de HTML venant du LLM
    texte = texte.replace("•", "\n- ")             # « • » -> vraie puce Markdown sur une nouvelle ligne
    numeros = {}                                   # (fichier, page) -> numéro

    # Fonction appelée par re.sub pour CHAQUE [ ... .pdf ... ] trouvé dans le texte
    def remplacer(trouve):
        # Une citation peut contenir plusieurs sources : [a.pdf, p. 1 ; b.pdf, p. 2]
        # On récupère chaque couple (nom du fichier, numéro de page)
        refs = re.findall(r"([\w\-. ]+?\.pdf)\s*,\s*p\.?\s*(\d+)", trouve.group(1))
        if not refs:                               # ce n'est pas une citation : on ne touche à rien
            return trouve.group(0)
        sortie = ""
        for nom, page in refs:
            cle = (nom.strip(), page)
            # nouvelle source -> numéro suivant ; source déjà vue -> on garde son numéro
            numeros.setdefault(cle, len(numeros) + 1)
            # petit rond numéroté ; title = texte affiché au survol de la souris
            sortie += f'<span class="ref" title="{cle[0]} · p. {page}">{numeros[cle]}</span>'
        return sortie

    texte = re.sub(r"\[([^\[\]]*\.pdf[^\[\]]*)\]", remplacer, texte)
    if numeros:
        # À la fin : la liste des références « 1 📄 fichier.pdf · p. X »
        texte += '\n\n<div class="refs">' + "".join(
            f'<span class="source"><b>{n}</b> 📄 {nom} · p. {page}</span>'
            for (nom, page), n in numeros.items()) + "</div>"
    return texte


def afficher_message(i, message):
    """Affiche un message du chat (question ou réponse avec formulaires et sources)."""
    # i = numéro du message : sert à créer des clés uniques pour les boutons
    avatar = "🧑" if message["role"] == "user" else "🏛️"
    with st.chat_message(message["role"], avatar=avatar):
        if message["role"] == "user":
            # la question est échappée (html.escape) : l'utilisateur ne peut pas injecter de HTML
            st.markdown(f'<div class="bulle">{html.escape(message["contenu"])}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(mettre_en_forme(message["contenu"]), unsafe_allow_html=True)
        if message["role"] == "assistant":
            if "je ne trouve pas" in message["contenu"].lower():
                # Information absente des documents : pas de formulaires, sites ni sources
                st.caption("Aucun document du projet ne contient cette information. "
                           "Le chatbot couvre la carte d'identité, le passeport et les services municipaux.")
                return                                 # on s'arrête ici
            # 1) Les formulaires à remplir
            if message["formulaires"]:
                st.markdown('<p class="titre-rubrique">📝 Formulaires à remplir</p>',
                            unsafe_allow_html=True)
            for fichier, nom in message["formulaires"]:
                chemin = DOSSIER_FORMULAIRES / fichier
                if chemin.exists():                    # on n'affiche que les PDF vraiment présents
                    carte_formulaire(nom, chemin, f"{i}_form_{fichier}")

            # 2) Les sites officiels, puis 3) les sources consultées (les k extraits trouvés)
            encadre_sites(message["sources"], message["contenu"])
            st.markdown("**📚 Sources consultées** (cliquez pour ouvrir le document)")
            documents = chemins_des_documents()
            deja_vus = set()                           # un même PDF n'est affiché qu'une fois
            for r in message["sources"]:
                if r["source"] in deja_vus or r["source"] not in documents:
                    continue
                deja_vus.add(r["source"])
                # :.3f = score avec 3 chiffres après la virgule
                apercu_pdf(f"{r['source']} — page {r['page']} (score {r['score']:.3f})",
                           documents[r["source"]], f"{i}_src_{r['source']}", page_citee=r["page"])


# ======================================================================
# Programme principal. ATTENTION : Streamlit relance TOUT ce script
# de haut en bas à chaque action (question, clic sur un bouton...).
# ======================================================================
st.markdown(CSS, unsafe_allow_html=True)               # on applique le style
st.markdown(css_photo_de_fond(), unsafe_allow_html=True)

# L'historique de la conversation est gardé dans session_state (sinon il serait perdu à chaque relance)
if "historique" not in st.session_state:
    st.session_state.historique = []

# ----- Barre latérale -----
st.sidebar.header("Votre situation")
choix = st.sidebar.radio("Où faites-vous la démarche ?", list(PUBLICS.keys()))
public = PUBLICS[choix]                                # ex. « En Tunisie » -> "tunisie"
if st.sidebar.button("🔄 Nouvelle conversation"):
    st.session_state.historique = []                   # on vide l'historique
st.sidebar.caption("Les réponses viennent uniquement des documents du projet. "
                   "Vérifiez toujours auprès de l'administration concernée.")

# ----- En-tête -----
st.markdown("""<div class="entete">
    <div class="logo">🏛️</div>
    <span class="etiquette">ASSISTANT IA · RAG</span>
    <h1>Assistant des démarches administratives</h1>
    <p>Carte d'identité, passeport, services municipaux : posez votre question,
       la réponse cite les documents officiels.</p>
    <div class="ligne"></div>
    <div class="pastilles">
        <span>📚 Réponses sourcées</span>
        <span>⚡ En quelques secondes</span>
        <span>🔒 Documents du projet uniquement</span>
    </div>
</div>""", unsafe_allow_html=True)

# spinner = message d'attente avec une petite roue pendant le chargement
with st.spinner("Chargement du modèle et des documents (première fois : environ 30 s)..."):
    modele, collection, correspondances = charger_ressources()

# ----- Page d'accueil : les 3 cartes (seulement si la conversation est vide) -----
if not st.session_state.historique:
    colonnes = st.columns(3)                           # 3 colonnes côte à côte
    # zip associe chaque colonne à une carte ; on « déballe » les 4 éléments de la carte
    for colonne, (titre, description, image, exemples) in zip(colonnes, CARTES):
        with colonne:
            afficher_carte(titre, description, image)
            for exemple in exemples:
                # un bouton par question d'exemple ; on_click range la question dans session_state
                st.button(exemple, key=exemple, on_click=choisir_question,
                          args=(exemple,), width="stretch")

# ----- Conversation -----
# On réaffiche tous les messages déjà échangés (le script a été relancé)
for i, message in enumerate(st.session_state.historique):
    afficher_message(i, message)

# La zone de saisie en bas de la page (renvoie None tant qu'on n'a rien envoyé)
question = st.chat_input("Exemple : J'ai perdu ma carte d'identité, que dois-je faire ?")
if not question:
    # pas de question tapée : peut-être une question d'exemple cliquée.
    # pop() la lit ET l'enlève (sinon elle serait reposée à chaque relance)
    question = st.session_state.pop("question_choisie", None)

if question:
    # 1) On ajoute la question à l'historique et on l'affiche
    message_user = {"role": "user", "contenu": question}
    st.session_state.historique.append(message_user)
    afficher_message(len(st.session_state.historique) - 1, message_user)

    # 2) La réponse : depuis la mémoire si la question est déjà connue, sinon RAG complet
    memoire = memoire_des_reponses()
    cle = (question.strip().lower(), public)           # même question (sans majuscules) + même zone
    with st.chat_message("assistant", avatar="🏛️"):
        if cle in memoire:                       # déjà posée : réponse immédiate
            reponse, resultats = memoire[cle]
        else:                                    # nouvelle question : RAG complet
            with st.spinner("Recherche dans les documents..."):
                # R + A : recherche des extraits et construction du contexte
                resultats, contexte = preparer(question, collection, modele, public=public)
            # G : le texte s'affiche au fur et à mesure ; write_stream renvoie ensuite le texte complet
            reponse = st.write_stream(generer_reponse_flux(question, contexte))
            reponse = nettoyer_citations(reponse)     # même format de citation partout
            memoire[cle] = (reponse, resultats)      # on la garde pour la prochaine fois
    # 3) Les formulaires à proposer (règles de choix_formulaire.py)
    formulaires = trouver_formulaires(reponse, resultats, correspondances)

    # 4) On range la réponse dans l'historique, puis on relance le script :
    #    afficher_message la réaffiche avec la mise en forme, les formulaires et les sources
    message_assistant = {"role": "assistant", "contenu": reponse,
                         "sources": resultats, "formulaires": formulaires}
    st.session_state.historique.append(message_assistant)
    st.rerun()