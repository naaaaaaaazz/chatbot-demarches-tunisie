# 🏛️ Assistant des démarches administratives tunisiennes

Chatbot **RAG** (*Retrieval-Augmented Generation*) qui répond aux questions des citoyens sur la **carte d'identité nationale**, le **passeport** (en Tunisie et à l'étranger) et les **services municipaux**, uniquement à partir de documents officiels.
Chaque réponse **cite ses sources** sous la forme `[document, page]`, et le chatbot **propose le formulaire officiel** à télécharger quand la démarche en demande un.

> Mini-projet – Sujet 6 : *Chatbot intelligent basé sur les documents administratifs*.

---

## ✨ Fonctionnalités

- Questions en langage naturel, avec une **interface de conversation** (Streamlit).
- Réponses **fondées uniquement sur les documents**, avec une citation `[fichier, p. X]` après chaque information.
- **« Je ne trouve pas cette information dans les documents »** pour les questions hors sujet.
- **Filtre selon la situation** : en Tunisie (police, municipalité) ou à l'étranger (consulat). Ce filtre est choisi par l'utilisateur ou détecté automatiquement dans la question.
- **Bouton de téléchargement du formulaire** correspondant à la démarche (CIN, passeport, autorisation parentale).
- **100 % local et gratuit** : le modèle de langage tourne sur l'ordinateur avec Ollama. Aucune clé API, aucune donnée envoyée sur internet.

---

## 🧠 Fonctionnement (le pipeline RAG)

```
INDEXATION (une seule fois)
PDF ──► extraction ──► nettoyage ──► découpage ──► embeddings ──► ChromaDB
        (pymupdf)      (regex)       (600 car.)    (e5-base)      (base vectorielle)

QUESTION → RÉPONSE
question ──► embedding ──► recherche des 5 meilleurs extraits ──► prompt + extraits ──► LLM (Ollama) ──► réponse citée
                           (filtre Tunisie/étranger)                                    qwen2.5:3b      + formulaire
```

| Étape | Fichier | Rôle |
|---|---|---|
| 1 | `src/extraction.py` | Extrait le texte de chaque page des PDF (avec le nom du fichier et le numéro de page) |
| 2 | `src/nettoyage.py` | Supprime les en-têtes, les pieds de page et les espaces inutiles |
| 3 | `src/decoupage.py` | Découpe en chunks de 600 caractères (chevauchement de 80) et ajoute les métadonnées : source, page, type, public |
| 4 | `src/embeddings.py` | Transforme chaque chunk en vecteur de 768 nombres (`intfloat/multilingual-e5-base`) |
| 5 | `src/indexation.py` | Stocke les vecteurs dans ChromaDB et fait la recherche (avec les améliorations ci-dessous) |
| 6 | `src/generation.py` | Construit le prompt (règles + extraits) et fait rédiger la réponse par `qwen2.5:3b` via Ollama |
| 7 | `src/choix_formulaire.py` | Choisit le formulaire à proposer, à partir des fiches citées dans la réponse |
| 7 | `src/app.py` | Interface web Streamlit (conversation, formulaires, sources) |
| 8 | `src/evaluation_*.py`, `src/comparer_chunks.py` | Évaluation de la recherche et des réponses |

### Améliorations de la recherche
- **Expansion de requête** : « cin » devient « carte d'identité nationale (CIN) », et « passport » devient « passeport ».
- **Au plus 2 extraits par document** : un document généraliste (le guide) ne peut plus prendre toutes les places.
- **Priorité aux fiches de procédure** (bonus de +0,01 au score).
- **Détection de la zone** dans la question (« en Tunisie », « consulat », « France »…).
- **Small-to-big** : on cherche avec des petits chunks, mais le LLM reçoit la **page entière** de la meilleure fiche, pour que la liste des papiers ne soit jamais coupée.

---

## 📁 Structure du projet

```
documents_admin_tunisie/
├── data/
│   ├── procedures/        14 fiches pratiques (CIN, Passeport, Municipalite)
│   ├── reglements/        5 lois (57-3, 75-40, 2015-46, 2024-22, 2024-23)
│   ├── guides/            1 guide d'orientation du citoyen
│   └── processed/         fichiers générés (pages, chunks, vecteurs) – non versionnés
├── formulaires/           formulaires officiels (PDF) + formulaires.csv (fiche → formulaire)
├── images/                illustrations de l'interface
├── evaluation/            questions de test et résultats de l'évaluation
├── src/                   tout le code Python
├── .streamlit/config.toml thème de l'interface
├── SOURCES.md             origine de chaque document (URL + date de consultation)
├── requirements.txt
└── README.md
```

**Corpus** : 20 PDF (14 procédures, 5 lois, 1 guide), 44 pages, 232 chunks. Tous les documents viennent de sources officielles (Idaraty, ministère de l'Intérieur, JORT…) : voir [`SOURCES.md`](SOURCES.md).

---

## ⚙️ Installation (Windows)

**Prérequis** : Python 3.12 ou plus récent, et [Ollama](https://ollama.com).

```powershell
# 1. Récupérer le projet
git clone [lien de votre dépôt GitHub]
cd documents_admin_tunisie

# 2. Créer et activer l'environnement virtuel
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Installer les bibliothèques
pip install -r requirements.txt

# 4. Télécharger le modèle de langage (≈ 2 Go)
ollama pull qwen2.5:3b
```

Au premier lancement, le modèle d'embeddings `multilingual-e5-base` (≈ 1 Go) est téléchargé automatiquement.

---

## ▶️ Utilisation

```powershell
# 1. Construire les chunks et les vecteurs (≈ 3 à 4 minutes, une seule fois)
python src/embeddings.py

# 2. Créer la base vectorielle ChromaDB (quelques secondes)
python src/indexation.py

# 3. Lancer le chatbot
streamlit run src/app.py
```

Le navigateur s'ouvre sur **http://localhost:8501**. Ollama doit être lancé.

### Évaluation
```powershell
python src/evaluation_recherche.py   # rappel@3, rappel@5, MRR (recherche simple vs améliorée)
python src/comparer_chunks.py        # comparaison des chunks de 300 et 600 caractères
python src/evaluation_reponses.py    # génère les réponses des 35 questions, à noter à la main
```

---

## 📊 Résultats

Jeu de test : **30 questions** dans le sujet et **5 questions hors sujet** (`evaluation/questions_test.csv`).

### Recherche (30 questions)
| Méthode | Rappel@3 | Rappel@5 | MRR |
|---|---|---|---|
| Recherche simple | 0,90 | 0,93 | 0,86 |
| + priorité aux fiches, 2 extraits max par document, expansion de requête | 0,90 | 0,97 | 0,84 |
| **+ détection de la zone (version finale)** | **0,93** | **1,00** | **0,87** |

### Taille des chunks
| Taille | Nombre de chunks | Rappel@3 | Rappel@5 | MRR |
|---|---|---|---|---|
| 300 caractères | 471 | 0,90 | 0,97 | 0,84 |
| **600 caractères (retenue)** | 232 | **0,93** | **1,00** | **0,87** |

**Choix de k** : avec k=3, le bon document est envoyé au LLM dans 93 % des cas, contre **100 % avec k=5**. On a donc retenu **k=5**.

### Réponses du LLM (notation manuelle, de 0 à 2)
| Mesure | Résultat |
|---|---|
| Exactitude moyenne (30 questions) | **1,53 / 2** (20 réponses à 2, 6 à 1, 4 à 0) |
| Fidélité moyenne (30 questions) | **1,83 / 2** (25 à 2, 5 à 1, aucune à 0) |
| Refus corrects (5 questions hors sujet) | **5 / 5** |
| Temps moyen par réponse | **114 s** (de 38 à 282 s) |

**Ce que montre l'évaluation :**
- les questions **précises** (prix, délais, nombre de photos, case du formulaire) obtiennent presque toutes 2/2 ;
- le chatbot **n'invente pas** (aucune fidélité à 0) et **refuse toujours** le hors sujet ;
- ses erreurs viennent du **mélange de deux démarches** (passeport et CIN), de **réponses incomplètes**, et de **deux refus à tort** sur des textes de loi pourtant bien retrouvés (Q22, Q23).

**Bug découvert grâce à l'évaluation** : le formulaire CIN était proposé pour des questions sans rapport (extrait de naissance, légalisation, permis de conduire…). Le choix du formulaire a été corrigé : plus de formulaire après un refus, et le plan B ne regarde que le 1ᵉʳ résultat. Une simulation sur les 35 réponses fait passer les formulaires proposés à tort de **9 à 1**.

---

## ⚠️ Limites

- **Lenteur** : sur un PC sans carte graphique, la recherche prend environ 0,3 s, mais la génération prend 2 à 4 minutes par réponse.
- **Petit modèle (3 milliards de paramètres)** : pour les questions ouvertes, les réponses sont parfois trop longues ou coupées, et un élément d'une longue liste est parfois oublié.
- **Sensibilité à la formulation** : les fautes de frappe et les abréviations peuvent gêner la recherche. Elles sont en partie corrigées par l'expansion de requête.
- **Hors sujet** : les scores des questions hors sujet (0,81 à 0,84) sont proches de ceux des vraies questions (≈ 0,85). C'est donc le LLM, et non un seuil, qui décide de refuser.
- Les documents ont été consultés en **novembre 2026** : les démarches et les tarifs peuvent changer.

## 🔭 Pistes d'amélioration
Un modèle plus grand (`qwen2.5:7b`) ou un GPU, une recherche hybride (BM25 + embeddings), un reranking, et l'affichage de la réponse mot par mot (streaming).

---

## 🔒 Sources et données personnelles
- L'origine de chaque document est dans [`SOURCES.md`](SOURCES.md).
- Les exemples de documents montrés dans l'interface (CIN, extrait de naissance) ont été **floutés** pour masquer toutes les données personnelles.
- ⚠️ Cet assistant aide à s'orienter, mais il ne remplace pas l'administration. Vérifiez toujours auprès du service concerné.

## 🛠️ Technologies
Python · PyMuPDF · LangChain Text Splitters · Sentence-Transformers (`intfloat/multilingual-e5-base`) · ChromaDB · Ollama (`qwen2.5:3b`) · Streamlit