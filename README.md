# 🏛️ Assistant des démarches administratives tunisiennes

Chatbot **RAG** (*Retrieval-Augmented Generation*) qui répond aux questions des citoyens sur la **carte d'identité nationale**, le **passeport** (en Tunisie et à l'étranger) et les **services municipaux**, uniquement à partir de documents officiels.
Chaque réponse **cite ses sources** (`[document, page]`), propose le **formulaire officiel** quand la démarche en demande un, et renvoie vers les **sites officiels** d'où viennent les documents.

> Mini-projet – Sujet 6 : *Chatbot intelligent basé sur les documents administratifs*.
>
> 📄 **Rapport complet : [`docs/Rapport_Chatbot_RAG.pdf`](docs/Rapport_Chatbot_RAG.pdf)**

---

## ✨ Fonctionnalités

- Questions en langage naturel, avec une **interface de conversation** (Streamlit).
- Réponses **fondées uniquement sur les documents**, organisées en rubriques (étapes, documents à fournir, coût, délai), avec une **source numérotée** après chaque information.
- **« Je ne trouve pas cette information dans les documents »** pour les questions hors sujet. Dans ce cas, aucune source ni aucun formulaire n'est affiché.
- **Filtre selon la situation** : en Tunisie (police, municipalité) ou à l'étranger (consulat). Ce filtre est choisi par l'utilisateur ou détecté automatiquement dans la question.
- **Formulaires officiels** : aperçu dans la page et téléchargement (CIN, passeport, autorisation parentale).
- **Sources consultables** : chaque PDF cité peut être ouvert dans la page, à la bonne page.
- **Sites officiels** : les liens écrits dans les PDF utilisés sont affichés sous la réponse (ils ne sont jamais choisis par le LLM).
- **Rapide** : réponse en **≈ 2 secondes** grâce à l'API Groq, avec **repli automatique sur un LLM local** (Ollama) si l'API n'est pas disponible.
- **Mémoire des réponses** : une question déjà posée s'affiche immédiatement.

---

## 🧠 Fonctionnement (le pipeline RAG)

```
INDEXATION (une seule fois)
PDF ──► extraction ──► nettoyage ──► découpage ──► embeddings ──► ChromaDB
        (pymupdf)      (regex)       (600 car.)    (e5-base)      (base vectorielle)

QUESTION → RÉPONSE
question ──► embedding ──► 5 meilleurs extraits ──► prompt + extraits ──► LLM ──────────────► réponse citée
                           (filtre Tunisie/étranger)                      Groq : gpt-oss-120b      + formulaire
                                                                          secours : Ollama qwen2.5:3b + sources
```

| Étape | Fichier | Rôle |
|---|---|---|
| 1 | `src/extraction.py` | Extrait le texte de chaque page des PDF (avec le nom du fichier et le numéro de page) |
| 2 | `src/nettoyage.py` | Supprime les en-têtes, les pieds de page et les espaces inutiles |
| 3 | `src/decoupage.py` | Découpe en chunks de 600 caractères (chevauchement de 80) et ajoute les métadonnées : source, page, type, public |
| 4 | `src/embeddings.py` | Transforme chaque chunk en vecteur de 768 nombres (`intfloat/multilingual-e5-base`) |
| 5 | `src/indexation.py` | Stocke les vecteurs dans ChromaDB et fait la recherche (avec les améliorations ci-dessous) |
| 6 | `src/generation.py` | Construit le prompt (règles + extraits), appelle le LLM (Groq, ou Ollama en secours) et remet les citations au bon format |
| 7 | `src/choix_formulaire.py` | Choisit le formulaire à proposer, à partir des fiches citées dans la réponse |
| 7 | `src/app.py` | Interface web Streamlit (conversation, formulaires, sites officiels, sources) |
| 8 | `src/evaluation_*.py`, `src/comparer_chunks.py`, `src/calcul_resultats.py` | Évaluation de la recherche et des réponses |
| 9 | `tests/test_projet.py` | Tests fonctionnels |

### Améliorations de la recherche
- **Expansion de requête** : « cin » devient « carte d'identité nationale (CIN) », et « passport » devient « passeport ».
- **Au plus 2 extraits par document** : un document généraliste (le guide) ne peut plus prendre toutes les places.
- **Priorité aux fiches de procédure** (bonus de +0,01 au score).
- **Détection de la zone** dans la question (« en Tunisie », « consulat », « France »…).
- **Small-to-big** : on cherche avec des petits chunks, mais le LLM reçoit la **page entière** de la meilleure fiche, pour que la liste des papiers ne soit jamais coupée.

---

## 🧰 Choix des outils

| Besoin | Outil choisi | Pourquoi |
|---|---|---|
| Lire les PDF | **PyMuPDF** | Rapide, garde le numéro de page, permet aussi d'afficher les pages en image |
| Découper le texte | **LangChain Text Splitters** | Découpe en respectant les phrases et les paragraphes, avec chevauchement |
| Embeddings | **Sentence-Transformers** (`multilingual-e5-base`) | Gratuit, local, bon en français, recommandé par le sujet |
| Base vectorielle | **ChromaDB** | Enregistre la base sur le disque et permet de filtrer par métadonnées (Tunisie / étranger) |
| LLM | **Groq** (`openai/gpt-oss-120b`) | Très rapide (≈ 2 s), meilleures réponses que le modèle local, offre gratuite |
| LLM de secours | **Ollama** (`qwen2.5:3b`) | Gratuit et local : l'application marche même sans Internet, mais lentement |
| Interface | **Streamlit** | Interface de chat moderne en Python, sans HTML/JavaScript à écrire |
| Clé API | **python-dotenv** | La clé est lue dans `.env` et n'est jamais écrite dans le code |

---

## 📁 Structure du projet

```
chatbot-demarches-tunisie/
├── data/
│   ├── procedures/        14 fiches pratiques (CIN, Passeport, Municipalite)
│   ├── reglements/        5 lois (57-3, 75-40, 2015-46, 2024-22, 2024-23)
│   ├── guides/            1 guide d'orientation du citoyen
│   └── processed/         fichiers générés (pages, chunks, vecteurs) – non versionnés
├── formulaires/           formulaires officiels (PDF) + formulaires.csv (fiche → formulaire)
├── images/                illustrations de l'interface
├── evaluation/            questions de test et résultats (recherche, réponses Groq et Ollama)
├── src/                   tout le code Python
├── tests/                 tests fonctionnels
├── docs/                  rapport du projet (PDF)
├── .streamlit/config.toml thème de l'interface
├── .env.example           modèle du fichier de configuration (sans clé)
├── SOURCES.md             origine de chaque document (URL + date de consultation)
├── requirements.txt
└── README.md
```

**Corpus** : 20 PDF (14 procédures, 5 lois, 1 guide), 44 pages, 232 chunks. Les lois viennent du Journal officiel ; les fiches viennent du ministère de l'Intérieur, du SICAD et d'Idaraty (non officiel, recoupé avec le ministère quand c'était possible) : voir [`SOURCES.md`](SOURCES.md).

---

## ⚙️ Installation (Windows)

**Prérequis** : Python 3.12 ou plus récent. Optionnel : [Ollama](https://ollama.com) pour le LLM de secours.

```powershell
# 1. Récupérer le projet
git clone https://github.com/naaaaaaaazz/chatbot-demarches-tunisie.git
cd chatbot-demarches-tunisie

# 2. Créer et activer l'environnement virtuel
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Installer les bibliothèques
pip install -r requirements.txt

# 4. (Optionnel) LLM de secours local (≈ 2 Go)
ollama pull qwen2.5:3b
```

Au premier lancement, le modèle d'embeddings `multilingual-e5-base` (≈ 1 Go) est téléchargé automatiquement.

### 🔑 Configuration de l'API Groq
1. Créer une clé gratuite sur [console.groq.com](https://console.groq.com) → **API Keys** → **Create API Key**.
2. Copier le modèle de configuration avec `copy .env.example .env` (à la racine du projet), puis ouvrir `.env` et remplacer `ta_cle_ici` par la clé :
   ```
   GROQ_API_KEY=ta_cle_ici
   LLM_FOURNISSEUR=groq
   GROQ_MODELE=openai/gpt-oss-120b
   ```
3. Le fichier `.env` est dans `.gitignore` : **la clé n'est jamais envoyée sur GitHub**.

### 🔁 Sans clé : le LLM de secours (Ollama)
Si `.env` n'existe pas, si Internet est coupé ou si le quota gratuit de Groq est dépassé, l'application affiche « API indisponible » dans le terminal et répond **automatiquement avec Ollama** (`qwen2.5:3b`, sur l'ordinateur).
Cela fonctionne **seulement si** Ollama est installé ([ollama.com](https://ollama.com)) et si le modèle a été téléchargé (`ollama pull qwen2.5:3b`, étape 4 de l'installation). Sinon, l'application affiche une erreur.
Avec Ollama, une réponse prend environ 2 minutes (sans carte graphique). Pour utiliser uniquement Ollama, mettre `LLM_FOURNISSEUR=ollama` dans `.env`.

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

Le navigateur s'ouvre sur **http://localhost:8501**.

### Lancer chaque étape séparément (facultatif)
Chaque fichier de `src/` peut être lancé seul, depuis la racine du projet, pour voir le résultat de son étape :

| Commande | Ce qu'elle fait | Ce qu'elle affiche ou crée |
|---|---|---|
| `python src/extraction.py` | Lit les 20 PDF page par page | Bilan (20 PDF, 44 pages) + `data/processed/pages_extraites.json` |
| `python src/nettoyage.py` | Supprime le bruit (en-têtes, numéros de page) | Caractères avant / après + `pages_nettoyees.json` |
| `python src/decoupage.py` | Découpe en chunks de 600 caractères | Nombre de chunks (232) + `chunks.json` |
| `python src/embeddings.py` | Enchaîne les 3 étapes ci-dessus, puis calcule les vecteurs | `embeddings.npy` (232 vecteurs de 768 nombres) |
| `python src/indexation.py` | Crée la base ChromaDB | Dossier `vector_db/` (232 chunks) |
| `python src/generation.py` | Pose 4 questions de test au chatbot dans le terminal | Réponses, sources et temps de recherche / génération |
| `python src/choix_formulaire.py` | Pose 3 questions de test et choisit le formulaire | Nombre de fiches ayant un formulaire (11), puis le formulaire proposé pour chaque question |
| `streamlit run src/app.py` | Lance l'interface | Chatbot sur http://localhost:8501 |

⚠️ Désactiver la traduction automatique de Chrome sur cette page : elle modifie la page pendant l'affichage de la réponse et provoque une erreur `removeChild`.

### 🆘 En cas de problème
| Message ou symptôme | Cause | Solution |
|---|---|---|
| PowerShell refuse d'activer le venv (« l'exécution de scripts est désactivée ») | Politique de sécurité de PowerShell | `Set-ExecutionPolicy -Scope Process RemoteSigned`, puis réactiver le venv |
| `Collection [chunks_600] does not exist` ou `FileNotFoundError: ... chunks.json` | La base n'a pas encore été construite | `python src/embeddings.py` puis `python src/indexation.py` |
| « API indisponible » dans le terminal | Pas de `.env`, pas d'Internet, ou quota Groq dépassé | Vérifier `.env` et la connexion ; sinon la réponse vient d'Ollama (plus lente) |
| Erreur de connexion à Ollama | Ollama n'est pas lancé, ou le modèle n'est pas téléchargé | Ouvrir Ollama, puis `ollama pull qwen2.5:3b` |
| Erreur `removeChild` dans le navigateur | Traduction automatique de Chrome | Désactiver la traduction pour la page |
| Port 8501 déjà utilisé | Une ancienne fenêtre Streamlit tourne encore | La fermer (Ctrl + C) ou `streamlit run src/app.py --server.port 8502` |
| `PermissionError` sur un fichier CSV | Le fichier est ouvert dans Excel | Fermer Excel, puis relancer |

### Tests fonctionnels
```powershell
python tests/test_projet.py
```
| Test | Ce qu'il vérifie |
|---|---|
| Base | ChromaDB contient les 232 chunks |
| Recherche | une question sur la CIN trouve une fiche CIN dans les 3 premiers résultats |
| Citations | les citations mal écrites par le LLM (`【…】`, gras, italique) sont remises au format `[fichier.pdf, p. X]` |
| Réponse | une vraie question reçoit une réponse qui cite un PDF |
| Refus | une question hors sujet donne « Je ne trouve pas » et aucun formulaire |
| Sécurité | aucune clé API dans le code, et `.env` est ignoré par git |

Résultat : **6 / 6 tests réussis**.

### Évaluation
```powershell
python src/evaluation_recherche.py                     # rappel@3, rappel@5, MRR (recherche simple vs améliorée)
python src/comparer_chunks.py                          # comparaison des chunks de 300 et 600 caractères
python src/evaluation_reponses.py                      # génère les réponses des 35 questions, à noter à la main
python src/calcul_resultats.py                         # résultats de la notation (réponses Groq)
python src/calcul_resultats.py reponses_ollama.csv     # résultats de la notation (anciennes réponses Ollama)
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

### Réponses : comparaison des deux LLM (notation manuelle, de 0 à 2)
La recherche est la même dans les deux cas : seul le LLM change.

| Mesure | Ollama `qwen2.5:3b` (local) | **Groq `gpt-oss-120b` (API, version finale)** |
|---|---|---|
| Exactitude moyenne (30 questions) | 1,53 / 2 (20 à 2, 6 à 1, 4 à 0) | **1,77 / 2** (24 à 2, 5 à 1, 1 à 0) |
| Fidélité moyenne (30 questions) | 1,83 / 2 (25 à 2, 5 à 1, 0 à 0) | **1,93 / 2** (28 à 2, 2 à 1, 0 à 0) |
| Refus corrects (5 questions hors sujet) | 5 / 5 | **5 / 5** |
| Temps moyen par réponse | 114 s (de 38 à 282 s) | **≈ 2 s** (de 1 à 5 s) |

Les réponses notées sont dans `evaluation/reponses_llm.csv` (Groq) et `evaluation/reponses_ollama.csv` (Ollama).

**Ce que montre l'évaluation :**
- le passage à l'API est **environ 50 fois plus rapide** et améliore à la fois l'exactitude et la fidélité ;
- les questions **précises** (prix, délais, nombre de photos, case du formulaire) obtiennent presque toutes 2/2 ;
- le chatbot **refuse toujours** le hors sujet et n'invente presque jamais (aucune fidélité à 0) ;
- les erreurs restantes viennent surtout de la **recherche** : quand la bonne fiche n'arrive qu'en 4ᵉ ou 5ᵉ position (Q4, Q12), le LLM ne reçoit pas sa page entière et la liste des papiers est incomplète. Il y a aussi quelques **réponses incomplètes** (Q14, Q23) et deux réponses avec **des étapes ajoutées sans source** (Q13, Q25).

**Bug découvert grâce à l'évaluation** : le formulaire CIN était proposé pour des questions sans rapport (extrait de naissance, légalisation, permis de conduire…). Le choix du formulaire a été corrigé : plus de formulaire après un refus, et le plan B ne regarde que le 1ᵉʳ résultat. Les formulaires proposés à tort sont passés de **9 à 1**.

---

## ⚠️ Limites

- **Dépendance à l'API** : il faut Internet et une clé Groq. L'offre gratuite limite le nombre de requêtes par minute et par jour. En secours, Ollama fonctionne hors ligne mais prend 2 à 4 minutes par réponse.
- **Données envoyées à l'API** : la question et les extraits des documents (publics) sont envoyés à Groq. Il ne faut donc pas écrire d'informations personnelles dans la question.
- **Réponses variables** : un LLM ne répond jamais deux fois exactement pareil. Les notes de l'évaluation correspondent à une exécution précise.
- **Recherche** : si la bonne fiche est mal classée, la réponse peut être incomplète (voir Q4).
- **Sensibilité à la formulation** : les fautes de frappe et les abréviations peuvent gêner la recherche. Elles sont en partie corrigées par l'expansion de requête.
- **Hors sujet** : les scores des questions hors sujet (0,81 à 0,84) sont proches de ceux des vraies questions (≈ 0,85). C'est donc le LLM, et non un seuil, qui décide de refuser.
- **Mémoire des réponses** : elle reconnaît seulement une question écrite exactement pareil, et elle est vidée quand on relance l'application.
- Les documents ont été consultés en **novembre 2026** : les démarches et les tarifs peuvent changer.

---

## 🔒 Sources et données personnelles
- L'origine de chaque document est dans [`SOURCES.md`](SOURCES.md).
- Les exemples de documents montrés dans l'interface (CIN, extrait de naissance) ont été **floutés** pour masquer toutes les données personnelles.
- La clé API reste dans `.env`, qui n'est jamais publié.