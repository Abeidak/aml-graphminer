# AML-GraphMiner — guide de lancement et de test (Windows)

Ce guide explique, pas à pas, comment installer le projet, lancer l'interface web, utiliser la
ligne de commande, exécuter les notebooks et les tests, travailler sur les vraies données (IBM AML,
Elliptic) et terminer les livrables du cahier des charges. Toutes les commandes sont à taper dans
**PowerShell**, depuis le dossier du projet :

```powershell
cd "C:\Users\dell\Desktop\MASTER PFE\PFE_FORMATION\PFE\aml-graphminer\aml-graphminer"
```

---

## 1. Installation (une seule fois)

Prérequis : Python **3.12** (`py -3.12 --version`), Git, et — facultatif — Docker Desktop.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# PyTorch CPU + pyg-lib (échantillonneur de voisins des GNN) : versions qui ont des roues Windows
.\.venv\Scripts\python.exe -m pip install "torch==2.10.0" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install pyg-lib -f https://data.pyg.org/whl/torch-2.10.0+cpu.html
# le projet avec toutes ses options + outils de test et de documentation
.\.venv\Scripts\python.exe -m pip install -e ".[all,dev,docs]"
.\.venv\Scripts\python.exe -m pip install jupyterlab ipykernel
.\.venv\Scripts\python.exe -m ipykernel install --sys-prefix --name amlgm --display-name "AML-GraphMiner (.venv)"
```

Ensuite, **à chaque nouvelle session PowerShell**, activez l'environnement :

```powershell
.\.venv\Scripts\Activate.ps1
amlgm --version          # doit afficher : amlgm, version 0.1.0
```

> **Attention — deux environnements.** Le dossier parent `PFE\aml-graphminer\` contient un autre
> `.venv` (Python 3.14, uniquement Jupyter) que VS Code active automatiquement. Utilisez celui
> du **projet** (`...\aml-graphminer\aml-graphminer\.venv`). Dans VS Code : `Ctrl+Shift+P` →
> *Python: Select Interpreter* → `.venv\Scripts\python.exe` du projet. Si PowerShell refuse
> `Activate.ps1`, lancez une fois `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## 2. Lancer l'interface web (le plus rapide pour une démonstration)

```powershell
amlgm serve --demo --open-browser
```

Le navigateur s'ouvre sur <http://127.0.0.1:8000/ui/> (premier lancement : ~10 s pour entraîner un
petit détecteur de démonstration). Ce que vous pouvez montrer au jury :

1. **Vue d'ensemble** : indicateurs clés, transactions par jour, alertes par motif, distribution
   des scores avec le seuil, montants, formats de paiement, métriques (PR-AUC, F1, rappel…).
2. **Alertes (motifs)** : cliquez une alerte → sa description, ses preuves chiffrées et son
   sous-graphe (fan-in, fan-out, smurfing, cycle, scatter-gather, gather-scatter, communauté).
3. **Transactions** : triez par score, filtrez « alertes du modèle uniquement », cliquez une
   transaction → **« Pourquoi ce score ? »** : explication en graphe (méthode au choix).
4. **Comptes** : cliquez n'importe quel nœud d'un graphe → profil du compte (activité,
   contreparties, alertes, voisinage).
5. **Charger un fichier…** : vos propres données (CSV/JSON/Parquet, format IBM ou canonique).
6. **Exporter** les alertes ou les scores en CSV ; **À propos** → documentation interactive de
   l'API (`/docs`, OpenAPI 3.1).

Arrêt du serveur : `Ctrl+C` dans PowerShell. Avec votre propre modèle :
`amlgm serve runs\sage --data data\Synth-Small_Trans.csv`.

---

## 3. Scénario complet en ligne de commande (données synthétiques)

```powershell
amlgm generate data --accounts 1500                          # données de type IBM + fichier de motifs
amlgm inspect  data\Synth-Small_Trans.csv                    # validation + résumé
amlgm eda      data\Synth-Small_Trans.csv --out eda          # analyse exploratoire (EDA.md + figures)
amlgm scan     data\Synth-Small_Trans.csv --out alerts       # 6 motifs + communautés à risque
amlgm train    data\Synth-Small_Trans.csv --model hgb  --out runs\hgb
amlgm train    data\Synth-Small_Trans.csv --model sage --out runs\sage --epochs 5     # GNN supervisé
amlgm train    data\Synth-Small_Trans.csv --model gae  --out runs\gae  --epochs 3     # GNN NON supervisé
amlgm score    runs\sage data\Synth-Small_Trans.csv --top 10
amlgm explain  runs\sage data\Synth-Small_Trans.csv --tx 9000 --method captum --plot why.png
amlgm explain  runs\sage data\Synth-Small_Trans.csv --tx 9000 --method gnnexplainer
amlgm benchmark data\Synth-Small_Trans.csv --models patterns --models rf --models hgb --models gcn --models sage --models gat --models gae --seeds 0 --seeds 1 --out results
amlgm export-neo4j data\Synth-Small_Trans.csv neo4j-import --model runs\sage
amlgm export-hub runs\sage hub\sage --repo-id VOTRE_COMPTE_HF/aml-graphminer-sage
```

`amlgm --help` et `amlgm <commande> --help` décrivent toutes les options. Méthodes d'explication :
`patterns` (preuves exactes, détecteur de motifs), `occlusion` (tout modèle), `gnnexplainer`,
`pgexplainer`, `captum`, `attention` (GAT).

---

## 4. Notebooks

```powershell
jupyter lab notebooks
```

Choisissez le noyau **« AML-GraphMiner (.venv) »**.

| notebook | contenu |
|---|---|
| `01_quickstart` | toute la boîte à outils sur données synthétiques (motifs, communautés, modèle, explications, GNN non supervisé, interface web) |
| `02_ibm_aml_benchmark` | EDA + comparaison motifs / tabulaire / GCN / GraphSAGE / GAT / auto-encodeur, graines, intervalles de confiance, rappel par motif ; mettez `$env:AML_DATA="data\HI-Small_Trans.csv"` avant `jupyter lab` pour les vraies données |
| `03_explainability` | preuves des motifs, occlusion, GNNExplainer, PGExplainer, Captum, attention |
| `04_gnn_training_colab` | benchmark des GNN (GCN, GraphSAGE, GAT, GAE, 3 graines) sur GPU gratuit **dans le navigateur** — Kaggle (recommandé) ou Colab ; voir §7 |

Les sources des notebooks sont dans `notebooks\src\*.py` ; après modification, régénérez-les avec
`python scripts\py2nb.py notebooks\src\01_quickstart.py notebooks\src\02_ibm_aml_benchmark.py notebooks\src\03_explainability.py --out notebooks --execute --workdir $env:TEMP\nbwork`.

---

## 5. Tests et contrôles qualité (ceux de la CI GitHub Actions)

```powershell
$env:MLFLOW_DISABLE_AGENT_HINT = "1"
pytest                                                         # ~440 tests (≈ 4 min)
pytest --cov=aml_graphminer --cov-report=term-missing          # couverture totale (≥ 90 % exigé par le CDC)
ruff check . ; ruff format --check .                           # style
mypy src                                                       # typage statique
sphinx-build -b html -n docs docs\_build\html                  # documentation (0 avertissement)
python -m build ; twine check dist\*                           # paquet PyPI (wheel + sdist)
```

Résultats obtenus lors de la vérification : **439 tests réussis, 0 échec**, couverture **98,3 %**
(installation complète) et **97,5 %** pour le cœur sans PyTorch (`pytest --cov=aml_graphminer
--cov-config=.coveragerc-core` dans un environnement sans torch), ruff et mypy sans erreur,
documentation sans avertissement, paquet validé par `twine check`.

Ouvrir la documentation construite : `start docs\_build\html\index.html`.

---

## 6. Docker (sans installer Python)

```powershell
docker build -f docker/Dockerfile -t aml-graphminer .
docker run --rm -p 127.0.0.1:8000:8000 aml-graphminer        # interface de démo sur http://127.0.0.1:8000/ui/
docker compose -f docker/docker-compose.demo.yml up --build   # idem avec docker compose
docker build -f docker/Dockerfile -t aml-graphminer:gnn --build-arg WITH_GNN=1 .   # image avec PyTorch/PyG
```

Neo4j (exploration du graphe en Cypher) :

```powershell
$env:NEO4J_PASSWORD = "change-me"
docker compose -f docker/docker-compose.neo4j.yml up -d
amlgm export-neo4j data\Synth-Small_Trans.csv neo4j-import --model runs\sage
docker compose -f docker/docker-compose.neo4j.yml exec neo4j cypher-shell -u neo4j -p change-me -f /import/import.cypher
```

Puis <http://127.0.0.1:7474> et les requêtes de `neo4j-import\queries.cypher` (cycles, fan-in…).

---

## 7. Vraies données (IBM AML et Elliptic)

1. Créez un compte Kaggle → *Settings* → *API* → **Create New Token** ; placez le fichier
   `kaggle.json` dans `C:\Users\dell\.kaggle\kaggle.json`.
2. Téléchargez :
   ```powershell
   amlgm download ibm --out data --size small            # HI-Small / LI-Small + *_Patterns.txt (~1 Go)
   amlgm download elliptic --out data\elliptic
   ```
3. Analyse exploratoire (jalon J1) : `amlgm eda data\HI-Small_Trans.csv --out results\eda-hi-small`.
   Vérifiez dans `amlgm -v inspect data\HI-Small_Trans.csv` que la ligne
   `attached N/M laundering-pattern transactions` donne N proche de M.
4. Baseline déterministe (jalon J2, PR-AUC visée > 0,5) et benchmark complet (jalon J3) :
   ```powershell
   python benchmarks\run_ibm.py --data-dir data --datasets HI-Small LI-Small --models patterns rf hgb xgb --seeds 0 1 2 --out benchmarks\results
   python benchmarks\run_elliptic.py --data-dir data\elliptic --models rf hgb --seeds 0 1 2 --out benchmarks\results
   python benchmarks\make_report.py benchmarks\results     # REPORT.md + results.tex (repris par le paper)
   ```
   Les seuils des motifs de `configs\patterns.yaml` ont été réglés sur les tranches de
   validation de HI-Small et LI-Small (jamais sur le test) : passez `--config configs\patterns.yaml`.
5. GNN sur GPU (sans VS Code, dans le navigateur) — **Kaggle**, recommandé car les fichiers IBM y
   sont déjà :
   1. `python -m build` → `dist\aml_graphminer-0.1.0-py3-none-any.whl` ;
   2. kaggle.com → *Create* → *New Notebook* → *File* → *Import Notebook* →
      `notebooks\04_gnn_training_colab.ipynb` ;
   3. *Add Input* → `ibm-transactions-for-anti-money-laundering-aml` ; *Add Input* → *Upload* →
      nouveau jeu privé `aml-graphminer-wheel` contenant le `.whl` ;
   4. *Session options* : *Accelerator* **GPU T4 x2**, *Internet* **On** (compte vérifié par
      téléphone) ;
   5. exécutez tout avec `QUICK = True` (quelques minutes), puis `QUICK = False` et
      *Save Version* → **Save & Run All (Commit)** : le calcul continue navigateur fermé ;
      téléchargez `aml-gnn-results.zip` dans l'onglet *Output* ;
   6. sur le PC : décompressez dans `gpu-results\` puis
      `python benchmarks\make_report.py benchmarks\results gpu-results\results`.
   Colab fonctionne aussi dans le navigateur (*Upload notebook*, *Runtime* → T4, fichiers dans
   `MyDrive/aml/`), mais la session gratuite s'arrête si l'onglet reste inactif et LI-Small peut
   dépasser sa mémoire.

---

## 8. Livrables qui demandent votre action (comptes personnels)

Ces étapes nécessitent vos identifiants ou votre signature d'auteur ; tout le reste est prêt.

| livrable CDC | quoi faire |
|---|---|
| L1 — dépôt GitHub (MIT) | fait : https://github.com/Abeidak/aml-graphminer (privé). Reste : renseigner auteurs / e-mails / ORCID dans `pyproject.toml` et `CITATION.cff`, puis rendre le dépôt public (*Settings* → *Change visibility*) ; la CI démarre à chaque `git push` |
| L2 — PyPI | créer le projet sur pypi.org, configurer le *trusted publishing* pour `.github/workflows/release.yml` (environnement `pypi`), puis `git tag v0.1.0 ; git push origin v0.1.0` |
| L3 — API déployée | `docker build` puis déploiement de l'image (Render, Fly.io…) ; commande par défaut = démo publique |
| L5 — ReadTheDocs | importer le dépôt sur readthedocs.org (`.readthedocs.yaml` fourni) |
| L6 — modèles HuggingFace | après entraînement sur IBM (notebook 04) : `amlgm export-hub runs\sage hub\sage --repo-id VOTRE_COMPTE/aml-graphminer-sage --push --token <jeton HF>` |
| L7 — article Software Impacts | compléter les `\todo{}` de `paper\main.tex` (auteurs, DOI Zenodo, résultats réels), compiler sur Overleaf, soumettre |
| L8 — mémoire | plan détaillé dans `paper\memoire_plan.md` ; reprenez figures et tableaux produits par les notebooks et `benchmarks\make_report.py` ; déclarez l'usage d'outils d'IA selon le règlement de votre université |

`RELEASE.md` contient la liste de contrôle complète avant publication.

---

## 9. Conformité au cahier des charges

| exigence (CDC / fiche sujet) | réalisation |
|---|---|
| O1 — ingestion CSV / JSON / Parquet → NetworkX + PyG, validation Pydantic, adaptateurs IBM et Elliptic | `ingestion/` (`csv_adapter`, `ibm_adapter`, `elliptic_adapter`, `schema`), `graph/` (`networkx_builder`, `pyg_builder`, conversion bidirectionnelle, graphes hétérogènes) |
| Neo4j ou NetworkX | les deux : NetworkX + export / chargement Neo4j (`amlgm export-neo4j`) |
| Feature engineering (fenêtres 1 h / 24 h / 7 j, PageRank, betweenness, degrés, entropie, vélocité, ratio entrant/sortant) | `features/` (sans fuite temporelle) |
| O2 — smurfing, fan-in / fan-out, cycles, scatter-gather, gather-scatter | `detectors/patterns/` (+ calibration, preuves, rappel par motif) |
| Détection de communautés (communautés à risque) | `detectors/patterns/community.py` (relais + Louvain) |
| O3 — GCN, GraphSAGE, GAT (classification des transactions, PyTorch Lightning) | `detectors/gnn/` (entraîneurs PyTorch et Lightning, focal loss, pondération) |
| Module non supervisé par GNN | `detectors/gnn/unsupervised.py` (auto-encodeur de graphe, `--model gae`) |
| O4 — GNNExplainer / PGExplainer, Captum, sous-graphe, export JSON | `explainers/` (6 méthodes, JSON / NetworkX / Markdown / figure) |
| O5 — PR-AUC, F1, recall@k sur IBM HI/LI et Elliptic, baselines RF / XGBoost | `evaluation/`, `benchmarks/`, notebook 02 (résultats réels : section 7) |
| O6 — API REST FastAPI (OpenAPI 3.1), CLI Click `amlgm scan\|train\|explain`, notebooks | `api/`, `cli.py`, `notebooks/01-04` |
| Interface graphique web d'investigation | `api/static/` servie sur `/ui/` |
| Générateur de données synthétiques réalistes | `synthetic/` (`amlgm generate`) |
| Docker + docker-compose, GitHub Actions, tests ≥ 90 %, Sphinx + MyST + RTD, MLflow, HuggingFace Hub, MIT, CITATION.cff | `docker/`, `.github/workflows/`, `tests/` (98,3 %), `docs/`, `tracking.py`, `hub.py`, `LICENSE`, `CITATION.cff` |
| Structure du dépôt §5.3 | respectée (voir `docs/architecture.md`, section « Mapping to the project specification ») |

---

## 10. Dépannage

| symptôme | solution |
|---|---|
| `amlgm` introuvable / `No module named aml_graphminer` | l'environnement n'est pas activé : `.\.venv\Scripts\Activate.ps1` |
| `'NeighborSampler' requires either 'pyg-lib' or 'torch-sparse'` | réinstallez `pyg-lib` avec la commande de la section 1 (même version que torch) |
| Jupyter ne voit pas le noyau / utilise Python 3.14 | `python -m ipykernel install --sys-prefix --name amlgm ...` puis choisir « AML-GraphMiner (.venv) » |
| `address already in use` au lancement de `serve` | `amlgm serve --demo --port 8001` |
| erreur MLflow avec `./mlruns` | le suivi utilise désormais `sqlite:///mlflow.db` ; visualisation : `mlflow ui --backend-store-uri sqlite:///mlflow.db` |
| `Kaggle credentials not found` | placez `kaggle.json` dans `C:\Users\dell\.kaggle\` (section 7) |
| interface : « exceed the limit » | l'interface garde au plus 2 millions de transactions en mémoire : chargez une tranche de dates, utilisez `amlgm score` pour un fichier entier |
