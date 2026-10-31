# État du projet AML-GraphMiner (mis à jour le 31/10/2026)

Suivi de tout ce qui a été fait et de ce qui reste, par rapport au cahier des charges (CDC) et à
la fiche sujet. Tous les chiffres sont **mesurés** (sources : `benchmarks/results/REPORT.md`,
`benchmarks/results_parts/hybrid/README.md`, `docs/limitations.md`). Légende : ✅ fait ·
🟡 partiel · ⏳ reste à faire · 👤 demande vos comptes ou votre signature.

---

## 1. Livrables du CDC (L1–L8)

| livrable | état | détail |
|---|---|---|
| L1 — Code source GitHub, licence MIT | ✅ | Dépôt **public** https://github.com/Abeidak/aml-graphminer (depuis le 29/10/2026), CI verte. Reste (facultatif) : ORCID dans `CITATION.cff`. |
| L2 — Package PyPI | ✅ | Publié : https://pypi.org/project/aml-graphminer/ — 0.1.0 le 29/10/2026, 0.1.1 le 30/10/2026 (les deux auteurs affichés) ; `pip install aml-graphminer` vérifié dans un environnement vierge. Publication automatique à chaque étiquette `vX.Y.Z`. |
| L3 — API REST FastAPI (OpenAPI 3.1) + Docker + déploiement cloud | ✅ | En ligne depuis le 29/10/2026 : https://aml-graphminer-demo.onrender.com/ui/ et https://aml-graphminer-demo.onrender.com/docs (Render, offre gratuite, données synthétiques, envois limités à 20 Mo ; veille après 15 min sans visite). |
| L4 — CLI Click `amlgm scan\|train\|explain` | ✅ | + `eda`, `score`, `benchmark`, `serve`, `explain --top` (lot d'alertes), `train --model hybrid`, exports Neo4j / Hub. |
| L5 — Sphinx + 3 notebooks + ReadTheDocs | ✅ | Documentation en ligne : https://aml-graphminer.readthedocs.io/ (construite sans erreur), notebooks 01–05 (01–03 et 05 exécutés, vérifiés par nbval en CI ; 05 = plugin Jupyter). |
| L6 — Modèles pré-entraînés sur Hugging Face | ✅ (privés) | 12 modèles HI-Small en dépôts privés `abeideck/aml-graphminer-<modèle>-hi-small`, rechargés avec succès depuis le Hub. Reste (facultatif) : les rendre publics, ajouter LI-Small. |
| L7 — Article Software Impacts | 🟡 👤 | `paper/main.tex` complet selon le plan du CDC, résultats mesurés, 23 références. Auteurs renseignés. Reste : DOI Zenodo, déclarations, source pour le chiffre du marché AML, compilation sur Overleaf, soumission. |
| L8 — Mémoire 60–80 p. + soutenance | 🟡 | **Mémoire rédigé : 71 pages** (Word et PDF dans `paper/memoire/`, A4, Times New Roman 12, interligne 1,5, sommaire, 15 figures, annexes A à G), tenu **hors du dépôt public** jusqu'à la soutenance (un logiciel anti-plagiat retrouverait sinon le texte en ligne). 16 diapositives avec les deux auteurs et les deux encadrants, discours dans les notes (`paper/soutenance/`). Reste : vos dédicaces, le jury et la date, la déclaration d'usage de l'IA, les références mauritaniennes marquées « À vérifier », relecture, répétition. |

## 2. Jalons (J1–J4)

| jalon | état | mesure |
|---|---|---|
| J1 — Pipeline opérationnel, EDA | ✅ | HI-Small chargé (5 078 345 transactions), EDA de HI et LI-Small (`results/eda-*`). |
| J2 — Motifs : PR-AUC > 0,5 sur HI-Small | ✅ de justesse | **0,503** sur le test (intervalle à 95 % : 0,479–0,526), LI-Small 0,189 ; 0,005 / 0,001 au départ. Règles déterministes dont les réglages et la calibration viennent de la validation ; ACH et exclusion multi-devises propres au simulateur IBM. |
| J3 — 3 GNN entraînés, benchmark publié | ✅ (benchmark dans le dépôt privé) | GraphSAGE **0,531 ± 0,022** sur HI-Small (> 0,5), GAT 0,447, GCN 0,374 ; meilleur hybride 0,641. |
| J4 — Article soumis | ⏳ 👤 | Manuscrit prêt hors champs personnels (voir L7). |

## 3. Objectifs (O1–O6) et fiche sujet

| objectif | état |
|---|---|
| O1 ingestion IBM / Elliptic / CSV-JSON-Parquet, Pydantic, NetworkX + PyG, Neo4j | ✅ (validation Pydantic branchée sur l'API et l'ingestion) |
| O2 motifs : smurfing, fan-in/out, cycles, scatter-gather, gather-scatter + communautés à risque | ✅ (+ relais, conversion des devises, contexte historique, filtre ACH, calibration par tranche ; J2 atteint de justesse) |
| O3 GCN, GraphSAGE, GAT + GNN non supervisé | ✅ |
| O4 GNNExplainer, PGExplainer, Captum, attention, sous-graphe, JSON | ✅ (+ effet mesuré du voisinage, explication par lot) |
| O5 PR-AUC, F1, recall@k sur IBM HI/LI et Elliptic, RF / XGBoost | ✅ (3 graines, écarts-types) |
| O6 API, CLI, notebooks | ✅ (+ interface web d'investigation) |
| Fiche sujet : générateur synthétique, détecteur hybride, plugin Jupyter | ✅ générateur, ✅ hybride, ✅ plugin Jupyter (`%load_ext aml_graphminer`, notebook 05) |

## 4. Résultats mesurés (PR-AUC sur le test)

| modèle | HI-Small | LI-Small | Elliptic |
|---|---|---|---|
| GraphSAGE | **0,531 ± 0,022** | 0,176 ± 0,018 | 0,497 ± 0,056 |
| motifs (règles, sans modèle appris) | 0,503 | 0,189 | — |
| forêt aléatoire | 0,499 ± 0,017 | **0,191 ± 0,007** | 0,786 ± 0,001 |
| HistGradientBoosting | 0,476 ± 0,011 | 0,106 ± 0,016 | **0,790 ± 0,004** |
| XGBoost | 0,474 ± 0,005 | 0,078 ± 0,007 | 0,788 ± 0,002 |
| GAT | 0,447 ± 0,007 | 0,110 ± 0,026 | 0,351 ± 0,042 |
| GCN | 0,374 ± 0,031 | 0,074 ± 0,024 | 0,300 ± 0,055 |
| régression logistique | 0,294 | 0,032 | 0,209 |
| auto-encodeur (sans étiquettes) | 0,003 | 0,001 | — |
| hybride modèle + motifs (graine 0) | **0,641** (HistGB ou GraphSAGE ; forêt 0,614) | **0,236** (forêt seule 0,191) | — |

## 5. Historique de ce qui a été fait

1. Audit complet puis second audit indépendant par rapport au CDC et à la fiche sujet ; corrections.
2. Données IBM réelles : inspection, EDA, réglage des motifs sur la validation, benchmark CPU
   (motifs, régression logistique, forêt, HistGB) sur 3 graines.
3. Kaggle (compte `bpmrim`, noyaux et jeu de données **privés**) : benchmark GNN sur GPU T4,
   Elliptic, XGBoost, un noyau d'entraînement par modèle ; modèles téléchargés dans `runs\` et
   vérifiés en local (format JSON portable pour XGBoost).
4. Dépôt GitHub privé créé, CI verte ; projet lancé en local (interface web).
5. Nuit du 27 au 28/10 (tâches 2 à 7) :
   - **2** PGExplainer : masques nuls expliqués (le voisinage change peu le score), effet du
     voisinage mesuré et affiché dans chaque explication GNN ;
   - **3** motifs : conversion des devises, 4 jours de contexte non étiqueté, filtre ACH,
     nouveau réglage sur la validation (HI 0,005 → 0,340, LI 0,001 → 0,147 sur le test) ;
   - **J2 (28/10, après-midi)** : détecteur de relais (motif *stack*), calibration par tranche de score,
     durée minimale des fans, exclusions ; tout choisi sur la validation : HI 0,503, LI 0,189 ;
     hybrides jusqu'à 0,641 (HI) et 0,236 (LI) ;
   - **bug corrigé** : écart entraînement / notation sur les transactions de même minute (HistGB
     0,317 → 0,484 via `score()`), benchmark inchangé ;
   - **4** détecteur hybride, explication par lot (`amlgm explain --top`), validation Pydantic,
     tests Hypothesis / PyG / Polars, nbval en CI (469 tests, couverture 98,4 %) ;
   - **5** corrections de documentation de l'audit ;
   - **6** article réécrit selon le plan du CDC ; **7** plan du mémoire et diapositives.
6. 29/10 : **vérification fonctionnelle de tous les livrables** (API, interface web, service
   déployé, CLI, notebooks) par rapport au CDC et à la fiche sujet :
   - API REST : 25/25 contrôles sur Render, 33/33 en local avec un GNN (`scripts/check_service.py`) ;
   - interface web pilotée dans un vrai navigateur : 12/12 sur Render, 16/16 en local
     (`tests/e2e/ui_check.js`) ; CLI : 17/17 commandes ;
   - défauts trouvés et corrigés : GNNExplainer sans attributs de compte, réponses périmées et
     double clic dans l'interface web, CLI Windows qui échouait sur une sortie redirigée ;
   - **plugin Jupyter** ajouté (dernier écart avec le CDC §1.4) : affichage riche des résultats,
     `%amlgm`, `%aml_scan`, `%aml_explain`, notebook 05.
7. 29/10 : **interface web entièrement en français**. L'API répond en anglais ; le fichier
   `api/static/fr.js` rédige en français les descriptions et preuves des motifs, les phrases
   d'alerte et d'explication (5 méthodes), les noms d'attributs, les formats de paiement, les
   devises, les messages d'erreur et le nom du jeu de démonstration ; nombres au format français.
   L'API fournit désormais les chiffres de ses phrases (`details` des explications, `evidence`
   des alertes listées, `origin` des jeux de données). Contrôles : `tests/test_web_texts.py`
   (fr.js exécuté par Node.js sur de vraies réponses de l'API) et un contrôle « aucun mot
   anglais » dans `tests/e2e/ui_check.js` : 17/17 en local avec GAT, 14/14 avec le détecteur de
   motifs, **13/13 sur Render** après déploiement (API : 25/25 sur Render).
8. 29–30/10 : **auteurs et encadrement** — Ahmed Ahmed Mahmoud Abeideck et Saade Jaavar Boubacar ;
   encadrant académique Ahmed El Emine Sejad, encadrant professionnel Mohamedou El Ghotob Cheikh
   Tourad (README, `CITATION.cff`, `LICENSE`, diapositive de titre) ; **PyPI** 0.1.0 puis 0.1.1 ;
   interface web : **graphe des 5 alertes les plus fortes** sur la vue d'ensemble, première alerte
   ouverte avec son sous-graphe, composantes séparées d'un graphe disposées côte à côte.
9. 30–31/10 : **améliorations et mémoire** —
   - clé d'accès facultative pour l'API (`X-API-Key`) ; entraînement Lightning suivi époque par
     époque et paramètres enregistrés dans MLflow ; coût de Polars mesuré (12,8 s au lieu de
     17,3 s, mais 4,2 Go au lieu de 2,6 Go au pic) ;
   - kit de validation des alertes par un expert (`scripts/expert_review.py`, 50 éléments à
     juger à l'aveugle, `GUIDE_FR.md` §7) ;
   - interface web : petits graphes lisibles, effets quasi nuls rédigés en clair, et fermeture
     d'un jeu de données qui annule ses requêtes en cours (plus d'erreur 404 ni de message
     d'erreur dans le jeu suivant) ; contrôles après déploiement sur Render (commit 407c717) :
     **interface 13/13, API 25/25** ;
   - **mémoire rédigé** (71 pages) : analyse exploratoire de HI-Small (fin de fichier presque
     entièrement illicite : 655 des 1 797 positifs du test tombent dans ses 8 derniers jours),
     schéma du modèle EdgeGNN, résultats détaillés des 9 modèles sur les 3 jeux, routes de
     l'API et configuration des motifs en annexe.

## 6. Ce qui reste

### À faire par vous (comptes personnels, signature) 👤

Procédure pas à pas (sans partager de mot de passe) : `GUIDE_FR.md`, section 8.

- [x] Auteurs et e-mails (`pyproject.toml`, `CITATION.cff`, `LICENSE`, `paper/main.tex`,
      diapositive de titre) ; ORCID facultatif, plus tard.
- [x] Dépôt public, ReadTheDocs, modèles sur le Hugging Face Hub (privés), API en ligne (Render).
- [x] Publication PyPI 0.1.0 (29/10) et 0.1.1 (30/10).
- [ ] Votre collègue : créer un compte pypi.org pour être ajoutée au projet (Manage → Collaborators)
      et configurer son adresse e-mail dans git (son premier commit porte une adresse
      `example.com`, qui ne le relie pas à son compte GitHub).
- [ ] Zenodo (DOI de la version), compléter C2 / C8 / C9 de l'article.
- [ ] Décider si les modèles du Hub deviennent publics (liens de l'article et du README).
- [ ] Supprimer le fichier de clé Render et révoquer la clé (tableau de bord Render).
- [ ] Article : source pour la taille du marché AML (ou supprimer la mention), déclarations
      (conflits d'intérêts, IA), compilation Overleaf, soumission à Software Impacts ; la
      référence à la directive UEMOA / BCEAO (article et `paper/memoire_plan.md`) ne s'applique
      pas à la Mauritanie : la remplacer par le cadre mauritanien.
- [x] Mémoire rédigé (71 pages, `paper/memoire/`). [ ] À compléter par vous : dédicaces,
      jury et date de soutenance, déclaration d'usage d'outils d'IA (annexe D), résumé arabe à
      faire relire, références au cadre mauritanien marquées « À vérifier », format exigé par
      l'université ; relecture des diapositives dans PowerPoint, répétition de la soutenance.
- [ ] Validation des alertes et explications avec un expert métier : kit prêt (`scripts/expert_review.py`, grille de 50 éléments dans `results/expert-review/`, procédure dans `GUIDE_FR.md` §7) ; reste à la faire remplir.

### Pistes techniques possibles (je peux les faire si vous le demandez)

- [ ] Consolider J2 (marge faible : 0,503) : détecteur du motif *bipartite* (sans structure
      exploitable aujourd'hui), meilleure couverture des gather-scatter (17 % trouvés par leur
      propre détecteur).
- [ ] Hybride sur LI-Small avec GraphSAGE (seul cas sans gain).
- [ ] Hybride sur 3 graines (aujourd'hui une seule graine, modèles Kaggle seed 0).
- [ ] Résultats de HI-Small avec et sans les 8 derniers jours du fichier, presque entièrement
      illicites (mémoire, section 4.7).
- [ ] Jeux *Medium* d'IBM (nécessitent le GPU Kaggle et plus de mémoire).

## 7. Environnement et matériel (fiche « Environnement et matériel »)

| prévu | utilisé |
|---|---|
| Développement : 8–16 Go RAM | portable Windows 11, Intel Core i7-1255U, 16 Go RAM, sans GPU dédié |
| Entraînement GNN : GPU ou Kaggle | Kaggle, GPU Tesla T4 (noyaux privés) ; Colab et cluster FST non utilisés |
| Benchmark Medium / Large | non fait (seuls HI/LI-Small et Elliptic) |
| Python 3.11+, PyTorch 2.x, PyG 2.5+, NetworkX 3.x, Pandas, scikit-learn, XGBoost, Captum, Pydantic, FastAPI, Click 8, pytest, Hypothesis, coverage, ruff, Sphinx + MyST, Docker | utilisés (Python 3.11–3.13 en CI) ; en plus : mypy, nbval, Neo4j |
| PyTorch Lightning | implémenté et testé (`--trainer lightning`), suivi par époque comme la boucle PyTorch ; comparaison sur données synthétiques : PR-AUC de test 0,851 (Lightning) contre 0,841 (PyTorch), courbes presque superposées dans MLflow ; le benchmark publié utilise la boucle PyTorch |
| Polars | mesuré (`benchmarks/io_engines.py`) : `load_dataset` de HI-Small en 12,8 s contre 17,3 s avec pandas, mais 4,2 Go de mémoire au pic contre 2,6 Go ; pandas reste le moteur par défaut |
| MLflow | utilisé : `amlgm train --mlflow` enregistre paramètres, courbes par époque et métriques de test (capture dans le mémoire) ; pas utilisé pour le benchmark publié |
| Poetry ou hatch | remplacé par setuptools + `build` + `twine` (standard, suffisant) |
| docker-compose | fichiers fournis ; non exécuté en local |
| Fly.io / Render / cluster FST | Render |
| Kaggle, GitHub + Actions, ReadTheDocs, Hugging Face Hub | utilisés |
| PyPI | en attente de l'étiquette `v0.1.0` |
| Elsevier (Software Impacts) | article prêt, non soumis |
| Données : HI/LI-Small, Elliptic, HI-Medium | HI/LI-Small et Elliptic ; HI-Medium non utilisé |

## 8. Limites connues

Données IBM synthétiques (les scores mesurent la capacité à retrouver les motifs du générateur) ;
filtre ACH propre au simulateur IBM ; l'auto-encodeur signale l'inhabituel, pas l'illicite ; API
sans authentification ; détails dans `docs/limitations.md`.
