# État du projet AML-GraphMiner (mis à jour le 28/10/2026)

Suivi de tout ce qui a été fait et de ce qui reste, par rapport au cahier des charges (CDC) et à
la fiche sujet. Tous les chiffres sont **mesurés** (sources : `benchmarks/results/REPORT.md`,
`benchmarks/results_parts/hybrid/README.md`, `docs/limitations.md`). Légende : ✅ fait ·
🟡 partiel · ⏳ reste à faire · 👤 demande vos comptes ou votre signature.

---

## 1. Livrables du CDC (L1–L8)

| livrable | état | détail |
|---|---|---|
| L1 — Code source GitHub, licence MIT | ✅ | Dépôt **public** https://github.com/Abeidak/aml-graphminer (depuis le 29/10/2026), CI verte. Reste (facultatif) : ORCID dans `CITATION.cff`. |
| L2 — Package PyPI | 🟡 👤 | Éditeur de confiance PyPI configuré (`Abeidak/aml-graphminer`, `release.yml`, environnement `pypi`) ; nom `aml-graphminer` libre. Reste : vrais nom et e-mail d'auteur, puis étiquette `v0.1.0`. |
| L3 — API REST FastAPI (OpenAPI 3.1) + Docker + déploiement cloud | ✅ | En ligne depuis le 29/10/2026 : https://aml-graphminer-demo.onrender.com/ui/ et https://aml-graphminer-demo.onrender.com/docs (Render, offre gratuite, données synthétiques, envois limités à 20 Mo ; veille après 15 min sans visite). |
| L4 — CLI Click `amlgm scan\|train\|explain` | ✅ | + `eda`, `score`, `benchmark`, `serve`, `explain --top` (lot d'alertes), `train --model hybrid`, exports Neo4j / Hub. |
| L5 — Sphinx + 3 notebooks + ReadTheDocs | ✅ | Documentation en ligne : https://aml-graphminer.readthedocs.io/ (construite sans erreur), notebooks 01–05 (01–03 et 05 exécutés, vérifiés par nbval en CI ; 05 = plugin Jupyter). |
| L6 — Modèles pré-entraînés sur Hugging Face | ✅ (privés) | 12 modèles HI-Small en dépôts privés `abeideck/aml-graphminer-<modèle>-hi-small`, rechargés avec succès depuis le Hub. Reste (facultatif) : les rendre publics, ajouter LI-Small. |
| L7 — Article Software Impacts | 🟡 👤 | `paper/main.tex` complet selon le plan du CDC, résultats mesurés, 23 références. Reste : auteurs, lien public + DOI Zenodo, déclarations, source pour le chiffre du marché AML, compilation sur Overleaf, soumission. |
| L8 — Mémoire 60–80 p. + soutenance | 🟡 | Plan détaillé avec résultats (`paper/memoire_plan.md`), 16 diapositives colorées et animées, discours dans les notes (`paper/soutenance/`). Reste : rédaction du mémoire, votre nom sur la diapositive 1, répétition. |

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
   motifs.

## 6. Ce qui reste

### À faire par vous (comptes personnels, signature) 👤

Procédure pas à pas (sans partager de mot de passe) : `GUIDE_FR.md`, section 8.

- [ ] Renseigner auteurs, e-mails, ORCID (`pyproject.toml`, `CITATION.cff`, `paper/main.tex`,
      diapositive de titre).
- [ ] Rendre le dépôt public, activer Zenodo (DOI), compléter C2 / C8 / C9 de l'article.
- [ ] PyPI (*trusted publishing*) puis `git tag v0.1.0 ; git push origin v0.1.0`.
- [ ] ReadTheDocs ; Hugging Face Hub (`amlgm export-hub runs\sage-HI-Small hub\sage --repo-id ... --push`).
- [ ] Déploiement cloud de l'API (image Docker fournie).
- [ ] Article : source pour la taille du marché AML (ou supprimer la mention), déclarations
      (conflits d'intérêts, IA), compilation Overleaf, soumission à Software Impacts.
- [ ] Mémoire : rédaction à partir de `paper/memoire_plan.md`, captures de l'interface web,
      relecture des diapositives dans PowerPoint, soutenance.
- [ ] Validation des alertes et explications avec un expert métier (`RELEASE.md`).

### Pistes techniques possibles (je peux les faire si vous le demandez)

- [ ] Consolider J2 (marge faible : 0,503) : détecteur du motif *bipartite* (sans structure
      exploitable aujourd'hui), meilleure couverture des gather-scatter (17 % trouvés par leur
      propre détecteur).
- [ ] Hybride sur LI-Small avec GraphSAGE (seul cas sans gain).
- [ ] Hybride sur 3 graines (aujourd'hui une seule graine, modèles Kaggle seed 0).
- [ ] Jeux *Medium* d'IBM (nécessitent le GPU Kaggle et plus de mémoire).

## 7. Limites connues

Données IBM synthétiques (les scores mesurent la capacité à retrouver les motifs du générateur) ;
filtre ACH propre au simulateur IBM ; l'auto-encodeur signale l'inhabituel, pas l'illicite ; API
sans authentification ; détails dans `docs/limitations.md`.
