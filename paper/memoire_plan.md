# Plan détaillé du mémoire de PFE — AML-GraphMiner

Plan proposé (≈ 70–90 pages hors annexes), relié à ce que contient le dépôt. Tous les chiffres
cités ont été **mesurés** avec le dépôt (sources indiquées entre crochets) ; aucun n'est un
objectif ou une estimation. Les figures et tableaux sont régénérables par les commandes données.

Sources des résultats :
[R] `benchmarks/results/REPORT.md` et `results.tex` (benchmark IBM AML + Elliptic) ;
[E] `results/eda-hi-small/EDA.md`, `results/eda-li-small/EDA.md` (analyse exploratoire) ;
[X] `docs/explainability.md` (effet mesuré du voisinage) ; [L] `docs/limitations.md` ;
[C] `configs/patterns.yaml` (seuils réglés sur la validation) ; [H] évaluation du détecteur
hybride (`benchmarks/results_parts/hybrid/README.md`).

---

## Pages liminaires

Page de titre, dédicaces, remerciements, résumé (FR, 250 mots) / *abstract* (EN) / ملخص (AR),
mots-clés, table des matières, listes des figures, tableaux et abréviations (AML, LBC/FT, GAFI,
BCEAO, GNN, GCN, GAT, PR-AUC, ACH…), déclaration d'usage d'outils d'IA selon le règlement de
l'université.

## Introduction générale (≈ 4 p.)

- Contexte : ampleur du blanchiment (ONUDC 2011 : ≈ 1 600 Md USD en 2009, 2,7 % du PIB mondial,
  dans la fourchette FMI de 2 à 5 %), obligations de déclaration (recommandations du GAFI ;
  directive n° 02/2015/CM/UEMOA et loi uniforme appliquée sous l'égide de la BCEAO).
- Problématique : alertes rares (0,10 % de transactions de blanchiment dans HI-Small [E]),
  systèmes à règles saturés de faux positifs, modèles profonds non explicables, expériences GNN
  non reproductibles (découpages aléatoires, fuites temporelles).
- Objectifs O1–O6 et jalons J1–J4 du cahier des charges ; contributions ; organisation du mémoire.

## Chapitre 1 — Blanchiment d'argent et détection (≈ 10 p.)

1.1 Définition, étapes (placement, empilage, intégration) et typologies : fan-in, fan-out,
smurfing, cycles, scatter-gather, gather-scatter, bipartite, stack (celles d'IBM AMLworld).
1.2 Cadre réglementaire : GAFI (recommandations, rapport 2021 sur les nouvelles technologies),
UEMOA/BCEAO, cellules de renseignement financier, exigence d'explicabilité.
1.3 Systèmes existants : moteurs de règles commerciaux, limites (faux positifs, coût), besoin
d'outils ouverts pour les banques régionales.

## Chapitre 2 — État de l'art (≈ 14 p.)

2.1 Apprentissage sur données tabulaires (régression logistique, forêts, gradient boosting,
XGBoost) et déséquilibre de classes (PR-AUC plutôt que ROC-AUC).
2.2 Graphes de transactions et GNN : GCN, GraphSAGE, GAT ; échantillonnage de voisinage ;
classification d'arêtes.
2.3 GNN pour la LBC : Weber et al. 2018 (AMLSim) et 2019 (Elliptic), Altman et al. 2023 (IBM
AMLworld), Egressy et al. 2024 (multigraphes orientés), Cardoso et al. 2022 (LaundroGraph),
van den Beukel et al. 2026 (générateur Tide).
2.4 Détection non supervisée (auto-encodeurs de graphes, DOMINANT).
2.5 Explicabilité : GNNExplainer, PGExplainer, Integrated Gradients (Captum), attention ;
fidélité d'une explication.
2.6 Jeux de données et pièges d'évaluation : fuite temporelle, statistiques ajustées sur tout
le jeu, seuil choisi sur le test ; synthèse comparative (tableau).

## Chapitre 3 — Analyse et conception (≈ 12 p.)

3.1 Besoins fonctionnels et non fonctionnels issus du CDC ; cas d'utilisation (chercheur,
enseignant, analyste LBC) ; analyse des notebooks Colab d'origine (`notebooks/colab_legacy/`) et
de leurs lacunes.
3.2 Architecture en six couches (`docs/architecture.md`, figure `paper/figures/architecture.pdf`) ;
diagramme de paquetages et de classes (détecteurs à interface commune `fit/score/save/load`).
3.3 Modèle de données canonique et validation (Pydantic + contrôles vectorisés).
3.4 Protocole d'évaluation : découpage temporel 60/20/20, caractéristiques causales ajustées sur
l'entraînement, seuil choisi sur la validation, graines multiples, intervalles de confiance.
3.5 Choix technologiques justifiés (PyTorch Geometric, FastAPI, Click, MLflow, Docker, Neo4j).

## Chapitre 4 — Réalisation (≈ 18 p.)

4.1 Ingestion : adaptateurs IBM (colonne `Account` dupliquée), Elliptic, CSV/JSON/Parquet,
moteur Polars, annotations `*_Patterns.txt` (62 % du blanchiment de HI-Small couvert par un
motif annoté).
4.2 Caractéristiques sans fuite (fenêtres 1 h / 24 h / 7 j, PageRank, degrés, vélocité) et tests
qui modifient le futur pour le prouver.
4.3 Détecteurs de motifs et communautés à risque : définitions formelles, paramètres YAML,
conversion des devises (taux inférés des transactions multi-devises, moindres carrés), contexte
historique non étiqueté (`lookback`), restriction aux virements ACH, calibration par précision
sur la validation, budgets de recherche des cycles.
4.4 Modèles appris : bases tabulaires, GCN / GraphSAGE / GAT (perte pondérée ou focale,
sélection sur la validation, entraînement PyTorch ou Lightning), auto-encodeur non supervisé,
**détecteur hybride** (empilement logistique modèle + motifs ajusté sur la validation).
4.5 Explicabilité : six méthodes, objet `Explanation` commun, explication par lot
(`amlgm explain --top`), mesure de l'effet du voisinage.
4.6 Interfaces : interface web d'investigation (captures : tableau de bord, alertes et
sous-graphes, comptes), CLI `amlgm`, API REST (OpenAPI 3.1), MLflow, export Hugging Face Hub,
export Neo4j, Docker.
4.7 Qualité logicielle : tests pytest + Hypothesis (propriétés : invariance à l'ordre des lignes,
cycles implantés toujours retrouvés, taux de change retrouvés), couverture, ruff, mypy, Sphinx,
intégration continue GitHub Actions (dont nbval sur les notebooks).
4.8 Entraînement sur GPU gratuit (Kaggle T4) : un noyau privé par modèle, portabilité des
modèles (format JSON d'XGBoost, versions de scikit-learn).

## Chapitre 5 — Expérimentations et résultats (≈ 16 p.)

5.1 Données [E] : HI-Small 5 078 345 transactions, 515 088 comptes, 5 177 blanchiments
(0,10 %), 15 devises, 1,42 % de paiements multi-devises ; LI-Small 6 924 049 transactions ;
l'ACH représente 600 797 transactions de HI-Small mais un taux de blanchiment de 0,746 %
(contre ≤ 0,04 % pour les autres formats). Figures EDA (montants, activité journalière, formats,
degrés, motifs). Elliptic : pas de temps 1–29 / 30–34 / 35–49, prévalence de test 6,5 %.
5.2 Protocole et environnement : découpage temporel ; prévalence de test 0,177 % (HI) et 0,067 %
(LI) ; CPU portable pour les modèles tabulaires et les motifs, GPU Kaggle T4 pour les GNN
(≈ 1 h 25 pour HI-Small, 2 h 15 pour LI-Small).
5.3 Résultats principaux [R] (PR-AUC test, moyenne ± écart-type sur 3 graines) :

| modèle | HI-Small | LI-Small | Elliptic |
|---|---|---|---|
| GraphSAGE | **0,531 ± 0,022** | 0,176 ± 0,018 | 0,497 ± 0,056 |
| forêt aléatoire | 0,499 ± 0,017 | **0,191 ± 0,007** | 0,786 ± 0,001 |
| HistGradientBoosting | 0,476 ± 0,011 | 0,106 ± 0,016 | **0,790 ± 0,004** |
| XGBoost | 0,474 ± 0,005 | 0,078 ± 0,007 | 0,788 ± 0,002 |
| GAT | 0,447 ± 0,007 | 0,110 ± 0,026 | 0,351 ± 0,042 |
| GCN | 0,374 ± 0,031 | 0,074 ± 0,024 | 0,300 ± 0,055 |
| motifs, sans apprentissage (réglés sur la validation) | 0,340 | 0,147 | — |
| régression logistique | 0,294 | 0,032 | 0,209 |
| auto-encodeur (sans étiquettes) | 0,003 ± 0,001 | 0,001 ± 0,000 | — |

5.4 Objectifs du CDC : PR-AUC > 0,5 atteint par GraphSAGE sur HI-Small (J3) ; non atteint sur
LI-Small (meilleur 0,191) ; le jalon J2 (motifs > 0,5) n'est pas atteint (0,340 sur HI-Small,
0,147 sur LI-Small), mais les motifs sont passés de 0,005 / 0,001 à ces valeurs. Analyse des causes
[L] et effet de chaque correction mesuré sur la validation de HI-Small (PR-AUC) : configuration
précédente 0,053 ; + conversion des devises 0,054 (les motifs IBM changent de devise à presque
chaque saut, mais durent 27 à 80 h, davantage que la fenêtre de validation de ≈ 51 h) ;
+ contexte des 4 jours précédents 0,085 ; + restriction à l'ACH 0,162 ; + nouveau réglage complet
0,258 (le même réglage sur tous les formats de paiement s'arrête à 0,164 ; LI-Small : 0,029 contre
0,093 avec l'ACH). Rappel « propre » des cycles sur le test : 0 → 0,81 ; scatter-gather : 0,02 → 0,85.
5.5 Rappel par motif du scanner (`per_pattern_recall.csv`) et comparaison supervisé /
non supervisé (l'auto-encodeur est au niveau du hasard : il signale l'inhabituel, pas l'illicite).
5.6 Détecteur hybride [H] (`benchmarks/results_parts/hybrid/README.md`, modèles Kaggle graine 0,
empilement ajusté sur la validation) : sur HI-Small la combinaison améliore les trois modèles —
forêt aléatoire 0,519 → 0,562, HistGB 0,484 → 0,545, GraphSAGE 0,504 → 0,527 (motifs seuls
0,340) ; sur LI-Small elle n'apporte rien (forêt 0,191 → 0,186, GraphSAGE 0,175 → 0,168, HistGB
0,106 → 0,113, sous les motifs seuls à 0,147) : discussion (prévalence 0,067 %, empilement
linéaire appris sur une validation de 827 positifs).
5.7 Études de cas d'explication [X] : pour GraphSAGE sur HI-Small, les 8 transactions les mieux
notées gardent un score de 0,960–1,000 sans leur voisinage (0,988–1,000 avec) ; retirer les
5 voisins classés premiers par GNNExplainer ne change pas le score : la décision repose sur la
transaction et les profils de ses comptes ; PGExplainer rend des masques nuls (réponse fidèle),
Captum est la vue informative. Captures de l'interface web et dossier d'alertes généré par
`amlgm explain --top`.
5.8 Discussion et menaces à la validité : données synthétiques (les scores mesurent la capacité
à retrouver les motifs du générateur), une seule exécution par modèle pour les modèles
sauvegardés, propriété ACH propre au simulateur, coût CPU (forêt aléatoire ≈ 99 min sur HI).

## Conclusion générale et perspectives (≈ 3 p.)

Bilan par objectif O1–O6 et livrables L1–L8 (dépôt GitHub privé prêt, PyPI, ReadTheDocs,
Hub et soumission restant à faire), limites, perspectives : GNN hétérogènes et temporels,
GNN pour multigraphes orientés, apprentissage fédéré multi-banques, validation sur données
réelles avec un partenaire, audit d'équité.

## Annexes

A. Installation (Windows) et guide de lancement (`GUIDE_FR.md`) ; B. référence de la CLI et de
la configuration YAML ; C. API REST (extraits OpenAPI) ; D. diagrammes de classes ; E. résultats
complets (`REPORT.md`, rappels par motif, environnement d'exécution) ; F. article Software Impacts
(`paper/main.tex`) ; G. planning réalisé vs. planning du CDC.

## Commandes pour régénérer figures et tableaux

```powershell
amlgm eda data\HI-Small_Trans.csv --out results\eda-hi-small
python benchmarks\run_ibm.py --data-dir data --datasets HI-Small LI-Small --models patterns --config configs\patterns.yaml --out benchmarks\results_parts\patterns
python benchmarks\make_report.py benchmarks\results benchmarks\results_parts\patterns benchmarks\results_parts\logreg benchmarks\results_parts\rf benchmarks\results_parts\hgb benchmarks\results_parts\xgb benchmarks\results_parts\gnn benchmarks\results_parts\elliptic-extra
python paper\figures\make_figures.py
```
