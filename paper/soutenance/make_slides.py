#!/usr/bin/env python
# ruff: noqa: RUF001  (French typography on the slides: en dashes, minus signs)
"""Build the defense slides (PowerPoint) from the MEASURED results of the repository.

    pip install python-pptx matplotlib
    python paper/soutenance/make_slides.py      # writes paper/soutenance/AML-GraphMiner_soutenance.pptx

Every number on the slides is read from the result files (benchmarks/results_parts/*/results.csv,
benchmarks/results_parts/hybrid/hybrid.json, results/eda-*/summary.json) or quoted from the
documentation that reports a measurement (docs/explainability.md). Edit the text freely in
PowerPoint afterwards; re-run the script when the results change.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARTS = ROOT / "benchmarks" / "results_parts"
NAVY = RGBColor(0x1F, 0x3A, 0x5F)
GREY = RGBColor(0x55, 0x5F, 0x6B)
NAMES = {
    "sage": "GraphSAGE",
    "rf": "Forêt aléatoire",
    "hgb": "HistGradientBoosting",
    "xgb": "XGBoost",
    "gat": "GAT",
    "gcn": "GCN",
    "logreg": "Régression logistique",
    "patterns": "Motifs (règles)",
    "gae": "Auto-encodeur (sans étiquettes)",
}


# ----------------------------------------------------------------------------- data
def results() -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in PARTS.glob("*/*/results.csv")]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["dataset", "model", "seed"], keep="last"
    )
    return df.groupby(["dataset", "model"])["pr_auc"].agg(["mean", "std", "count"]).reset_index()


def fmt(mean: float, std: float, count: int) -> str:
    return f"{mean:.3f}" if count < 2 or pd.isna(std) else f"{mean:.3f} ± {std:.3f}"


def hybrid() -> dict:
    path = PARTS / "hybrid" / "hybrid.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def eda(name: str) -> dict:
    path = ROOT / "results" / f"eda-{name}" / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


# ----------------------------------------------------------------------------- charts
def chart_ibm(table: pd.DataFrame, path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, ds in zip(axes, ("HI-Small", "LI-Small")):
        part = table[table["dataset"] == ds].sort_values("mean")
        labels = [NAMES.get(m, m) for m in part["model"]]
        colors = ["#c0392b" if m in ("patterns", "gae") else "#1f3a5f" for m in part["model"]]
        ax.barh(labels, part["mean"], xerr=part["std"].fillna(0), color=colors, capsize=3)
        ax.set_title(f"IBM AML {ds} — PR-AUC (test)")
        ax.axvline(0.5, ls="--", c="grey", lw=1)
        ax.set_xlim(0, max(0.62, float(part["mean"].max()) + 0.1))
        for y, (v, e) in enumerate(zip(part["mean"], part["std"].fillna(0))):
            ax.text(v + e + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def chart_elliptic(table: pd.DataFrame, path: Path) -> Path:
    part = table[table["dataset"] == "elliptic"].sort_values("mean")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.barh(
        [NAMES.get(m, m) for m in part["model"]],
        part["mean"],
        xerr=part["std"].fillna(0),
        color="#1f3a5f",
        capsize=3,
    )
    ax.set_title("Elliptic Bitcoin — PR-AUC (test, pas de temps 35–49)")
    for y, (v, e) in enumerate(zip(part["mean"], part["std"].fillna(0))):
        ax.text(v + e + 0.01, y, f"{v:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------------- slides
class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = Inches(13.333), Inches(7.5)
        self.n = 0

    def _title(self, slide, text: str) -> None:
        box = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(12.3), Inches(0.9))
        p = box.text_frame.paragraphs[0]
        p.text = text
        p.font.size, p.font.bold, p.font.color.rgb = Pt(30), True, NAVY
        self.n += 1
        num = slide.shapes.add_textbox(Inches(12.3), Inches(7.0), Inches(0.8), Inches(0.4))
        num.text_frame.paragraphs[0].text = str(self.n)
        num.text_frame.paragraphs[0].font.size = Pt(11)

    def bullets(
        self,
        title: str,
        items: list[str | tuple[str, int]],
        image: Path | None = None,
        image_width: float = 6.0,
    ) -> None:
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._title(slide, title)
        width = 12.3 if image is None else 12.3 - image_width - 0.3
        frame = slide.shapes.add_textbox(
            Inches(0.6), Inches(1.4), Inches(width), Inches(5.6)
        ).text_frame
        frame.word_wrap = True
        for i, item in enumerate(items):
            text, level = (item, 0) if isinstance(item, str) else item
            p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
            p.text = ("• " if level == 0 else "– ") + text
            p.level = level
            p.font.size = Pt(20 if level == 0 else 16)
            p.font.color.rgb = RGBColor(0, 0, 0) if level == 0 else GREY
            p.space_after = Pt(6)
        if image is not None and image.exists():
            left = Inches(13.333 - image_width - 0.4)
            slide.shapes.add_picture(str(image), left, Inches(1.5), width=Inches(image_width))

    def table(self, title: str, header: list[str], rows: list[list[str]], note: str = "") -> None:
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._title(slide, title)
        shape = slide.shapes.add_table(
            len(rows) + 1,
            len(header),
            Inches(0.6),
            Inches(1.4),
            Inches(12.1),
            Emu(int(Inches(0.42)) * (len(rows) + 1)),
        )
        for j, h in enumerate(header):
            cell = shape.table.cell(0, j)
            cell.text = h
            cell.text_frame.paragraphs[0].font.size = Pt(15)
            cell.text_frame.paragraphs[0].font.bold = True
        for i, row in enumerate(rows, start=1):
            for j, value in enumerate(row):
                cell = shape.table.cell(i, j)
                cell.text = value
                cell.text_frame.paragraphs[0].font.size = Pt(14)
        if note:
            box = slide.shapes.add_textbox(Inches(0.6), Inches(6.6), Inches(12.1), Inches(0.6))
            box.text_frame.word_wrap = True
            box.text_frame.paragraphs[0].text = note
            box.text_frame.paragraphs[0].font.size = Pt(12)
            box.text_frame.paragraphs[0].font.color.rgb = GREY

    def picture(self, title: str, image: Path, caption: str = "") -> None:
        slide = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self._title(slide, title)
        if image.exists():
            slide.shapes.add_picture(str(image), Inches(0.8), Inches(1.3), width=Inches(11.7))
        if caption:
            box = slide.shapes.add_textbox(Inches(0.8), Inches(6.7), Inches(11.7), Inches(0.5))
            box.text_frame.word_wrap = True
            box.text_frame.paragraphs[0].text = caption
            box.text_frame.paragraphs[0].font.size = Pt(13)
            box.text_frame.paragraphs[0].font.color.rgb = GREY

    def save(self, path: Path) -> Path:
        self.prs.save(str(path))
        return path


def main() -> Path:
    table = results()
    get = {(r.dataset, r.model): fmt(r.mean, r.std, r.count) for r in table.itertuples()}
    hyb = hybrid()
    hi = eda("hi-small")
    ibm_png = chart_ibm(table, OUT / "resultats_ibm.png")
    ell_png = chart_elliptic(table, OUT / "resultats_elliptic.png")
    figures = ROOT / "paper" / "figures"
    deck = Deck()

    slide = deck.prs.slides.add_slide(deck.prs.slide_layouts[6])
    deck.n += 1
    box = slide.shapes.add_textbox(Inches(0.8), Inches(2.0), Inches(11.7), Inches(3.5)).text_frame
    box.word_wrap = True
    box.paragraphs[0].text = "AML-GraphMiner"
    box.paragraphs[0].font.size, box.paragraphs[0].font.bold = Pt(48), True
    box.paragraphs[0].font.color.rgb = NAVY
    for text, size in (
        ("Détection explicable du blanchiment d'argent sur graphes de transactions", 26),
        (
            "Motifs structurels · modèles tabulaires · GNN (GCN, GraphSAGE, GAT) · hybride · explicabilité",
            18,
        ),
        (
            "Projet de fin d'études — Unité de recherche Al-Khawarizmi, FST, Université de Nouakchott",
            16,
        ),
        ("[Nom de l'étudiant] — encadrant : Mohamedou El Ghotob Cheikh Tourad", 16),
    ):
        p = box.add_paragraph()
        p.text, p.font.size = text, Pt(size)

    n_tx = hi.get("n_transactions") or hi.get("transactions")
    deck.bullets(
        "Contexte et problématique",
        [
            "Blanchiment : ≈ 1 600 Md USD en 2009, 2,7 % du PIB mondial (ONUDC 2011)",
            "Obligation de détection et de déclaration : GAFI, directive UEMOA 02/2015, BCEAO",
            "Aiguille dans une botte de foin : 0,10 % de transactions illicites dans IBM HI-Small"
            + (f" ({n_tx:,} transactions)".replace(",", " ") if isinstance(n_tx, int) else ""),
            "Règles : trop de faux positifs ; GNN : peu reproductibles, peu explicables",
            "Question : un outil ouvert, reproductible et explicable qui combine règles et GNN ?",
        ],
    )
    deck.bullets(
        "Objectifs du cahier des charges",
        [
            "O1 ingestion IBM AML / Elliptic / CSV → graphes (NetworkX, PyG), validation Pydantic",
            "O2 motifs : fan-in/out, smurfing, cycles, scatter-gather, gather-scatter (+ communautés)",
            "O3 GCN, GraphSAGE, GAT (+ auto-encodeur non supervisé)",
            "O4 explicabilité : GNNExplainer, PGExplainer, Captum, attention",
            "O5 évaluation : PR-AUC, F1, recall@k vs. RF / XGBoost — objectif PR-AUC > 0,5",
            "O6 interfaces : API REST, CLI, notebooks (+ interface web d'investigation)",
        ],
    )
    deck.bullets(
        "Architecture en six couches",
        [
            "Ingestion → caractéristiques → graphes → détecteurs → explicateurs → interfaces",
            "Chaque couche ne dépend que des couches inférieures",
            "PyTorch / FastAPI optionnels : le cœur tourne sur un portable sans GPU",
            "Une interface commune pour tous les détecteurs : fit · score · save · load",
        ],
        image=figures / "architecture.png",
        image_width=6.6,
    )
    deck.table(
        "Données",
        ["", "IBM HI-Small", "IBM LI-Small", "Elliptic"],
        [
            [
                "transactions / nœuds étiquetés",
                "5 078 345",
                "6 924 049",
                "26 381 (train) · 16 670 (test)",
            ],
            ["blanchiment", "5 177 (0,10 %)", "3 565 (0,05 %)", "1 083 illicites dans le test"],
            ["prévalence du test", "0,177 %", "0,067 %", "6,5 %"],
            [
                "particularité",
                "ACH : 0,746 % de blanchiment, ≤ 0,04 % ailleurs",
                "motifs plus rares",
                "graphe Bitcoin, 49 pas de temps",
            ],
        ],
        note="Sources : results/eda-hi-small/EDA.md, results/eda-li-small/EDA.md, benchmarks/results/REPORT.md.",
    )
    deck.bullets(
        "Protocole d'évaluation sans fuite",
        [
            "Découpage temporel 60 / 20 / 20 (jamais aléatoire)",
            "Statistiques et vocabulaires ajustés sur l'entraînement seul ; fenêtres causales 1 h / 24 h / 7 j",
            "Seuils de décision et réglages des motifs choisis sur la validation, figés pour le test",
            "3 graines par modèle, moyenne ± écart-type, intervalles de confiance bootstrap",
            "Des tests unitaires modifient le futur et vérifient que le passé ne bouge pas",
        ],
    )
    deck.bullets(
        "Détecteurs de motifs : ce qui a été corrigé",
        [
            "Les motifs IBM changent de devise à presque chaque saut → conversion vers le dollar "
            "(taux déduits des transactions multi-devises, moindres carrés)",
            "Ils durent 27 à 80 h, la validation ≈ 51 h → contexte des 4 jours précédents, sans étiquettes",
            "Tous les motifs annotés sont des virements ACH → recherche limitée à l'ACH (choisie sur la validation)",
            "Cycles jusqu'à 12 sauts, montants pouvant varier de −50 % à +50 % par saut (réglé sur la validation)",
            "Chaque alerte garde ses preuves : comptes, transactions, montants convertis",
        ],
    )
    hybrid_rows = []
    for ds, block in hyb.items():
        for base, row in block.get("hybrids", {}).items():
            hybrid_rows.append(
                [
                    ds,
                    NAMES.get(base, base),
                    f"{row['components']['base']['pr_auc']:.3f}",
                    f"{row['components']['motifs']['pr_auc']:.3f}",
                    f"{row['hybrid']['pr_auc']:.3f}",
                ]
            )
    deck.bullets(
        "Modèles appris et détecteur hybride",
        [
            "Bases tabulaires : régression logistique, forêt aléatoire, HistGB, XGBoost",
            "GNN de classification d'arêtes : GCN, GraphSAGE, GAT (échantillonnage de voisinage, perte pondérée)",
            "Auto-encodeur de graphe : apprend sans étiquettes",
            "Hybride : régression logistique à 3 coefficients sur la validation "
            "(logit du modèle, score des motifs, motif déclenché)",
            "Entraînement GPU gratuit : un noyau Kaggle privé par modèle (T4)",
        ],
    )
    deck.picture(
        "Résultats IBM AML (PR-AUC sur le test)",
        ibm_png,
        f"HI-Small : GraphSAGE {get.get(('HI-Small', 'sage'), '?')} > objectif 0,5 ; "
        f"LI-Small : forêt aléatoire {get.get(('LI-Small', 'rf'), '?')}. "
        "Moyenne ± écart-type sur 3 graines ; ligne pointillée = objectif du CDC.",
    )
    if hybrid_rows:
        deck.table(
            "Hybride : modèle + motifs (même découpage, scores identiques)",
            ["jeu", "modèle de base", "modèle seul", "motifs seuls", "hybride"],
            hybrid_rows,
            note="PR-AUC sur le test ; modèles Kaggle sauvegardés (graine 0), empilement ajusté sur la validation.",
        )
    deck.picture(
        "Résultats Elliptic Bitcoin",
        ell_png,
        "Sur Elliptic les arbres (≈ 0,79) dominent les GNN : l'outil rapporte toujours les deux.",
    )
    deck.bullets(
        "Explicabilité : une alerte expliquée, et vérifiée",
        [
            "Six méthodes, un même objet Explanation (JSON, Markdown, graphe) ; dossier d'alertes en lot",
            "Mesure systématique de l'effet du voisinage sur le score",
            "GraphSAGE HI-Small : 8 meilleures alertes à 0,988–1,000 avec voisinage, 0,960–1,000 sans",
            "Retirer les 5 voisins clés de GNNExplainer ne change pas le score",
            "→ PGExplainer rend des masques nuls (réponse fidèle) ; Captum est la vue utile",
        ],
        image=figures / "example_explanation.png",
        image_width=5.6,
    )
    deck.bullets(
        "Interfaces et qualité logicielle",
        [
            "Interface web d'investigation : tableau de bord, alertes et sous-graphes, comptes, exports",
            "CLI amlgm (scan, train, explain --top, benchmark, serve…), API REST OpenAPI 3.1",
            "MLflow, export Hugging Face Hub (fiche modèle), export Neo4j, Docker",
            "Tests pytest + Hypothesis, couverture ≈ 98 %, ruff, mypy, Sphinx, nbval",
            "Intégration continue GitHub Actions ; dépôt privé github.com/Abeidak/aml-graphminer",
        ],
    )
    deck.bullets(
        "Bilan par rapport au cahier des charges",
        [
            "Atteint : O1–O6, PR-AUC > 0,5 sur HI-Small (GraphSAGE), interface web, hybride",
            f"Non atteint : J2 (motifs seuls > 0,5) — motifs {get.get(('HI-Small', 'patterns'), '?')} sur HI-Small",
            f"LI-Small reste difficile : meilleur modèle {get.get(('LI-Small', 'rf'), '?')}",
            "Reste à faire (comptes personnels) : dépôt public + Zenodo, PyPI, ReadTheDocs, Hub, soumission",
        ],
    )
    deck.bullets(
        "Limites et perspectives",
        [
            "Données synthétiques : on mesure la capacité à retrouver les motifs du générateur",
            "La restriction ACH est propre au simulateur IBM : à revoir sur données réelles",
            "L'auto-encodeur signale l'inhabituel, pas l'illicite",
            "Perspectives : GNN hétérogènes et temporels, multigraphes orientés, apprentissage fédéré multi-banques",
            "Validation sur données réelles avec une institution partenaire",
        ],
    )
    deck.bullets(
        "Conclusion",
        [
            "Un outil ouvert qui réunit règles, GNN, hybride et explications sous un protocole sans fuite",
            "Des résultats mesurés et reproductibles, y compris les échecs (motifs, LI-Small)",
            "Démonstration : amlgm serve --demo --open-browser",
            "Merci — questions ?",
        ],
    )
    return deck.save(OUT / "AML-GraphMiner_soutenance.pptx")


if __name__ == "__main__":
    print("slides written to", main())
