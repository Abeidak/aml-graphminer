# %% [markdown]
# # Explaining alerts
#
# A bank cannot act on an unexplained score: a compliance officer must understand **why** an
# account was flagged, and a regulator may ask for the evidence. AML-GraphMiner offers six
# complementary explanation methods behind one data model (`Explanation`):
#
# | method | works with | nature |
# |---|---|---|
# | `patterns` | motif scanner | exact — the alert, its accounts, transactions and numbers |
# | `occlusion` | any detector | model-agnostic — remove neighbours and watch the score move |
# | `gnnexplainer` | GNN detectors | GNNExplainer (Ying et al., 2019): a mask learned per alert |
# | `pgexplainer` | GNN detectors | PGExplainer (Luo et al., 2020): one explainer network trained on many alerts |
# | `captum` | GNN detectors | Captum Integrated Gradients: neighbours, account features and transaction features |
# | `attention` | GAT detectors | attention weights (descriptive, not causal) |
#
# This notebook runs the first two on synthetic data; the GNN ones are shown in the last
# section and execute when PyTorch Geometric (and Captum) are installed. The same methods are
# available from `amlgm explain --method ...`, the REST API and the web interface.

# %%
# %pip install -q "aml-graphminer[viz]"

# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import aml_graphminer as aml
from aml_graphminer.detectors.bundle import MotifDetector, TabularDetector
from aml_graphminer.explainers import (
    OcclusionExplainer,
    TransactionIndex,
    detector_score_fn,
    explain_with_patterns,
    plot_explanation,
)
from aml_graphminer.utils import has_module

warnings.filterwarnings("ignore")

df = aml.generate_transactions(
    n_accounts=900, n_background=7000, days=10, illicit_ratio=0.02, seed=21
)
index = TransactionIndex(df)
masks = aml.temporal_split(df)
print(f"{len(df):,} transactions; test split starts at tx {int(df.index[masks['test']][0])}")

# %% [markdown]
# ## 1. Exact evidence from the motif scanner
#
# Train a calibrated motif detector (the calibration learns each motif's precision on the
# validation split so that alerts of different detectors can be ranked together), then explain
# the highest-scoring test transaction.

# %%
motifs = MotifDetector.fit(df, fractions=(0.6, 0.2, 0.2))
result = motifs.scanner.scan(df)
test_scores = motifs.score(df[masks["test"]], context=df[~masks["test"]])
top_tx = int(test_scores.sort_values(ascending=False).index[0])
print("explaining transaction", top_tx, "| score", round(float(test_scores[top_tx]), 3))
by_pattern = explain_with_patterns(df, top_tx, result, motifs.threshold)
print(by_pattern.to_markdown())

# %%
plot_explanation(by_pattern)
plt.show()

# %% [markdown]
# Reading the figure: the explained transaction is red; every other edge belongs to the same
# alert and is drawn thicker the higher the alert's score. In the JSON form each pattern entry
# lists the evidence dictionary of the detector (hub account, number of counterparties, time
# span, conservation ratio, ...) — everything needed to write the suspicious-activity narrative.

# %%
by_pattern.patterns[0]

# %% [markdown]
# ## 2. Model-agnostic occlusion for a learned detector
#
# `OcclusionExplainer` removes each transaction (and each account) of the neighbourhood and
# measures the score drop. It is **local**: the neighbourhood radius `hops` must cover what the
# model looks at (a 2-layer GNN sees 2 hops).

# %%
tabular = TabularDetector.fit(df, "rf", seed=0)
scores = tabular.score(df[masks["test"]], context=df[~masks["test"]])
target = int(scores.sort_values(ascending=False).index[0])
for hops in (1, 2):
    explanation = OcclusionExplainer(
        detector_score_fn(tabular), hops=hops, max_edges=40, threshold=tabular.threshold
    ).explain(df, target, index)
    print(f"--- hops={hops}: {len(explanation.edges)} transactions considered")
    print(explanation.summary)

# %%
plot_explanation(explanation)
plt.show()

# %% [markdown]
# **Attribute view.** Besides neighbours, the explanation says which *raw attributes* of the
# explained transaction (amount, currency, payment format, banks) push its score up or down:
# each one is replaced by the typical value of the neighbourhood and the score is recomputed.
# Because the other columns are left unchanged, derived features (e.g. paid / received ratio)
# change too — read it as a sensitivity, not as a causal effect.

# %%
pd.DataFrame(explanation.features).round(4)

# %% [markdown]
# ## 3. Sanity checks you should always run
#
# * **Fidelity** — does removing the most important neighbours really lower the score?
# * **Stability** — do two seeds give the same top neighbours?
#
# The first is cheap to check with the same tooling:

# %%
top_ids = [e.tx_id for e in explanation.edges if e.role != "target"][:3]
subset = df.drop(index=top_ids)
before = float(tabular.score(df.loc[[target]], context=df.drop(index=target)).iloc[0])
after = float(tabular.score(subset.loc[[target]], context=subset.drop(index=target)).iloc[0])
print(
    f"score with the full history: {before:.3f}  |  without its 3 most important neighbours: {after:.3f}"
)

# %% [markdown]
# ## 4. GNN explainers (needs `aml-graphminer[gnn]`)
#
# The code below trains a small GAT and explains its strongest test alert with GNNExplainer,
# PGExplainer and the attention weights. It is skipped automatically when PyTorch Geometric is
# missing.

# %%
GNN = has_module("torch") and has_module("torch_geometric")
if GNN:
    from aml_graphminer.detectors.gnn import GNNConfig, GNNDetector
    from aml_graphminer.explainers import (
        AttentionExplainer,
        GNNExplainerAdapter,
        PGExplainerAdapter,
    )

    cfg = GNNConfig(
        kind="gat", hidden=32, layers=2, epochs=5, batch_size=1024, num_neighbors=[8, 4]
    )
    gnn = GNNDetector.fit(df, cfg, seed=0)
    gnn_scores = gnn.score(df[masks["test"]], context=df[~masks["test"]])
    ranked = gnn_scores.sort_values(ascending=False)
    gnn_target = int(ranked.index[0])
    for explainer in (
        GNNExplainerAdapter(gnn.trained, hops=2),
        AttentionExplainer(gnn.trained, hops=2),
    ):
        print(explainer.explain(df, gnn_target, index).to_markdown())
else:
    print("PyTorch Geometric is not installed here: GNN explainers skipped.")

# %% [markdown]
# **PGExplainer** is *parametric*: instead of optimising one mask per alert, it trains once — on
# the neighbourhoods of the 15 strongest alerts here — a small network that predicts the
# importance of every transaction from the embeddings of its two accounts. New alerts are then
# explained in one forward pass, consistently with each other. When every transaction receives
# the same mask, the explainer is telling you that the model barely uses the neighbourhood for
# this alert — the decision comes from the transaction and account features, which Captum below
# attributes precisely.

# %%
if GNN:
    pg = PGExplainerAdapter(gnn.trained, hops=2, epochs=20)
    pg.fit(df, [int(t) for t in ranked.index[:15]], index)
    pg_explanation = pg.explain(df, gnn_target, index)
    print(pg_explanation.to_markdown())
    plot_explanation(pg_explanation)
    plt.show()

# %% [markdown]
# **Captum** (Integrated Gradients) attributes the probability of laundering to the neighbouring
# transactions, to the features of the two accounts (`sender.*`, `receiver.*`) and to the
# features of the transaction itself (`tx.*`: amount, payment format, recent activity windows
# ...), with a baseline equal to an average transaction. Needs `aml-graphminer[explain]`.

# %%
if GNN and has_module("captum"):
    from aml_graphminer.explainers import CaptumExplainerAdapter

    captum_explanation = CaptumExplainerAdapter(gnn.trained, hops=2).explain(df, gnn_target, index)
    print(captum_explanation.summary)
    captum_table = pd.DataFrame(captum_explanation.features).round(4)
else:
    captum_table = "Captum or PyTorch Geometric is not installed here: Captum attributions skipped."
captum_table
