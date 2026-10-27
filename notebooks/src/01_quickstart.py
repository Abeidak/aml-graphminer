# %% [markdown]
# # AML-GraphMiner — quick start
#
# This notebook walks through the whole toolkit in a few minutes **on synthetic data**
# (an IBM-AML-like generator ships with the package, so nothing has to be downloaded):
#
# 1. generate / load transactions into the canonical schema,
# 2. split them **in time** (train < validation < test) like a real deployment would,
# 3. detect classical laundering motifs (fan-in / fan-out, smurfing, cycles, scatter-gather,
#    gather-scatter) and measure what they recover,
# 4. train a tabular detector on leak-free features,
# 5. explain an alert in a way an investigator can audit,
# 6. save the detector and score new transactions with it,
# 7. find risk communities (rings of relay accounts),
# 8. detect unusual transactions **without labels** with a graph auto-encoder,
# 9. do all of this in the web investigation interface.
#
# > The numbers below come from a **synthetic** dataset and say nothing about real-world
# > performance; reproducible benchmarks on the IBM AML data are in `02_ibm_aml_benchmark`.

# %%
# %pip install -q "aml-graphminer[viz]"

# %%
import warnings

import matplotlib.pyplot as plt
import pandas as pd

import aml_graphminer as aml

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)
print("aml-graphminer", aml.__version__)

# %% [markdown]
# ## 1. Data
#
# `generate_transactions` produces a table with the same logical columns as the IBM AML files
# (`Timestamp, From Bank, Account, To Bank, Account, Amount Received, ...`) and injects
# laundering motifs together with look-alike *legitimate* behaviour (payroll bursts, merchant
# bursts, unrelated triangles) so that detectors are not trivially perfect.
#
# With real data you would call `aml.read_transactions("HI-Small_Trans.csv")` instead — the
# result has the same canonical schema.

# %%
df = aml.generate_transactions(
    n_accounts=900, n_background=7000, days=10, illicit_ratio=0.02, seed=7
)
print(
    f"{len(df):,} transactions, {df['sender'].nunique() + df['receiver'].nunique():,} account slots"
)
print("injected motifs:", df.attrs["injected"])
df.head()

# %%
print("laundering share: {:.2%}".format(df["label"].mean()))
df.groupby("pattern").size().sort_values(ascending=False)

# %% [markdown]
# ## 2. Temporal split
#
# Random splits leak the future into the past in transaction data. `temporal_split` cuts the
# time-ordered table 60 / 20 / 20 — exactly the protocol of the original Colab notebooks.

# %%
masks = aml.temporal_split(df)
pd.DataFrame(
    {
        "transactions": {k: int(v.sum()) for k, v in masks.items()},
        "laundering": {k: int(df.loc[v, "label"].sum()) for k, v in masks.items()},
    }
)

# %% [markdown]
# ## 3. Structural-motif detection
#
# The scanner needs no training: each detector looks for one well-known laundering structure.
# Every alert carries **evidence** (which accounts, which transactions, which numbers).

# %%
scanner = aml.PatternScanner()
result = scanner.scan(df)
print(result.summary())
result.to_frame().head(8)[["pattern", "score", "n_tx", "n_accounts", "total_amount"]]

# %% [markdown]
# How much of the injected laundering does each detector recover, and how precise are its
# alerts? (`recall_same` counts only the *matching* motif, `recall_any` any alert.)

# %%
from aml_graphminer.detectors.patterns.evaluation import (
    alert_precision_table,
    evaluate_scan,
    pattern_recall_table,
)

print({k: round(v, 3) for k, v in evaluate_scan(df, result).items() if isinstance(v, float)})
pattern_recall_table(df, result).round(3)

# %%
alert_precision_table(df, result).round(3)

# %% [markdown]
# ## 4. Leak-free features and a tabular detector
#
# `FeaturePipeline` is fitted **on the training transactions only** and then applied to the
# other splits. Edge features use trailing windows (1 h, 24 h, 7 d, all-time), so a
# transaction never sees the future; node features are rates computed on the graph that was
# observable at each split.

# %%
pipeline = aml.FeaturePipeline().fit(df, masks["train"])
features = pipeline.transform_splits(df, masks)
print(
    "edge features:", features["train"].edge.shape, "| node features:", features["train"].node.shape
)
features["train"].edge_cols[:12]

# %%
from aml_graphminer.detectors.bundle import TabularDetector

detector = TabularDetector.fit(df, "hgb", seed=0)
detector.describe()

# %% [markdown]
# `metrics` are computed on the **test** split with the threshold chosen on **validation**.
# On a ~2 % positive-rate dataset prefer PR-AUC / recall@k over accuracy or ROC-AUC.

# %%
pd.Series(detector.metrics).round(4)

# %%
test = df[masks["test"]]
scores = detector.score(test, context=df[~masks["test"]])  # history is context, not scored
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].hist(scores[test["label"] == 0], bins=40, alpha=0.7, label="legitimate", log=True)
ax[0].hist(scores[test["label"] == 1], bins=40, alpha=0.7, label="laundering", log=True)
ax[0].axvline(detector.threshold, color="k", ls="--", lw=1)
ax[0].set(xlabel="score", ylabel="transactions (log)", title="Score distribution (test)")
ax[0].legend()
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_predictions(test["label"], scores, ax=ax[1])
ax[1].set_title("Precision-recall (test)")
plt.tight_layout()

# %% [markdown]
# ## 5. Explanations an investigator can audit
#
# Two complementary tools:
#
# * **Pattern evidence** — exact: the alert that contains the transaction and the numbers
#   behind it (no approximation).
# * **Occlusion** — model-agnostic: remove each neighbouring transaction and see how the score
#   moves. It works with *any* detector (tabular or GNN).

# %%
from aml_graphminer.explainers import (
    OcclusionExplainer,
    TransactionIndex,
    detector_score_fn,
    explain_with_patterns,
    plot_explanation,
)

alert = result.alerts[0]
tx_id = int(alert.tx_ids[0])
explanation = explain_with_patterns(df, tx_id, result)
print(explanation.to_markdown())

# %%
plot_explanation(explanation)
plt.show()

# %%
index = TransactionIndex(df)
flagged = scores[scores >= detector.threshold].sort_values(ascending=False)
target = int(flagged.index[0])
occlusion = OcclusionExplainer(
    detector_score_fn(detector), hops=2, max_edges=40, threshold=detector.threshold
).explain(df, target, index)
print(occlusion.to_markdown())

# %%
plot_explanation(occlusion)
plt.show()

# %% [markdown]
# Every explanation is plain data: `explanation.to_json()`, `explanation.to_networkx()`.

# %%
print(occlusion.to_json()[:600], "...")

# %% [markdown]
# ## 6. Persist and reuse
#
# A saved detector is a directory (`config.json` + weights) that contains the feature
# statistics, so scoring new data needs no re-fitting. The same directory can be exported to
# the Hugging Face Hub with `amlgm export-hub`.

# %%
import tempfile

from aml_graphminer.detectors.bundle import load_detector

with tempfile.TemporaryDirectory() as tmp:
    detector.save(tmp)
    restored = load_detector(tmp)
    same = (restored.score(test, context=df[~masks["test"]]) - scores).abs().max()
print("max |score difference| after reload:", float(same))

# %% [markdown]
# ## 7. Risk communities
#
# Besides the six motifs, the scanner looks for **risk communities**: small groups organised
# around *relay* accounts (accounts that forward most of what they receive) whose money mostly
# stays inside the group within a few days — rings of mule accounts. Groups are connected
# components of the relay transactions, split with Louvain when they are too large.

# %%
communities = [a for a in result.alerts if a.pattern == "community"]
print(f"{len(communities)} risk communities")
if communities:
    strongest = max(communities, key=lambda a: a.score)
    print(strongest.evidence)
pattern_recall_table(df, scanner.scan(df)).round(3)

# %% [markdown]
# ## 8. Learning without labels: the graph auto-encoder
#
# Confirmed laundering cases are rare and arrive months later. `AutoEncoderDetector` (CLI:
# `amlgm train --model gae`) is trained **without any label**: a GNN encodes the accounts and two
# decoders try to rebuild every transaction (its features and the link itself). Transactions it
# rebuilds badly are unusual; the score is their percentile in the training distribution.
# Labels, when present, are only used to choose the alert threshold. Needs the `gnn` extra.

# %%
from aml_graphminer.utils import has_module

if has_module("torch") and has_module("torch_geometric"):
    from aml_graphminer.detectors.gnn import AutoEncoderDetector, GNNConfig

    gae = AutoEncoderDetector.fit(
        df, GNNConfig(kind="sage", hidden=32, epochs=3, batch_size=1024), seed=0
    )
    base_rate = df.loc[masks["test"], "label"].mean()
    print(
        f"unsupervised GAE — test PR-AUC {gae.metrics['pr_auc']:.3f} "
        f"(a random ranking scores {base_rate:.3f}), ROC-AUC {gae.metrics['roc_auc']:.3f}"
    )
else:
    print("PyTorch Geometric is not installed here: the auto-encoder is skipped.")

# %% [markdown]
# ## 9. The web investigation interface
#
# Everything above is also available in a browser:
#
# ```bash
# amlgm serve --demo                 # demo detector + synthetic data, http://127.0.0.1:8000/ui/
# amlgm serve runs/hgb --data data/Synth-Small_Trans.csv
# ```
#
# The interface shows the dashboard of a dataset, the motif alerts with their sub-graph, every
# transaction with its score and an explanation drawn as a graph, and the profile of any account.
# The transaction graph can also be explored in Neo4j: `amlgm export-neo4j <file> <dir>`.

# %% [markdown]
# ## Next steps
#
# * `02_ibm_aml_benchmark` — EDA and the benchmark (motifs vs. tabular vs. GCN/SAGE/GAT vs. the
#   unsupervised auto-encoder) on the IBM AML datasets, with several seeds and confidence intervals.
# * `03_explainability` — GNNExplainer, PGExplainer, Captum, attention weights and how to read an
#   explanation.
# * `amlgm --help` — the same workflow from the command line.
