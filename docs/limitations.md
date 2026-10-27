# Limitations and responsible use

**Data.** The IBM datasets are synthetic: injected laundering motifs are easier to recover than
real laundering, and the labels are exact, which real labels never are (real labels come from
investigated cases and are biased towards what past systems flagged). Elliptic is a Bitcoin graph
with anonymised features; conclusions do not transfer automatically to bank data.

**Detectors.** The motif thresholds of `configs/patterns.yaml` were tuned on the validation
splits of IBM AML HI-Small and LI-Small (the package defaults were chosen for the synthetic
generator). Even tuned, the motif scanner reaches a test PR-AUC of only 0.005 (HI-Small) and 0.001
(LI-Small), far below the learned models (`benchmarks/results/REPORT.md`): only 62 % (HI) and 29 %
(LI) of the laundering transactions belong to an annotated motif, the IBM motifs span several
days and change currency at almost every hop (amounts are conserved only after conversion, which
the detectors do not perform), and bipartite / stack / random motifs have no dedicated detector.
Learned models inherit the biases of their labels. The cycle search is bounded by
budgets and can be incomplete on dense graphs (a warning is logged and `truncated` is set). The
risk-community detector ignores busy accounts (`max_member_degree`) and only sees groups whose
internal flows fit in `max_span`; schemes that reuse very active accounts are left to the other
detectors. The unsupervised auto-encoder flags what is *unusual*, which is not the same as what
is illicit: unusual but legitimate behaviour (a one-off large purchase) scores high too, and the
auto-encoder learns to reconstruct anomalies if it is trained for many epochs.

**Interfaces.** The web interface keeps the analysed tables in server memory (5 investigations,
2 million transactions each) and has no user accounts; it is meant for a single analyst on a
workstation or behind an authenticating proxy. GNN scores use a fixed sampling seed (or full
neighbourhoods with `inference: full`), so they are reproducible, but the score of a transaction
still depends on the transactions sent with it as context.

**Explanations.** Local and approximate: occlusion measures a *score change*, not a cause;
GNNExplainer, PGExplainer, Captum attributions and attention are known to disagree with each
other. Check fidelity before relying on them (see {doc}`explainability`).

**Optional components.** The code that needs an extra (GNN training and Lightning, the
GNNExplainer / attention adapters, the FastAPI wrapper, Parquet / Polars input, MLflow, the
Hugging Face push) is tested only when the extra is installed; without it the corresponding tests
are skipped, so a plain `pytest` run on a minimal install does not cover them. The continuous
integration runs them in dedicated jobs (see `CONTRIBUTING.md`). The notebooks `01`–`03` run on
synthetic data; `04` needs a GPU runtime and the IBM files. Report problems together with the
exact versions of PyTorch and PyG.

**Responsible use.**

* Scores prioritise human review; they are not accusations. Keep analysts in the loop and
  document the decision, not just the score.
* Check error rates across customer groups before any deployment; the toolkit does not do
  fairness auditing for you.
* Transaction data are personal data: apply the retention, minimisation and access rules of your
  jurisdiction, do not log request bodies, and do not upload real customer data to third-party
  services (including public notebooks and Hub repositories).
* The package is research software under the MIT licence **without warranty**; it is not a
  compliance product.
