# Structural-motif detectors

The detectors look for laundering structures documented in the AML literature and in the IBM
generator. They need **no training**: every alert lists the accounts and transactions involved
and the numbers that triggered it.

```python
scanner = aml.PatternScanner()  # all detectors, default thresholds
result = scanner.scan(df)  # df: canonical, time-sorted table
result.to_frame()  # one row per alert, best first
result.transaction_scores()  # per-transaction score and strongest motif
result.account_scores()  # per-account maximum alert score
result.to_json("alerts.json")  # alerts + summary
```

| detector | structure | evidence keys |
|---|---|---|
| `fan_in` | many distinct accounts pay one hub in a short window (collection) | `hub, n_counterparties, n_transactions, span_seconds, window_seconds, amount_cv, burst_ratio` |
| `fan_out` | one account pays many distinct accounts in a short window (dispersion) | same as fan-in |
| `smurfing` | *structuring*: many payments just below a reporting threshold; *split*: one large inflow chopped into similar parts | `mode, role, account, n_transactions, threshold, mean_amount, n_counterparties` |
| `cycle` | money returns to its origin through 3–6 accounts, in time order, amounts almost conserved | `length, span_seconds, first_amount, last_amount, amount_retained, cycle` |
| `scatter_gather` | source → *k* intermediaries → sink | `source, sink, n_intermediaries, span_seconds, dispersed_amount` |
| `gather_scatter` | *k* sources → hub → *k′* targets with conserved volume | `hub, n_sources, n_targets, gathered_amount, scattered_amount, conservation_ratio` |
| `relay` (opt-in) | an account forwards, within a few days, 0.9–1.2× what it has just received, in another currency (`A → B → C`; the IBM *stack* motif) | `relay, forwarded_share, delay_seconds, currency_in, currency_out` |
| `community` | a small group organised around relay accounts that keeps most of its money inside within a few days (risk community, ring of mules) | `n_members, n_relays, n_internal_tx, internal_share, mean_flow_through, span_seconds, internal_amount, has_cycle` |

Alert scores are in `[0, 1]` and grow with the strength of the evidence (more counterparties, a
tighter burst, better conservation of the amount ...).

## Risk communities

The six motifs have a fixed shape. Laundering rings often do not, but they share a signature:
**relay accounts** that forward most of what they receive, trading mostly among themselves in a
short time. The `community` detector (`CommunityDetector`) looks for them in three steps:

1. **Relays** — accounts with incoming *and* outgoing money whose flow-through
   `min(in, out) / max(in, out)` is at least `min_flow_through` (0.75). Accounts with more than
   `max_member_degree` transactions (merchants, payroll, bank accounts) are ignored altogether:
   their flows are legitimately balanced and they would glue unrelated accounts together.
2. **Groups** — connected components of the transactions that touch at least one relay (so the
   source and the sink around the relays belong to the group); components larger than
   `max_size` are split into **Louvain communities** (modularity, `resolution`, fixed `seed`).
3. **Risk** — a group of `min_size` to `max_size` accounts with at least `min_relays` relays is an
   alert when at least `min_internal_tx` transactions link its members within `max_span` (3 days)
   and at least `min_internal_share` (60 %) of the members' outgoing volume stays inside it.

The score combines the internal share, the share of relays, the number of internal transactions
per member, the speed of the flows and the presence of a directed cycle. On the synthetic
generator the detector recovers rings, scatter-gather and gather-scatter schemes made of fresh
accounts with very few false alerts; schemes that reuse busy accounts are left to the dedicated
motif detectors and to the learned models. The Louvain partition of a whole graph is also
available directly: `aml_graphminer.graph.communities(build_digraph(df))`.

## Configuration

Each detector has a section in the YAML configuration (`amlgm init-config` writes all of them
with their defaults) and can be switched off with `enabled: false`:

```yaml
patterns:
  fan_in:    {min_counterparties: 8, window: 12h, min_burst_ratio: 4.0}
  smurfing:  {threshold: 10000, band: 0.25, currency: US Dollar}
  cycle:     {max_length: 5, max_span: 3d, min_hop_ratio: 0.85, cross_currency: false}
  community: {min_flow_through: 0.8, max_span: 2d, max_member_degree: 30}
```

```python
config = aml.PatternConfig.from_dict({"fan_in": {"min_counterparties": 8}})
scanner = aml.PatternScanner(config, only=["fan_in", "cycle"])
```

Unknown keys, impossible values and malformed durations (`"24x"`) raise `ConfigError` when the
configuration is loaded, not in the middle of a long scan.

Important parameters:

* `window`, `hop_window`, `max_span` — durations such as `30min`, `24h`, `7d`.
* `min_burst_ratio` (fan-in/out) — how many times the hub's *own* average activity the burst
  must exceed; it keeps payroll accounts and merchants from alerting continuously.
* `min_hop_ratio` / `max_hop_ratio` (cycle, scatter-gather) — allowed fraction of the received
  amount that is forwarded on the next hop (fees, small losses).
* `cross_currency` — a hop between two currencies whose exchange rate is unknown cannot be
  checked, so it is rejected by default (see *Currencies* below).
* `max_cycles`, `max_branching`, `max_expansions` — search budgets. If a budget is exhausted the
  cycle detector logs a warning and sets `detector.truncated = True`; results are then incomplete and
  you should tighten the window rather than ignore the warning.

## Currencies, history and payment formats

Scanner-wide settings sit next to the detector sections:

```yaml
patterns:
  currency: {convert: true, reference: US Dollar, rates: null}
  lookback: 4d
  transaction_types: null        # e.g. [ACH]
  exclude: []                    # self_transfers, same_bank, cross_currency
  combine: max                   # or noisy_or
  calibration_bins: 1            # > 1: precision per band of alert score
```

* **`currency`** — laundering chains often change currency between hops: an account receives
  dollars and forwards euros. Every amount is therefore also expressed in the `reference`
  currency before the cycle, scatter-gather, gather-scatter, split-smurfing and community
  detectors compare or add amounts. The rates are inferred from the table itself: a payment of
  `paid` units of currency *a* received as `recv` units of *b* gives `rate(a) / rate(b) =
  recv / paid`; the median of every exchanged pair is taken and the rates are solved by least
  squares, through intermediate currencies when needed (`exchange_rates`). A currency never
  exchanged with the reference keeps its raw amounts (its hops are not comparable); `rates`
  (units of the reference per unit) overrides the inferred values. Alert amounts
  (`total_amount`, evidence) are in the reference currency. `convert: false` restores the raw
  amounts.
* **`lookback`** — when a time slice is scored (validation and test slices of the benchmark,
  `MotifDetector.fit`, `score` with a context), the transactions of this period before the slice
  are given to the scanner as **unlabelled** context: a 3-day cycle that started before the slice
  is then seen whole. Only the transactions of the slice are scored and only the alerts touching
  them are returned (`PatternScanner.scan(df, history=...)`); the labels of the context are never
  read, so this is what a deployed scanner sees, with no leakage.
* **`transaction_types`** — only the listed payment formats are searched (the others keep a
  score of 0). In the IBM AML simulator every laundering motif is made of ACH transfers
  (`docs/limitations.md`), which is why the tuned `configs/patterns.yaml` sets `[ACH]`; this is a
  property of that simulator, not a general rule, and should be re-examined on other data.
* **`exclude`** — transactions left out of the search: transfers to oneself, between two
  accounts of the same bank (internal transfers, a classic low-risk category), or paid in one
  currency and received in another (never laundering in the IBM data — a simulator property).
  Exchange rates are still inferred from the excluded cross-currency payments.
* **`combine`** — the score of a transaction covered by several alerts: the strongest one
  (`max`) or all of them as independent evidence (`noisy_or`, `1 − Π(1 − weight)`).

Detector options added for the IBM data (all off by default): `fan_in` / `fan_out` take
`min_span` (the fan must spread over at least this duration — IBM laundering fans last days,
legitimate bursts such as payroll minutes) and `min_mean_amount` (in the reference currency);
`gather_scatter` takes `scatter_lead` (the hub may pay out up to this long *before* it is paid,
as it often does in the IBM data); the `relay` section is disabled unless `enabled: true`.

## Calibration

Different detectors have different precision. `scanner.fit_calibration(validation_df)` learns,
on a **labelled validation slice**, the Laplace-smoothed share of illicit transactions covered by
each detector's alerts; calibrated scores (`precision × alert score`) can then be compared across
detectors. With `calibration_bins: k > 1` the precision is learned per detector **and per band
of alert score** (quantile bands of the validation alerts, shrunk towards the detector's own
precision), because inside a detector the alert score does not always rank alerts well — on IBM
HI-Small the best-scored fan-in alerts were the least precise. `MotifDetector.fit` does this on
the validation split automatically and selects the decision threshold there; the bands are saved
with the model.

## Using the scanner as a model

`MotifDetector` wraps a scanner in the same interface as the learned models
(`score`, `predict`, `save`, `load`, `describe`), so a motif detector appears in
`run_benchmark` (`models=["patterns", ...]`), in the CLI (`amlgm train --model patterns`) and in
the REST API.

## Limits

* Detectors find what their definition covers. A laundering scheme that is not a fan, a cycle,
  smurfing, scatter-gather, gather-scatter or a ring of relay accounts is invisible to them — that
  is what the learned models (supervised GNNs and the unsupervised auto-encoder) are for.
* Real transaction graphs contain legitimate look-alikes (payroll, marketplaces, refunds). The
  defaults are a starting point; **tune thresholds on your own validation data** before trusting
  the precision.
* The cycle search is exponential in the worst case; it is bounded by the budgets above.
