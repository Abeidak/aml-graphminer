# Extending the toolkit

## A new structural-motif detector

A detector receives a `TxArrays` view of the table (integer-coded accounts, sorted timestamps,
amounts, currencies) and returns `Alert`s. Register it and the scanner, the CLI configuration,
the benchmark and the explainers pick it up.

```python
import numpy as np
from aml_graphminer.detectors.patterns import PatternDetector, PatternScanner, register_detector


class VeryLargePayment(PatternDetector):
    name = "very_large_payment"

    def detect(self, ctx):
        rows = np.flatnonzero(ctx.paid > 1_000_000)
        return [
            self.make_alert(
                ctx,
                self.name,
                np.array([r]),
                [int(ctx.src[r])],
                1.0,
                {"amount": float(ctx.paid[r])},
            )
            for r in rows
        ]


register_detector("very_large_payment", lambda config: VeryLargePayment())
result = PatternScanner(only=["very_large_payment"]).scan(df)
```

A custom detector has no section in `PatternConfig`, so it is always enabled when selected. Put
the thresholds in the constructor or read them from your own configuration. Write the
evidence dictionary for an investigator: names and units, not internal indices.

## A new input format

```python
import pandas as pd
from aml_graphminer.ingestion.adapters import Adapter


class MyBankAdapter(Adapter):
    name = "mybank"

    def matches(self, columns):
        return "booking_date" in columns and "iban_from" in columns

    def to_frame(self, raw):
        return pd.DataFrame(
            {
                "timestamp": pd.to_datetime(raw["booking_date"]),
                "sender": raw["iban_from"],
                "receiver": raw["iban_to"],
                "amount": raw["amount_eur"],
                "currency": "EUR",
            }
        )


df = aml.read_transactions("export.csv", schema=MyBankAdapter())
```

`convert()` validates the result with `validate_transactions` (types, negative amounts, missing
values, ordering). Anything not provided gets a documented default.

## A new learned model

Any object with `score(df, context=None)`, `predict`, `save(directory)` and `describe()` can be
benchmarked and explained. To make it loadable through `load_detector`, write a `config.json`
with a `model_type` and add a branch to `aml_graphminer.detectors.bundle.load_detector`.
For the explainers, wrap it with `detector_score_fn(model)`.

## A new explainer

Subclass nothing: build an `Explanation` (edges, nodes, features, patterns, summary) from
whatever signal you compute and return it; `to_markdown`, `to_json`, `to_networkx` and
`plot_explanation` work on it.
