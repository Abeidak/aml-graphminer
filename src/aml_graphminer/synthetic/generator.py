"""Synthetic IBM-AML-like transaction generator.

Why a generator?
    * The CI, the unit tests and the quick-start notebook must run in seconds without
      downloading the multi-GB IBM files;
    * the project brief (fiche de sujet) asks for a *realistic synthetic data generator*;
    * it exposes **ground-truth motif labels** (``pattern`` / ``attempt_id``) so that the
      detectors can be validated motif by motif.

The output follows the canonical schema (:mod:`aml_graphminer.ingestion.schema`) plus an
``attempt_id`` column (``-1`` for legitimate traffic).

.. warning::
   Results obtained on this generator say **nothing** about performance on real data:
   detectors and generator are written by the same team and share the same motif
   vocabulary. Use the IBM AML / Elliptic benchmarks for any scientific claim.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..ingestion.schema import validate_transactions

PATTERN_TYPES: tuple[str, ...] = (
    "fan_out",
    "fan_in",
    "cycle",
    "scatter_gather",
    "gather_scatter",
    "smurfing",
)
PAYMENT_FORMATS = ("Cheque", "Credit Card", "ACH", "Cash", "Wire", "Reinvestment", "Bitcoin")
_FORMAT_P = np.array([0.12, 0.30, 0.22, 0.12, 0.10, 0.10, 0.04])
_FORMAT_SCALE = {  # median amount multiplier by payment format
    "Cheque": 1.2,
    "Credit Card": 0.25,
    "ACH": 1.0,
    "Cash": 0.35,
    "Wire": 3.0,
    "Reinvestment": 0.8,
    "Bitcoin": 1.5,
}
CURRENCIES = ("US Dollar", "Euro", "Yuan", "Rupee", "Yen", "Swiss Franc", "Canadian Dollar")
_CURRENCY_P = np.array([0.55, 0.15, 0.08, 0.06, 0.06, 0.05, 0.05])
_FX = {  # crude units-of-currency per USD (only used to make Amount Received differ)
    "US Dollar": 1.0,
    "Euro": 0.92,
    "Yuan": 7.2,
    "Rupee": 83.0,
    "Yen": 150.0,
    "Swiss Franc": 0.88,
    "Canadian Dollar": 1.36,
}
_HOURLY = np.array(
    [1, 1, 1, 1, 1, 2, 3, 5, 7, 8, 8, 8, 8, 8, 8, 8, 7, 6, 5, 4, 3, 2, 2, 1], dtype="float64"
)
_HOURLY = _HOURLY / _HOURLY.sum()
_ILLICIT_FORMATS = ("Wire", "ACH", "Cash", "Cheque", "Bitcoin")
_ROW = tuple  # (ts_seconds, sender, receiver, amount, currency, fmt, label, pattern, attempt)


class _Accounts:
    """Pool of accounts with heavy-tailed activity, plus a factory for fresh accounts."""

    def __init__(self, rng: np.random.Generator, n_accounts: int, n_banks: int) -> None:
        self.rng = rng
        self.n_banks = n_banks
        self.counter = 0
        self.ids = np.array([self._new() for _ in range(n_accounts)], dtype=object)
        weights = rng.pareto(1.3, n_accounts) + 1.0
        self.send_p = weights / weights.sum()
        recv = weights**0.8 * rng.uniform(0.6, 1.4, n_accounts)
        self.recv_p = recv / recv.sum()

    def _new(self) -> str:
        bank = int(self.rng.integers(1, self.n_banks + 1))
        acc = f"{self.counter:07X}{int(self.rng.integers(0, 16)):X}"
        self.counter += 1
        return f"{bank}_{acc}"

    def fresh(self) -> str:
        return self._new()

    def existing(self) -> str:
        return str(self.rng.choice(self.ids, p=self.send_p))

    def mixed(self, p_existing: float = 0.5) -> str:
        return self.existing() if self.rng.random() < p_existing else self.fresh()

    def distinct(
        self, k: int, p_existing: float = 0.5, exclude: set[str] | None = None
    ) -> list[str]:
        out: list[str] = []
        seen = set(exclude or ())
        while len(out) < k:
            acc = self.mixed(p_existing)
            if acc not in seen:
                seen.add(acc)
                out.append(acc)
        return out


def _bank(account: str) -> str:
    return account.split("_", 1)[0]


class _Generator:
    def __init__(self, seed: int, n_accounts: int, n_banks: int, days: float, start: str) -> None:
        self.rng = np.random.default_rng(seed)
        self.accounts = _Accounts(self.rng, n_accounts, n_banks)
        self.days = days
        self.t0 = pd.Timestamp(start)
        self.rows: list[_ROW] = []
        self.attempt = 0

    # ------------------------------------------------------------------ helpers
    def _time(self, size: int | None = None) -> np.ndarray | float:
        """Random time (seconds from start) with a diurnal profile, minute resolution."""
        n = 1 if size is None else size
        day = self.rng.uniform(0, max(self.days - 1, 0.001), n).astype(int)
        hour = self.rng.choice(24, size=n, p=_HOURLY)
        minute = self.rng.integers(0, 60, n)
        sec = day * 86400.0 + hour * 3600.0 + minute * 60.0
        return float(sec[0]) if size is None else sec

    def _amount(self, fmt: str, median: float = 900.0) -> float:
        value = float(self.rng.lognormal(np.log(median * _FORMAT_SCALE[fmt]), 1.25))
        return round(min(max(value, 1.0), 5e6), 2)

    def _emit(
        self,
        ts: float,
        sender: str,
        receiver: str,
        amount: float,
        fmt: str,
        label: int = 0,
        pattern: str = "",
        attempt: int = -1,
        currency: str | None = None,
    ) -> None:
        cur = currency or str(self.rng.choice(CURRENCIES, p=_CURRENCY_P))
        self.rows.append(
            (
                float(int(ts // 60) * 60),
                sender,
                receiver,
                round(float(amount), 2),
                cur,
                fmt,
                label,
                pattern,
                attempt,
            )
        )

    # ------------------------------------------------------------------ background
    def background(self, n: int) -> None:
        acc = self.accounts
        src = self.rng.choice(len(acc.ids), size=n, p=acc.send_p)
        dst = self.rng.choice(len(acc.ids), size=n, p=acc.recv_p)
        clash = src == dst
        dst[clash] = (dst[clash] + self.rng.integers(1, len(acc.ids), clash.sum())) % len(acc.ids)
        times = np.asarray(self._time(n))
        fmts = np.asarray(self.rng.choice(PAYMENT_FORMATS, size=n, p=_FORMAT_P), dtype=object)
        for i in range(n):
            fmt = str(fmts[i])
            self._emit(float(times[i]), acc.ids[src[i]], acc.ids[dst[i]], self._amount(fmt), fmt)

    def hard_negatives(self, n_background: int) -> None:
        """Legitimate traffic that *looks* like a laundering motif (drives false positives)."""
        acc, rng = self.accounts, self.rng
        scale = max(1, n_background // 4000)
        for _ in range(scale):  # payroll: legitimate fan-out
            employer = acc.existing()
            t = float(self._time())
            base = float(rng.uniform(1200, 4500))
            for emp in acc.distinct(int(rng.integers(6, 20)), p_existing=1.0, exclude={employer}):
                self._emit(
                    t + rng.uniform(0, 1800), employer, emp, base * rng.uniform(0.8, 1.25), "ACH"
                )
        for _ in range(scale):  # merchant bursts: legitimate fan-in
            merchant = acc.existing()
            t = float(self._time())
            for cust in acc.distinct(int(rng.integers(8, 25)), p_existing=1.0, exclude={merchant}):
                self._emit(
                    t + rng.uniform(0, 7200), cust, merchant, rng.uniform(5, 180), "Credit Card"
                )
        for _ in range(max(1, n_background // 8000)):  # legitimate triangles (amounts unrelated)
            a, b, c = acc.distinct(3, p_existing=1.0)
            t = float(self._time())
            for s, r in ((a, b), (b, c), (c, a)):
                t += rng.uniform(3600, 86400)
                self._emit(t, s, r, rng.uniform(40, 3000), "ACH")
        for _ in range(scale * 3):  # refunds: 2-cycles
            a, b = acc.distinct(2, p_existing=1.0)
            t = float(self._time())
            amount = rng.uniform(20, 800)
            self._emit(t, a, b, amount, "Credit Card")
            self._emit(t + rng.uniform(3600, 172800), b, a, amount, "Credit Card")

    # ------------------------------------------------------------------ laundering motifs
    def _value(self) -> float:
        return float(np.clip(self.rng.lognormal(np.log(40000), 1.0), 3000, 2e6))

    def _new_attempt(self) -> int:
        self.attempt += 1
        return self.attempt - 1

    def fan_out(self) -> None:
        rng, acc = self.rng, self.accounts
        aid, src = self._new_attempt(), acc.mixed(0.5)
        mules = acc.distinct(int(rng.integers(4, 13)), 0.35, {src})
        fmt, cur = str(rng.choice(_ILLICIT_FORMATS)), str(rng.choice(CURRENCIES, p=_CURRENCY_P))
        t, value = float(self._time()), self._value()
        window = float(rng.uniform(600, 3 * 3600))
        for m in mules:
            amt = value / len(mules) * rng.uniform(0.8, 1.2)
            self._emit(t + rng.uniform(0, window), src, m, amt, fmt, 1, "fan_out", aid, cur)

    def fan_in(self) -> None:
        rng, acc = self.rng, self.accounts
        aid, sink = self._new_attempt(), acc.mixed(0.5)
        senders = acc.distinct(int(rng.integers(4, 13)), 0.35, {sink})
        fmt, cur = str(rng.choice(_ILLICIT_FORMATS)), str(rng.choice(CURRENCIES, p=_CURRENCY_P))
        t, value = float(self._time()), self._value()
        window = float(rng.uniform(600, 3 * 3600))
        for s in senders:
            amt = value / len(senders) * rng.uniform(0.8, 1.2)
            self._emit(t + rng.uniform(0, window), s, sink, amt, fmt, 1, "fan_in", aid, cur)

    def cycle(self) -> None:
        rng, acc = self.rng, self.accounts
        aid = self._new_attempt()
        length = int(rng.integers(3, 7))
        nodes = acc.distinct(length, 0.35)
        fmt, cur = str(rng.choice(_ILLICIT_FORMATS)), str(rng.choice(CURRENCIES, p=_CURRENCY_P))
        t, amount = float(self._time()), self._value()
        for i in range(length):
            t += float(rng.uniform(300, 6 * 3600))
            self._emit(t, nodes[i], nodes[(i + 1) % length], amount, fmt, 1, "cycle", aid, cur)
            amount *= float(rng.uniform(0.97, 0.995))

    def scatter_gather(self) -> None:
        rng, acc = self.rng, self.accounts
        aid = self._new_attempt()
        src, sink = acc.distinct(2, 0.5)
        mids = acc.distinct(int(rng.integers(3, 9)), 0.2, {src, sink})
        fmt, cur = str(rng.choice(_ILLICIT_FORMATS)), str(rng.choice(CURRENCIES, p=_CURRENCY_P))
        t, value = float(self._time()), self._value()
        for m in mids:
            t1 = t + float(rng.uniform(0, 3600))
            amt = value / len(mids) * float(rng.uniform(0.9, 1.1))
            self._emit(t1, src, m, amt, fmt, 1, "scatter_gather", aid, cur)
            self._emit(
                t1 + float(rng.uniform(600, 6 * 3600)),
                m,
                sink,
                amt * float(rng.uniform(0.95, 0.99)),
                fmt,
                1,
                "scatter_gather",
                aid,
                cur,
            )

    def gather_scatter(self) -> None:
        rng, acc = self.rng, self.accounts
        aid, hub = self._new_attempt(), acc.mixed(0.3)
        k_in, k_out = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        parties = acc.distinct(k_in + k_out, 0.3, {hub})
        sources, targets = parties[:k_in], parties[k_in:]
        fmt, cur = str(rng.choice(_ILLICIT_FORMATS)), str(rng.choice(CURRENCIES, p=_CURRENCY_P))
        t, value = float(self._time()), self._value()
        total = 0.0
        t_last = t
        for s in sources:
            amt = value / k_in * float(rng.uniform(0.85, 1.15))
            ts = t + float(rng.uniform(0, 4 * 3600))
            t_last = max(t_last, ts)
            total += amt
            self._emit(ts, s, hub, amt, fmt, 1, "gather_scatter", aid, cur)
        for g in targets:
            amt = total * float(rng.uniform(0.95, 0.99)) / k_out
            ts = t_last + float(rng.uniform(900, 6 * 3600))
            self._emit(ts, hub, g, amt, fmt, 1, "gather_scatter", aid, cur)

    def smurfing(self) -> None:
        rng, acc = self.rng, self.accounts
        aid, placer = self._new_attempt(), acc.mixed(0.5)
        k = int(rng.integers(6, 16))
        mules = acc.distinct(k, 0.2, {placer})
        fmt = str(rng.choice(["Cash", "Wire", "ACH"]))
        cur = "US Dollar"
        t = float(self._time())
        for m in mules:
            self._emit(
                t + float(rng.uniform(0, 12 * 3600)),
                placer,
                m,
                float(rng.uniform(7500, 9990)),
                fmt,
                1,
                "smurfing",
                aid,
                cur,
            )

    def finalise(self) -> pd.DataFrame:
        cols = [
            "ts_sec",
            "sender",
            "receiver",
            "amount",
            "currency_received",
            "transaction_type",
            "label",
            "pattern",
            "attempt_id",
        ]
        frame = pd.DataFrame(self.rows, columns=cols)
        frame["timestamp"] = self.t0 + pd.to_timedelta(frame.pop("ts_sec"), unit="s")
        frame["sender_bank"] = frame["sender"].map(_bank)
        frame["receiver_bank"] = frame["receiver"].map(_bank)
        # Paid currency is the sender-side currency.
        paid = frame["currency_received"].to_numpy(dtype=object).copy()
        # Cross-currency payments: ~18 % of legitimate traffic but only ~4 % of laundering
        # chains (a mule re-sends what it received, in the same currency).
        cross_p = np.where(frame["label"].to_numpy() == 1, 0.04, 0.18)
        cross = self.rng.random(len(frame)) < cross_p
        paid[cross] = self.rng.choice(CURRENCIES, size=int(cross.sum()), p=_CURRENCY_P)
        frame["currency"] = paid
        fx_paid = frame["currency"].map(_FX).to_numpy(dtype="float64")
        fx_recv = frame["currency_received"].map(_FX).to_numpy(dtype="float64")
        frame["amount_received"] = (frame["amount"].to_numpy() * fx_recv / fx_paid).round(2)
        return frame


def generate_transactions(
    n_accounts: int = 1500,
    n_background: int = 15000,
    *,
    n_banks: int = 12,
    days: float = 10.0,
    illicit_ratio: float = 0.01,
    patterns: Mapping[str, int] | None = None,
    seed: int = 0,
    start: str = "2022-09-01",
    hard_negatives: bool = True,
) -> pd.DataFrame:
    """Generate an IBM-AML-like table with injected laundering motifs.

    Parameters
    ----------
    n_accounts:
        Size of the base account population (patterns add fresh accounts on top).
    n_background:
        Number of legitimate background transactions.
    n_banks:
        Number of banks accounts are spread over.
    days:
        Time span; transactions are spread over ``days`` days (IBM Small = 10 days).
    illicit_ratio:
        Approximate fraction of laundering transactions (IBM: 0.1-0.2 %; defaults to 1 % so
        small test sets contain enough positives). Ignored when ``patterns`` is given.
    patterns:
        Explicit number of instances per motif, e.g. ``{"cycle": 5, "fan_in": 10}``. Valid
        names: :data:`PATTERN_TYPES`.
    seed:
        Random seed (fully deterministic output).
    hard_negatives:
        Add legitimate payroll bursts, merchant bursts, unrelated triangles and refunds that
        resemble motifs (so detectors are not trivially perfect).

    Returns
    -------
    pandas.DataFrame
        Canonical transaction table plus ``attempt_id``. ``result.attrs["injected"]`` holds
        the number of injected instances per motif.
    """
    if patterns is None:
        avg_size = {
            "fan_out": 8,
            "fan_in": 8,
            "cycle": 4.5,
            "scatter_gather": 11,
            "gather_scatter": 11,
            "smurfing": 10.5,
        }
        mean_size = float(np.mean(list(avg_size.values())))
        total = max(len(PATTERN_TYPES), round(illicit_ratio * n_background / mean_size))
        base, extra = divmod(total, len(PATTERN_TYPES))
        patterns = {p: base + (1 if i < extra else 0) for i, p in enumerate(PATTERN_TYPES)}
    unknown = set(patterns) - set(PATTERN_TYPES)
    if unknown:
        raise ValueError(f"unknown pattern(s) {sorted(unknown)}; valid: {PATTERN_TYPES}")

    gen = _Generator(seed, n_accounts, n_banks, days, start)
    gen.background(n_background)
    if hard_negatives:
        gen.hard_negatives(n_background)
    for name, count in patterns.items():
        builder = getattr(gen, name)
        for _ in range(int(count)):
            builder()

    frame = gen.finalise()
    out = validate_transactions(frame)
    out["attempt_id"] = out["attempt_id"].astype("int64")
    out.attrs["injected"] = {k: int(v) for k, v in patterns.items()}
    return out


def write_ibm_files(
    df: pd.DataFrame, directory: str | Path, name: str = "Synth-Small"
) -> tuple[Path, Path]:
    """Write ``<name>_Trans.csv`` and ``<name>_Patterns.txt`` in the IBM AML layout.

    Used by the tests and demos to exercise the real-data code path (``schema='ibm'``,
    pattern attachment) without the original files.
    """
    from ..ingestion.readers import write_ibm_csv

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    trans = write_ibm_csv(df, directory / f"{name}_Trans.csv")
    patterns_path = directory / f"{name}_Patterns.txt"
    with patterns_path.open("w", encoding="utf-8") as handle:
        laundering = df[df["label"] == 1.0]
        for _attempt_id, group in laundering.groupby("attempt_id", sort=True):
            kind = str(group["pattern"].iloc[0]).upper().replace("_", "-")
            handle.write(f"BEGIN LAUNDERING ATTEMPT - {kind}\n")
            for _, r in group.sort_values("timestamp").iterrows():
                handle.write(
                    ",".join(
                        [
                            r["timestamp"].strftime("%Y/%m/%d %H:%M"),
                            str(r["sender_bank"]),
                            r["sender"].split("_", 1)[1],
                            str(r["receiver_bank"]),
                            r["receiver"].split("_", 1)[1],
                            f"{r['amount_received']:.2f}",
                            r["currency_received"],
                            f"{r['amount']:.2f}",
                            r["currency"],
                            r["transaction_type"],
                            "1",
                        ]
                    )
                    + "\n"
                )
            handle.write(f"END LAUNDERING ATTEMPT - {kind}\n")
    return trans, patterns_path
