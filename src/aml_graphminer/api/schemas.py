"""Pydantic request / response models of the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransactionIn(BaseModel):
    """One transaction in canonical form (IBM AML column names are also accepted by the service)."""

    model_config = ConfigDict(extra="allow")
    # explicit type of the extra fields (IBM column names, custom attributes); also makes the model
    # importable by Sphinx autodoc, which cannot resolve pydantic's inherited annotation
    __pydantic_extra__: dict[str, Any] = Field(init=False)

    timestamp: str = Field(examples=["2022-09-01 10:05:00"])
    sender: str = Field(examples=["70_ACC_A"])
    receiver: str = Field(examples=["12_ACC_C"])
    amount: float = Field(ge=0, examples=[9500.0])
    currency: str = "US Dollar"
    transaction_type: str = "unknown"
    amount_received: float | None = None
    currency_received: str | None = None
    sender_bank: str = ""
    receiver_bank: str = ""


RECORDS_HELP = (
    "Transactions with the canonical column names (schema TransactionIn) or the IBM AML ones; "
    "every record is validated with the Pydantic Transaction model."
)


class ScoreRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1, description=RECORDS_HELP)
    context: list[dict[str, Any]] | None = Field(
        default=None, description="Earlier transactions used as graph context (not scored)."
    )
    threshold: float | None = Field(default=None, ge=0, le=1)


class ScoredTransaction(BaseModel):
    row: int = Field(description="Position of the transaction in the request.")
    tx_id: int
    score: float
    alert: bool


class ScoreResponse(BaseModel):
    threshold: float
    n_transactions: int
    n_alerts: int
    results: list[ScoredTransaction]


class ExplainRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1, description=RECORDS_HELP)
    row: int = Field(ge=0, description="Position of the transaction to explain.")
    method: str = Field(
        default="auto",
        description="auto, patterns, occlusion, gnnexplainer, pgexplainer, captum or attention.",
    )
    hops: int | None = Field(default=None, ge=1, le=4)
    max_edges: int = Field(default=80, ge=5, le=500)


class ScanRequest(BaseModel):
    transactions: list[dict[str, Any]] = Field(min_length=1, description=RECORDS_HELP)
    top: int = Field(default=20, ge=1, le=500)


class HealthResponse(BaseModel):
    status: str
    library_version: str
    model: dict[str, Any]
    explain_methods: list[str] = Field(
        default_factory=list, description="Explanation methods available for the loaded model."
    )


__all__ = [
    "ExplainRequest",
    "HealthResponse",
    "ScanRequest",
    "ScoreRequest",
    "ScoreResponse",
    "ScoredTransaction",
    "TransactionIn",
]
