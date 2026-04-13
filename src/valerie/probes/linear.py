"""Linear probe definitions."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from valerie.config import ProbeConfig


def build_logistic_regression_probe(config: ProbeConfig) -> Pipeline:
    """Build the default sklearn probe used in phase 4."""
    classifier = LogisticRegression(
        C=config.regularization,
        max_iter=config.max_iter,
        solver=config.solver,
        random_state=config.random_seed,
    )
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
