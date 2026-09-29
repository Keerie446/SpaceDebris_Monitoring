from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RISK_LABELS = ["Low", "Medium", "High"]


def compute_risk_score(
    df: pd.DataFrame,
    *,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """
    Pseudo risk score from risk factors.
    This is used to create Low/Medium/High labels and to train supervised models.
    """
    if weights is None:
        weights = {
            "crowding": 2.0,
            "altitude_km": 0.6,
            "velocity_km_s": 1.2,
            "object_size": 1.0,
            "decay_status": 0.8,
        }

    # Normalize each component to roughly [0,1] by robust scaling.
    def robust_norm(x: pd.Series) -> pd.Series:
        x = pd.to_numeric(x, errors="coerce")
        q1, q3 = x.quantile([0.25, 0.75])
        denom = (q3 - q1) if (q3 - q1) != 0 else 1.0
        return (x - q1) / denom

    comp = {
        "crowding": robust_norm(df["crowding"]) if "crowding" in df else 0.0,
        "altitude_km": robust_norm(df["altitude_km"]),
        "velocity_km_s": robust_norm(df["velocity_km_s"]),
        "object_size": robust_norm(df["object_size"]),
        "decay_status": robust_norm(df["decay_status"]),
    }

    score = (
        weights["crowding"] * comp["crowding"]
        + weights["altitude_km"] * comp["altitude_km"]
        + weights["velocity_km_s"] * comp["velocity_km_s"]
        + weights["object_size"] * comp["object_size"]
        + weights["decay_status"] * comp["decay_status"]
    )

    # Shift and scale into a cleaner range for thresholding.
    score = score.replace([np.inf, -np.inf], np.nan).fillna(score.median())
    score = (score - score.min()) / (score.max() - score.min() + 1e-12)
    return score


def label_risk(score: pd.Series, *, low_q: float = 0.33, high_q: float = 0.66) -> pd.Series:
    """
    Map a continuous score into Low/Medium/High using rank-percentiles.

    Using rank-percentiles avoids degenerate cases where many values are equal
    (which can cause quantile thresholds to collapse and produce fewer classes).
    """
    s = score.copy()
    if len(s) == 0:
        return pd.Series(index=score.index, dtype="object")
    n = len(s)
    if n == 1:
        return pd.Series(["Low"], index=score.index, dtype="object")

    # percent_rank in [0,1]
    ranks = s.rank(method="first")
    percent_rank = (ranks - 1) / (n - 1)

    labels = pd.Series(index=score.index, dtype="object")
    labels[percent_rank <= low_q] = "Low"
    labels[(percent_rank > low_q) & (percent_rank <= high_q)] = "Medium"
    labels[percent_rank > high_q] = "High"
    return labels


@dataclass(frozen=True)
class RiskModelArtifacts:
    rf: Pipeline
    logreg: Pipeline | None
    kmeans: KMeans
    score_quantiles: dict[str, float]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    class_to_index: dict[str, int]
    class_order: list[str]


def train_risk_models(
    df: pd.DataFrame,
    *,
    kmeans_k: int = 5,
    low_q: float = 0.33,
    high_q: float = 0.66,
) -> tuple[RiskModelArtifacts, pd.DataFrame]:
    required = ["altitude_km", "velocity_km_s", "inclination_deg", "object_size", "orbit_type", "decay_status", "crowding"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for ML training: {missing}")

    # Pseudo-label generation.
    risk_score = compute_risk_score(df)
    risk_label = label_risk(risk_score, low_q=low_q, high_q=high_q)

    df_train = df.copy()
    df_train["risk_score"] = risk_score
    df_train["risk_label"] = risk_label

    df_train = df_train.dropna(subset=["risk_score", "risk_label"])

    numeric_features = ["altitude_km", "velocity_km_s", "inclination_deg", "object_size", "decay_status", "crowding"]
    categorical_features = ["orbit_type"]
    feature_columns = numeric_features + categorical_features

    class_order = RISK_LABELS
    class_to_index = {c: i for i, c in enumerate(class_order)}
    y = df_train["risk_label"].map(class_to_index).astype(int)

    # Preprocess: numeric scaling + orbit_type one-hot.
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )

    rf = Pipeline(
        steps=[
            ("pre", pre),
            ("model", RandomForestClassifier(
                n_estimators=250,
                random_state=42,
                class_weight="balanced",
            )),
        ]
    )

    logreg = Pipeline(
        steps=[
            ("pre", pre),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                multi_class="multinomial",
            )),
        ]
    )

    rf.fit(df_train[feature_columns], y)

    # Logistic regression requires at least 2 classes.
    if y.nunique() >= 2:
        logreg.fit(df_train[feature_columns], y)
    else:
        logreg = None

    # KMeans clustering for "orbital clustering visualization"
    kmeans_feats = ["altitude_km", "inclination_deg", "velocity_km_s", "object_size", "crowding"]
    kmeans = KMeans(n_clusters=kmeans_k, random_state=42, n_init="auto")
    kmeans.fit(df_train[kmeans_feats])

    artifacts = RiskModelArtifacts(
        rf=rf,
        logreg=logreg,
        kmeans=kmeans,
        score_quantiles={"low": float(risk_score.quantile(low_q)), "high": float(risk_score.quantile(high_q))},
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        class_to_index=class_to_index,
        class_order=class_order,
    )

    return artifacts, df_train


def predict_risk(
    artifacts: RiskModelArtifacts,
    df_pred: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns risk label + probabilities (RF + Logistic regression averaged).
    """
    # Ensure columns exist.
    for c in artifacts.numeric_features:
        if c not in df_pred.columns:
            df_pred[c] = np.nan
    for c in artifacts.categorical_features:
        if c not in df_pred.columns:
            df_pred[c] = "Unknown"

    X = df_pred[artifacts.feature_columns]

    def proba_to_fixed(proba: np.ndarray, classes: np.ndarray, n_classes: int) -> np.ndarray:
        """
        Expand model probabilities into a fixed [Low, Medium, High] order,
        even when the training set only contained a subset of classes.
        """
        fixed = np.zeros((proba.shape[0], n_classes), dtype=float)
        for j, cls in enumerate(classes):
            if 0 <= int(cls) < n_classes:
                fixed[:, int(cls)] = proba[:, j]
        return fixed

    rf_model = artifacts.rf.named_steps["model"]
    rf_classes = getattr(rf_model, "classes_", np.array([], dtype=int))

    rf_proba_raw = artifacts.rf.predict_proba(X)

    n_classes = len(artifacts.class_order)
    rf_proba = proba_to_fixed(rf_proba_raw, rf_classes, n_classes)

    if artifacts.logreg is None:
        proba = rf_proba
    else:
        log_model = artifacts.logreg.named_steps["model"]
        log_classes = getattr(log_model, "classes_", np.array([], dtype=int))
        log_proba_raw = artifacts.logreg.predict_proba(X)
        log_proba = proba_to_fixed(log_proba_raw, log_classes, n_classes)
        proba = 0.5 * (rf_proba + log_proba)

    idx = proba.argmax(axis=1)
    labels = [artifacts.class_order[i] for i in idx]

    out = df_pred.copy()
    out["pred_risk_label"] = labels
    for i, label in enumerate(artifacts.class_order):
        out[f"p_{label.lower()}"] = proba[:, i]
    return out

