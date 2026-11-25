from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import math
import os
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import _build_dashboard_state
from src.feature_engineering.feature_engineer import FEATURE_COLUMNS


CACHE_SCHEMA_VERSION = 1
CACHE_DIRECTORY = Path(__file__).with_name("cache")
CACHE_PATH = CACHE_DIRECTORY / "evaluation_data.pkl"
_RUNTIME_DISTRIBUTIONS = (
    "Faker",
    "numpy",
    "pandas",
    "scikit-learn",
    "shap",
    "xgboost"
    )


def _source_fingerprint():
    digest = sha256()
    digest.update(
        f"cache-schema:{CACHE_SCHEMA_VERSION}\n".encode("utf-8")
        )
    digest.update(f"python:{sys.version}\n".encode("utf-8"))
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            distribution_version = version(distribution)
        except PackageNotFoundError:
            distribution_version = "not-installed"
        digest.update(
            f"{distribution}:{distribution_version}\n".encode("utf-8")
            )

    source_paths = sorted(
        (PROJECT_ROOT / "src").rglob("*.py")
        )
    source_paths.append(PROJECT_ROOT / "requirements.txt")
    for source_path in source_paths:
        relative_path = source_path.relative_to(PROJECT_ROOT)
        digest.update(str(relative_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _build_evaluation_data():
    state = _build_dashboard_state()
    transactions = state["transactions"]
    risk_model = state["risk_model"]
    test_transaction_ids = tuple(risk_model.test_transaction_ids)

    return {
        "features": state["features"].copy(),
        "labels": transactions["is_fraud"].astype("int64").copy(),
        "fraud_types": transactions["fraud_type"].copy(),
        "train_transaction_ids": tuple(
            risk_model.train_transaction_ids
            ),
        "test_transaction_ids": test_transaction_ids,
        "test_risk_scores": risk_model.evaluation_results[
            "risk_score"
            ].reindex(test_transaction_ids).astype("float64").copy()
        }


def _validate_evaluation_data(evaluation_data):
    required_keys = {
        "features",
        "labels",
        "fraud_types",
        "train_transaction_ids",
        "test_transaction_ids",
        "test_risk_scores"
        }
    if (
        not isinstance(evaluation_data, dict)
        or not required_keys.issubset(evaluation_data)
        ):
        raise ValueError("Evaluation cache has an invalid structure")

    features = evaluation_data["features"]
    labels = evaluation_data["labels"]
    fraud_types = evaluation_data["fraud_types"]
    train_ids = evaluation_data["train_transaction_ids"]
    test_ids = evaluation_data["test_transaction_ids"]
    risk_scores = evaluation_data["test_risk_scores"]

    if (
        not isinstance(features, pd.DataFrame)
        or features.empty
        or features.index.has_duplicates
        or tuple(features.columns) != FEATURE_COLUMNS
        ):
        raise ValueError("Cached features are invalid")
    if (
        not isinstance(labels, pd.Series)
        or not labels.index.equals(features.index)
        or labels.isnull().any()
        or not labels.isin([0, 1]).all()
        ):
        raise ValueError("Cached labels are invalid")
    if (
        not isinstance(fraud_types, pd.Series)
        or not fraud_types.index.equals(features.index)
        or fraud_types.loc[labels == 1].isnull().any()
        ):
        raise ValueError("Cached fraud types are invalid")
    if (
        not isinstance(train_ids, tuple)
        or not isinstance(test_ids, tuple)
        or len(set(train_ids)) != len(train_ids)
        or len(set(test_ids)) != len(test_ids)
        or set(train_ids) & set(test_ids)
        or set(train_ids) | set(test_ids) != set(features.index)
        ):
        raise ValueError("Cached train/test split is invalid")
    if (
        not isinstance(risk_scores, pd.Series)
        or tuple(risk_scores.index) != test_ids
        or risk_scores.isnull().any()
        or any(
            not math.isfinite(float(score))
            or not 0 <= float(score) <= 1
            for score in risk_scores
            )
        ):
        raise ValueError("Cached test risk scores are invalid")


def _read_valid_cache(cache_path, source_fingerprint):
    if not cache_path.exists():
        return None
    try:
        cache_record = pd.read_pickle(cache_path)
        if (
            not isinstance(cache_record, dict)
            or cache_record.get("cache_schema_version")
            != CACHE_SCHEMA_VERSION
            or cache_record.get("source_fingerprint")
            != source_fingerprint
            ):
            return None
        evaluation_data = cache_record["evaluation_data"]
        _validate_evaluation_data(evaluation_data)
    except Exception:
        return None
    return evaluation_data


def _write_cache(
    cache_path,
    source_fingerprint,
    evaluation_data
    ):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
        )
    cache_record = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "evaluation_data": evaluation_data
        }
    try:
        pd.to_pickle(cache_record, temporary_path)
        os.replace(temporary_path, cache_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_or_build_evaluation_data(cache_path=None):
    """Return shared evaluation inputs from cache or a fresh build."""
    cache_path = Path(cache_path) if cache_path else CACHE_PATH
    source_fingerprint = _source_fingerprint()
    evaluation_data = _read_valid_cache(
        cache_path,
        source_fingerprint
        )
    if evaluation_data is not None:
        return evaluation_data, "loaded"

    evaluation_data = _build_evaluation_data()
    _validate_evaluation_data(evaluation_data)
    _write_cache(
        cache_path,
        source_fingerprint,
        evaluation_data
        )
    return evaluation_data, "built"
