import json
import math
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.app import _build_dashboard_state
from src.feature_engineering.feature_engineer import FEATURE_COLUMNS


BASELINE_AMOUNT_THRESHOLD = 2_000.0
RESULTS_PATH = Path(__file__).with_name("audit_results.json")
TOP_FRACTIONS = (0.01, 0.05)


def _binary_precision_recall(actual_labels, predicted_labels):
    true_positives = int((
        (actual_labels == 1)
        & (predicted_labels == 1)
        ).sum())
    false_positives = int((
        (actual_labels == 0)
        & (predicted_labels == 1)
        ).sum())
    false_negatives = int((
        (actual_labels == 1)
        & (predicted_labels == 0)
        ).sum())

    precision_denominator = true_positives + false_positives
    recall_denominator = true_positives + false_negatives
    precision = (
        true_positives / precision_denominator
        if precision_denominator
        else 0.0
        )
    recall = (
        true_positives / recall_denominator
        if recall_denominator
        else 0.0
        )
    return float(precision), float(recall)


def _recall_at_top_fraction(actual_labels, scores, fraction):
    alert_count = max(1, math.ceil(len(scores) * fraction))
    ranked_scores = pd.DataFrame({
        "score": scores,
        "_transaction_id": scores.index
        }).sort_values(
            ["score", "_transaction_id"],
            ascending=[False, True],
            kind="mergesort"
            )
    alerted_ids = ranked_scores.index[:alert_count]
    fraud_count = int((actual_labels == 1).sum())
    detected_fraud_count = int((actual_labels.loc[alerted_ids] == 1).sum())
    return float(detected_fraud_count / fraud_count)


def _ranking_metrics(actual_labels, scores):
    metrics = {
        "pr_auc": float(average_precision_score(actual_labels, scores))
        }
    for fraction in TOP_FRACTIONS:
        percentage = int(fraction * 100)
        metrics[f"recall_at_top_{percentage}_percent"] = (
            _recall_at_top_fraction(
                actual_labels,
                scores,
                fraction
                )
            )
    return metrics


def build_audit_results():
    state = _build_dashboard_state()
    transactions = state["transactions"]
    features = state["features"]
    risk_model = state["risk_model"]

    labels = transactions["is_fraud"].astype("int64")
    train_ids = list(risk_model.train_transaction_ids)
    test_ids = list(risk_model.test_transaction_ids)
    train_labels = labels.loc[train_ids]
    test_labels = labels.loc[test_ids]

    test_fraud_transactions = transactions.loc[
        test_ids
        ].loc[lambda frame: frame["is_fraud"]]
    fraud_count_by_type = {
        str(fraud_type): int(count)
        for fraud_type, count in test_fraud_transactions[
            "fraud_type"
            ].value_counts().sort_index().items()
        }

    single_feature_roc_auc = {
        feature: float(roc_auc_score(labels, features[feature]))
        for feature in FEATURE_COLUMNS
        }

    baseline_scores = features.loc[
        test_ids,
        "transaction_amount"
        ].astype("float64")
    baseline_predictions = (
        baseline_scores >= BASELINE_AMOUNT_THRESHOLD
        ).astype("int64")
    baseline_precision, baseline_recall = _binary_precision_recall(
        test_labels,
        baseline_predictions
        )

    model_scores = risk_model.evaluation_results[
        "risk_score"
        ].reindex(test_ids)
    return {
        "dataset": {
            "total_transactions": int(len(transactions)),
            "fraud_count": int(labels.sum()),
            "train_fraud_count": int(train_labels.sum()),
            "test_fraud_count": int(test_labels.sum()),
            "test_fraud_count_by_type": fraud_count_by_type
            },
        "single_feature_roc_auc": single_feature_roc_auc,
        "baseline": {
            "rule": "transaction_amount >= 2000",
            "precision": baseline_precision,
            "recall": baseline_recall,
            **_ranking_metrics(test_labels, baseline_scores)
            },
        "xgboost": _ranking_metrics(test_labels, model_scores)
        }


def _print_results(results):
    dataset = results["dataset"]
    print("Dataset")
    print(f"  Total transactions: {dataset['total_transactions']}")
    print(f"  Fraud count: {dataset['fraud_count']}")
    print(f"  Fraud count in train: {dataset['train_fraud_count']}")
    print(f"  Fraud count in test: {dataset['test_fraud_count']}")
    print("  Fraud count by fraud_type in test:")
    for fraud_type, count in dataset[
        "test_fraud_count_by_type"
        ].items():
        print(f"    {fraud_type}: {count}")

    print("\nSingle-feature ROC AUC on all rows")
    for feature in FEATURE_COLUMNS:
        print(
            f"  {feature}: "
            f"{results['single_feature_roc_auc'][feature]:.6f}"
            )

    baseline = results["baseline"]
    print("\nBaseline rule: transaction_amount >= 2000")
    print(f"  Precision (test): {baseline['precision']:.6f}")
    print(f"  Recall (test): {baseline['recall']:.6f}")

    print("\nTest-set ranking metrics")
    for display_name, result_name in (
        ("XGBoost", "xgboost"),
        ("Baseline (transaction amount score)", "baseline")
        ):
        metrics = results[result_name]
        print(f"  {display_name}")
        print(f"    PR-AUC: {metrics['pr_auc']:.6f}")
        print(
            "    Recall at top 1%: "
            f"{metrics['recall_at_top_1_percent']:.6f}"
            )
        print(
            "    Recall at top 5%: "
            f"{metrics['recall_at_top_5_percent']:.6f}"
            )


def main():
    results = build_audit_results()
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
        )
    _print_results(results)
    print(f"\nSaved results to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
