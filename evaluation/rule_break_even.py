from dataclasses import asdict
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from faker import Faker
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_cache import load_or_build_evaluation_data
from src.dashboard.app import (
    CUSTOMER_COUNT,
    RANDOM_SEED,
    REFERENCE_DATETIME,
    SCENARIOS_PER_TYPE,
    _generate_entities,
    _generate_transactions
    )
from src.data_generation.fraud_generator import INVESTMENT_SCAM


MODEL_ALERT_RATE = 0.01
RULE_AMOUNT_THRESHOLD = 2_000.0
EXPECTED_EXTRA_ALERTS = 506
EXPECTED_GENUINE_EXTRA_ALERTS = 503
EXPECTED_RULE_ONLY_FRAUDS = 3
REVIEW_COSTS_GBP = (2, 5, 10, 20)
CHURN_COSTS_GBP = (0, 1, 5, 10)
RESULTS_PATH = Path(__file__).with_name("rule_break_even.json")
SUMMARY_PATH = Path(__file__).with_name("rule_break_even.md")
SHADOW_RESULTS_PATH = Path(__file__).with_name(
    "shadow_mode_results.json"
    )
METADATA_CACHE_PATH = (
    Path(__file__).with_name("cache")
    / "transaction_metadata.pkl"
    )
METADATA_CACHE_SCHEMA_VERSION = 1
SCENARIO_ID_COLUMNS = (
    "scenario_id",
    "fraud_scenario_id",
    "scam_scenario_id"
    )
PENNY = Decimal("0.01")


def _decimal_amount(value):
    return Decimal(str(value)).quantize(
        PENNY,
        rounding=ROUND_HALF_UP
        )


def _sum_amounts(values):
    return sum(
        (_decimal_amount(value) for value in values),
        Decimal("0.00")
        ).quantize(PENNY)


def _currency(value):
    return f"£{value:,.2f}"


def _generate_transaction_metadata():
    """Replay generation only; do not rebuild features or train a model."""
    random.seed(RANDOM_SEED)
    Faker.seed(RANDOM_SEED)
    selection_random = random.Random(RANDOM_SEED + 1)

    customer_indices = list(range(CUSTOMER_COUNT))
    selection_random.shuffle(customer_indices)
    investment_customer_indices = set(
        customer_indices[:SCENARIOS_PER_TYPE]
        )
    mule_customer_indices = set(
        customer_indices[
            SCENARIOS_PER_TYPE:SCENARIOS_PER_TYPE * 2
            ]
        )
    entities = _generate_entities(
        investment_customer_indices,
        mule_customer_indices,
        selection_random,
        REFERENCE_DATETIME
        )
    transactions = _generate_transactions(
        entities,
        investment_customer_indices,
        mule_customer_indices,
        REFERENCE_DATETIME
        )
    metadata = pd.DataFrame([
        asdict(transaction)
        for transaction in transactions
        ]).set_index("transaction_id")
    required_columns = [
        "account_id",
        "amount",
        "timestamp",
        "is_fraud",
        "fraud_type"
        ]
    scenario_columns = [
        column
        for column in SCENARIO_ID_COLUMNS
        if column in metadata.columns
        ]
    return metadata[required_columns + scenario_columns].copy()


def _validate_transaction_metadata(metadata, evaluation_data):
    required_columns = {
        "account_id",
        "amount",
        "timestamp",
        "is_fraud",
        "fraud_type"
        }
    feature_index = evaluation_data["features"].index
    if (
        not isinstance(metadata, pd.DataFrame)
        or not required_columns.issubset(metadata.columns)
        or metadata.index.has_duplicates
        or set(metadata.index) != set(feature_index)
        ):
        raise ValueError("Transaction metadata cache is invalid")

    metadata = metadata.reindex(feature_index).copy()
    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"],
        errors="raise"
        )
    if (
        metadata["account_id"].isnull().any()
        or metadata["timestamp"].isnull().any()
        or metadata["amount"].isnull().any()
        ):
        raise ValueError("Transaction metadata contains missing values")

    cached_amounts = evaluation_data["features"][
        "transaction_amount"
        ].astype("float64")
    metadata_amounts = metadata["amount"].astype("float64")
    if ((cached_amounts - metadata_amounts).abs() > 1e-9).any():
        raise ValueError("Metadata amounts do not match evaluation cache")
    if not metadata["is_fraud"].astype("int64").equals(
        evaluation_data["labels"].astype("int64")
        ):
        raise ValueError("Metadata labels do not match evaluation cache")
    cached_types = evaluation_data["fraud_types"].fillna("")
    metadata_types = metadata["fraud_type"].fillna("")
    if not metadata_types.equals(cached_types):
        raise ValueError("Metadata fraud types do not match evaluation cache")
    return metadata


def _write_metadata_cache(metadata):
    METADATA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = METADATA_CACHE_PATH.with_name(
        f".{METADATA_CACHE_PATH.name}.{os.getpid()}.tmp"
        )
    record = {
        "cache_schema_version": METADATA_CACHE_SCHEMA_VERSION,
        "metadata": metadata
        }
    try:
        pd.to_pickle(record, temporary_path)
        os.replace(temporary_path, METADATA_CACHE_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)


def _load_or_build_transaction_metadata(evaluation_data):
    if METADATA_CACHE_PATH.exists():
        try:
            record = pd.read_pickle(METADATA_CACHE_PATH)
            if (
                not isinstance(record, dict)
                or record.get("cache_schema_version")
                != METADATA_CACHE_SCHEMA_VERSION
                ):
                raise ValueError("Metadata cache schema does not match")
            metadata = _validate_transaction_metadata(
                record["metadata"],
                evaluation_data
                )
        except Exception:
            pass
        else:
            return metadata, "loaded"

    metadata = _generate_transaction_metadata()
    metadata = _validate_transaction_metadata(
        metadata,
        evaluation_data
        )
    _write_metadata_cache(metadata)
    return metadata, "built"


def _shadow_result_at_one_percent():
    shadow_results = json.loads(
        SHADOW_RESULTS_PATH.read_text(encoding="utf-8")
        )
    matches = [
        result
        for result in shadow_results["results"]
        if math.isclose(result["k_percent"], 1.0)
        ]
    if len(matches) != 1:
        raise ValueError("Expected one shadow-mode result for K=1%")
    return matches[0]


def _scenario_details(metadata, transaction_id):
    transaction = metadata.loc[transaction_id]
    scenario_column = next(
        (
            column
            for column in SCENARIO_ID_COLUMNS
            if (
                column in metadata.columns
                and pd.notna(transaction[column])
                )
            ),
        None
        )
    if scenario_column:
        scenario_value = transaction[scenario_column]
        scenario_mask = metadata[scenario_column].eq(scenario_value)
        scenario_identifier = {
            "source": scenario_column,
            "value": str(scenario_value)
            }
        scenario_label = f"{scenario_column}={scenario_value}"
    else:
        account_id = str(transaction["account_id"])
        fraud_type = str(transaction["fraud_type"])
        scenario_mask = (
            metadata["account_id"].eq(account_id)
            & metadata["fraud_type"].eq(fraud_type)
            )
        scenario_identifier = {
            "source": "fallback_account_id_and_fraud_type",
            "account_id": account_id,
            "fraud_type": fraud_type
            }
        scenario_label = f"{account_id} + {fraud_type}"

    scenario_transactions = metadata.loc[
        scenario_mask & metadata["is_fraud"].astype(bool)
        ].copy()
    scenario_transactions["_transaction_id"] = (
        scenario_transactions.index
        )
    scenario_transactions = scenario_transactions.sort_values(
        ["timestamp", "_transaction_id"],
        kind="mergesort"
        )
    later_transactions = scenario_transactions.loc[
        scenario_transactions["timestamp"]
        > transaction["timestamp"]
        ]
    return {
        "scenario_identifier": scenario_identifier,
        "scenario_label": scenario_label,
        "scenario_transactions": scenario_transactions,
        "later_transactions": later_transactions
        }


def build_rule_break_even_results(evaluation_data, metadata):
    test_ids = list(evaluation_data["test_transaction_ids"])
    test_features = evaluation_data["features"].loc[test_ids]
    test_labels = evaluation_data["labels"].loc[test_ids]
    test_fraud_types = evaluation_data["fraud_types"].loc[test_ids]
    model_scores = evaluation_data["test_risk_scores"].reindex(
        test_ids
        )
    ranked_scores = pd.DataFrame({
        "risk_score": model_scores,
        "_transaction_id": model_scores.index
        }).sort_values(
            ["risk_score", "_transaction_id"],
            ascending=[False, True],
            kind="mergesort"
            )
    model_alert_count = max(
        1,
        math.ceil(len(test_ids) * MODEL_ALERT_RATE)
        )
    model_alerts = pd.Series(False, index=test_labels.index)
    model_alerts.loc[
        ranked_scores.index[:model_alert_count]
        ] = True
    rule_flags = (
        test_features["transaction_amount"]
        >= RULE_AMOUNT_THRESHOLD
        )
    extra_alerts = rule_flags & ~model_alerts
    genuine_extra_alerts = extra_alerts & test_labels.eq(0)
    rule_only_frauds = extra_alerts & test_labels.eq(1)

    shadow_result = _shadow_result_at_one_percent()
    expected_from_shadow = (
        shadow_result["rule_reactivation"]
        )
    checks = {
        "model alert count": (
            model_alert_count,
            shadow_result["model_alert_count"]
            ),
        "rule flag count": (
            int(rule_flags.sum()),
            shadow_result["rule_flag_count"]
            ),
        "extra alert count": (
            int(extra_alerts.sum()),
            expected_from_shadow["extra_alert_count"]
            ),
        "rule-only fraud count": (
            int(rule_only_frauds.sum()),
            expected_from_shadow["extra_fraud_count"]
            )
        }
    for label, (actual, expected) in checks.items():
        if actual != expected:
            raise ValueError(
                f"{label} differs from the K=1% shadow result: "
                f"{actual} != {expected}"
                )
    if (
        int(extra_alerts.sum()) != EXPECTED_EXTRA_ALERTS
        or int(genuine_extra_alerts.sum())
        != EXPECTED_GENUINE_EXTRA_ALERTS
        or int(rule_only_frauds.sum())
        != EXPECTED_RULE_ONLY_FRAUDS
        ):
        raise ValueError("Cached data does not match expected rule counts")
    if not test_fraud_types.loc[rule_only_frauds].eq(
        INVESTMENT_SCAM
        ).all():
        raise ValueError("Rule-only frauds must all be investment scams")

    caught_ids = list(rule_only_frauds.index[rule_only_frauds])
    caught_ids.sort(
        key=lambda transaction_id: (
            metadata.at[transaction_id, "timestamp"],
            transaction_id
            )
        )
    payment_results = []
    all_later_ids = set()
    for transaction_id in caught_ids:
        transaction = metadata.loc[transaction_id]
        scenario = _scenario_details(metadata, transaction_id)
        scenario_transactions = scenario["scenario_transactions"]
        later_transactions = scenario["later_transactions"]
        all_later_ids.update(later_transactions.index)
        payment_results.append({
            "transaction_id": transaction_id,
            "timestamp": transaction["timestamp"].isoformat(
                sep=" "
                ),
            "amount_gbp": float(_decimal_amount(
                transaction["amount"]
                )),
            "model_risk_score": float(model_scores.loc[
                transaction_id
                ]),
            "fraud_type": str(transaction["fraud_type"]),
            "scenario_identifier": scenario[
                "scenario_identifier"
                ],
            "scenario_label": scenario["scenario_label"],
            "scam_payment_count": int(len(
                scenario_transactions
                )),
            "full_scam_value_gbp": float(_sum_amounts(
                scenario_transactions["amount"]
                )),
            "later_payment_count": int(len(later_transactions)),
            "later_payment_value_gbp": float(_sum_amounts(
                later_transactions["amount"]
                )),
            "later_payment_transaction_ids": list(
                later_transactions.index
                )
            })

    scenario_identifiers = {
        json.dumps(
            payment["scenario_identifier"],
            sort_keys=True
            )
        for payment in payment_results
        }
    if len(scenario_identifiers) != 1:
        raise ValueError(
            "Expected the three rule-only frauds to share one scam"
            )

    caught_id_set = set(caught_ids)
    l_low = _sum_amounts(
        metadata.loc[caught_ids, "amount"]
        )
    raw_per_payment_later_sum = _sum_amounts(
        payment["later_payment_value_gbp"]
        for payment in payment_results
        )
    unique_later_ids = sorted(all_later_ids)
    unique_later_value = _sum_amounts(
        metadata.loc[unique_later_ids, "amount"]
        if unique_later_ids
        else []
        )
    incremental_later_ids = sorted(
        all_later_ids - caught_id_set
        )
    incremental_later_value = _sum_amounts(
        metadata.loc[incremental_later_ids, "amount"]
        if incremental_later_ids
        else []
        )
    l_high = (l_low + incremental_later_value).quantize(PENNY)

    cost_table = []
    for review_cost in REVIEW_COSTS_GBP:
        cells = []
        for churn_cost in CHURN_COSTS_GBP:
            total_cost = (
                EXPECTED_EXTRA_ALERTS * review_cost
                + EXPECTED_GENUINE_EXTRA_ALERTS * churn_cost
                )
            cells.append({
                "churn_cost_per_genuine_alert_gbp": churn_cost,
                "total_cost_gbp": total_cost,
                "pays_off_under_l_low": (
                    l_low >= Decimal(total_cost)
                    ),
                "pays_off_under_l_high": (
                    l_high >= Decimal(total_cost)
                    )
                })
        cost_table.append({
            "review_cost_per_alert_gbp": review_cost,
            "cells": cells
            })

    low_payoff_count = sum(
        cell["pays_off_under_l_low"]
        for row in cost_table
        for cell in row["cells"]
        )
    high_payoff_count = sum(
        cell["pays_off_under_l_high"]
        for row in cost_table
        for cell in row["cells"]
        )
    highest_cost = max(
        cell["total_cost_gbp"]
        for row in cost_table
        for cell in row["cells"]
        )
    narrowest_margin = min(l_low, l_high) - Decimal(highest_cost)
    conclusion = (
        "Keeping the rule pays off in "
        f"{low_payoff_count}/16 illustrative cells under L_low "
        f"and {high_payoff_count}/16 under L_high. At the "
        f"highest assumed cost, the margin is {_currency(narrowest_margin)}; "
        "the result is driven by one synthetic investment-scam scenario."
        )

    return {
        "question": "Is keeping the £2,000 rule worth it?",
        "method": {
            "model_alert_rate_percent": 1.0,
            "model_alert_count": model_alert_count,
            "rule": "transaction_amount >= 2000",
            "scenario_grouping": (
                "Use a scenario identifier if present; otherwise "
                "account_id + fraud_type."
                ),
            "prevented_loss_low": (
                "Sum of the rule-only caught payments."
                ),
            "prevented_loss_high": (
                "L_low plus unique later scenario payments not "
                "already included in L_low."
                ),
            "payoff_test": "prevented loss >= total cost"
            },
        "rule_only_payments": payment_results,
        "prevented_loss": {
            "l_low_gbp": float(l_low),
            "raw_sum_of_per_payment_later_values_gbp": float(
                raw_per_payment_later_sum
                ),
            "unique_later_payment_transaction_ids": (
                unique_later_ids
                ),
            "unique_later_payment_value_gbp": float(
                unique_later_value
                ),
            "incremental_later_payment_transaction_ids": (
                incremental_later_ids
                ),
            "incremental_later_payment_value_gbp": float(
                incremental_later_value
                ),
            "l_high_gbp": float(l_high),
            "deduplication_note": (
                "All three caught payments are in one scam. Later "
                "payments are deduplicated, and caught payments "
                "already counted in L_low are not added again."
                )
            },
        "cost_assumptions": {
            "illustrative_not_real_figures": True,
            "extra_alert_count": EXPECTED_EXTRA_ALERTS,
            "genuine_extra_alert_count": (
                EXPECTED_GENUINE_EXTRA_ALERTS
                ),
            "rule_only_fraud_count": EXPECTED_RULE_ONLY_FRAUDS,
            "formula": (
                "506 * review cost per alert + 503 * churn "
                "cost per genuine alert"
                ),
            "review_costs_per_alert_gbp": list(
                REVIEW_COSTS_GBP
                ),
            "churn_costs_per_genuine_alert_gbp": list(
                CHURN_COSTS_GBP
                )
            },
        "cost_table": cost_table,
        "conclusion": conclusion,
        "limitations": [
            "Synthetic data",
            "Illustrative costs, not real figures",
            "Deterrence is not measured",
            "Reimbursement is not modelled"
            ]
        }


def _write_summary(results):
    payments = results["rule_only_payments"]
    prevented_loss = results["prevented_loss"]
    scenario_label = payments[0]["scenario_label"]
    payment_text = "; ".join(
        f"`{payment['transaction_id']}` "
        f"£{payment['amount_gbp']:,.2f}"
        for payment in payments
        )
    later_text = "; ".join(
        f"`{payment['transaction_id']}` "
        f"£{payment['later_payment_value_gbp']:,.2f}"
        for payment in payments
        )
    table_rows = []
    for row in results["cost_table"]:
        cells = []
        for cell in row["cells"]:
            low = "Y" if cell["pays_off_under_l_low"] else "N"
            high = "Y" if cell["pays_off_under_l_high"] else "N"
            cells.append(
                f"£{cell['total_cost_gbp']:,} ({low}/{high})"
                )
        table_rows.append(
            f"| £{row['review_cost_per_alert_gbp']} | "
            + " | ".join(cells)
            + " |"
            )
    lines = [
        "# Is keeping the £2,000 rule worth it?",
        "",
        "**Question:** At a 1% model alert rate, does the extra loss prevented by the £2,000 rule justify its alert cost?",
        "**Method:** Use cached test scores; model alerts are the top 1%, rule-only means amount >= £2,000 outside those alerts, and scams use an ID if available or account + fraud type.",
        f"**Rule-only payments:** {payment_text}; all are investment scams.",
        f"**Scam:** All three map to `{scenario_label}`; its four payments total £{payments[0]['full_scam_value_gbp']:,.2f}.",
        f"**Later value after each caught payment:** {later_text}.",
        f"**Prevented losses:** L_low = £{prevented_loss['l_low_gbp']:,.2f}; L_high = £{prevented_loss['l_high_gbp']:,.2f}, after deduplicating later payments already in L_low.",
        "**Illustrative assumptions, not real figures:** cost = 506 x review cost + 503 x genuine-alert churn cost; cells show total cost (L_low pays?/L_high pays?).",
        "",
        "| Review cost | Churn £0 | Churn £1 | Churn £5 | Churn £10 |",
        "|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
        f"**Conclusion:** {results['conclusion']}",
        "**Limitations:** Synthetic data; illustrative costs; deterrence is not measured; reimbursement is not modelled."
        ]
    if len(lines) >= 20:
        raise ValueError("Rule break-even summary must be under 20 lines")
    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
        )


def _print_results(results, evaluation_cache_status, metadata_status):
    print("Rule break-even analysis at the 1% model alert rate")
    print(f"Evaluation cache: {evaluation_cache_status}")
    print(f"Transaction metadata cache: {metadata_status}")
    print()
    print("Fraud payments caught only by the £2,000 rule:")
    for payment in results["rule_only_payments"]:
        print(
            f"  {payment['transaction_id']}: "
            f"£{payment['amount_gbp']:,.2f} "
            f"({payment['fraud_type']})"
            )
        print(f"    Scenario: {payment['scenario_label']}")
        print(
            "    Full scam: "
            f"£{payment['full_scam_value_gbp']:,.2f} "
            f"across {payment['scam_payment_count']} payments"
            )
        print(
            "    Payments after this one: "
            f"£{payment['later_payment_value_gbp']:,.2f} "
            f"across {payment['later_payment_count']} "
            + (
                "payment"
                if payment["later_payment_count"] == 1
                else "payments"
                )
            )

    prevented_loss = results["prevented_loss"]
    print()
    print("Prevented-loss estimates:")
    print(f"  L_low:  £{prevented_loss['l_low_gbp']:,.2f}")
    print(f"  L_high: £{prevented_loss['l_high_gbp']:,.2f}")
    print(
        "  Deduplication: all three payments share one scam; "
        "later caught payments already in L_low are not added twice."
        )

    print()
    print("Illustrative cost assumptions (not real figures)")
    print(
        "Cost = 506 extra alerts × review cost + "
        "503 genuine alerts × churn cost"
        )
    print(
        "Each cell: total cost (pays under L_low / pays under L_high)"
        )
    print(
        "Review cost | Churn £0 | Churn £1 | "
        "Churn £5 | Churn £10"
        )
    for row in results["cost_table"]:
        cell_text = []
        for cell in row["cells"]:
            low = "Yes" if cell["pays_off_under_l_low"] else "No"
            high = "Yes" if cell["pays_off_under_l_high"] else "No"
            cell_text.append(
                f"£{cell['total_cost_gbp']:,} ({low}/{high})"
                )
        print(
            "Review "
            f"£{row['review_cost_per_alert_gbp']:<2} | "
            + " | ".join(cell_text)
            )
    print()
    print(f"Conclusion: {results['conclusion']}")


def main():
    started_at = time.perf_counter()
    evaluation_data, evaluation_cache_status = (
        load_or_build_evaluation_data()
        )
    metadata, metadata_status = (
        _load_or_build_transaction_metadata(evaluation_data)
        )
    results = build_rule_break_even_results(
        evaluation_data,
        metadata
        )
    _write_summary(results)
    runtime_seconds = time.perf_counter() - started_at
    results["run"] = {
        "evaluation_cache": evaluation_cache_status,
        "transaction_metadata_cache": metadata_status,
        "runtime_seconds": runtime_seconds
        }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8"
        )
    _print_results(
        results,
        evaluation_cache_status,
        metadata_status
        )
    print()
    print(f"Saved JSON: {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved report: {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Runtime: {runtime_seconds:.2f} seconds")


if __name__ == "__main__":
    main()
