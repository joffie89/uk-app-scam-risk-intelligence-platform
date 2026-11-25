import json
import math
from pathlib import Path
import sys
import time

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.evaluation_cache import load_or_build_evaluation_data
from src.data_generation.fraud_generator import (
    INVESTMENT_SCAM,
    MONEY_MULE_LAYERING
    )


ALERT_RATES = (0.005, 0.01, 0.02, 0.05)
FRAUD_TYPES = (INVESTMENT_SCAM, MONEY_MULE_LAYERING)
RULE_AMOUNT_THRESHOLD = 2_000.0
RESULTS_PATH = Path(__file__).with_name("shadow_mode_results.json")
CHART_PATH = Path(__file__).with_name("shadow_mode.png")
SUMMARY_PATH = Path(__file__).with_name("shadow_mode.md")
_CHART_FONT_PATHS = (
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path(
        "/usr/share/fonts/truetype/liberation2/"
        "LiberationSans-Regular.ttf"
        ),
    Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    )


def _rate_label(alert_rate):
    return f"{alert_rate * 100:g}%"


def _fraud_outcomes(
    labels,
    model_alerts,
    rule_flags,
    scope=None
    ):
    if scope is None:
        scope = pd.Series(True, index=labels.index)
    frauds = labels.eq(1) & scope
    outcomes = {
        "total_frauds_in_test": int(frauds.sum()),
        "frauds_caught_by_both": int((
            frauds & model_alerts & rule_flags
            ).sum()),
        "frauds_caught_only_by_rule": int((
            frauds & ~model_alerts & rule_flags
            ).sum()),
        "frauds_caught_only_by_model": int((
            frauds & model_alerts & ~rule_flags
            ).sum()),
        "frauds_caught_by_neither": int((
            frauds & ~model_alerts & ~rule_flags
            ).sum())
        }
    partition_total = sum(
        outcomes[key]
        for key in (
            "frauds_caught_by_both",
            "frauds_caught_only_by_rule",
            "frauds_caught_only_by_model",
            "frauds_caught_by_neither"
            )
        )
    if partition_total != outcomes["total_frauds_in_test"]:
        raise ValueError("Fraud outcomes must form a complete partition")
    return outcomes


def build_shadow_mode_results(evaluation_data):
    test_ids = list(evaluation_data["test_transaction_ids"])
    test_features = evaluation_data["features"].loc[test_ids]
    test_labels = evaluation_data["labels"].loc[test_ids]
    test_fraud_types = evaluation_data["fraud_types"].loc[test_ids]
    model_scores = evaluation_data["test_risk_scores"].reindex(
        test_ids
        )
    rule_flags = (
        test_features["transaction_amount"]
        >= RULE_AMOUNT_THRESHOLD
        )
    ranked_scores = pd.DataFrame({
        "risk_score": model_scores,
        "_transaction_id": model_scores.index
        }).sort_values(
            ["risk_score", "_transaction_id"],
            ascending=[False, True],
            kind="mergesort"
            )

    rate_results = []
    for alert_rate in ALERT_RATES:
        model_alert_count = max(
            1,
            math.ceil(len(test_ids) * alert_rate)
            )
        model_alerts = pd.Series(False, index=test_labels.index)
        model_alerts.loc[
            ranked_scores.index[:model_alert_count]
            ] = True

        fraud_outcomes = _fraud_outcomes(
            test_labels,
            model_alerts,
            rule_flags
            )
        by_fraud_type = {
            fraud_type: _fraud_outcomes(
                test_labels,
                model_alerts,
                rule_flags,
                scope=test_fraud_types.eq(fraud_type)
                )
            for fraud_type in FRAUD_TYPES
            }
        extra_alerts = rule_flags & ~model_alerts
        extra_alert_count = int(extra_alerts.sum())
        extra_fraud_count = int((
            extra_alerts & test_labels.eq(1)
            ).sum())
        extra_alert_precision = (
            extra_fraud_count / extra_alert_count
            if extra_alert_count
            else 0.0
            )
        if (
            extra_fraud_count
            != fraud_outcomes["frauds_caught_only_by_rule"]
            ):
            raise ValueError(
                "Extra rule frauds must equal rule-only frauds"
                )
        if sum(
            outcomes["total_frauds_in_test"]
            for outcomes in by_fraud_type.values()
            ) != fraud_outcomes["total_frauds_in_test"]:
            raise ValueError("Fraud-type totals must match all fraud")

        rate_results.append({
            "k_percent": float(alert_rate * 100),
            "model_alert_count": int(model_alert_count),
            "rule_flag_count": int(rule_flags.sum()),
            "fraud_outcomes": fraud_outcomes,
            "by_fraud_type": by_fraud_type,
            "rule_reactivation": {
                "extra_alert_count": extra_alert_count,
                "extra_fraud_count": extra_fraud_count,
                "extra_alert_precision": float(extra_alert_precision)
                }
            })

    return {
        "metadata": {
            "test_transaction_count": int(len(test_ids)),
            "total_test_frauds": int(test_labels.sum()),
            "rule": "transaction_amount >= 2000",
            "rule_mode": "shadow",
            "model_alert_rates_percent": [
                float(alert_rate * 100)
                for alert_rate in ALERT_RATES
                ]
            },
        "results": rate_results
        }


def _font(size):
    for font_path in _CHART_FONT_PATHS:
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default(size=size)


def _centred_text(draw, centre_x, y, text, font, fill):
    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    draw.text(
        (centre_x - text_width / 2, y),
        text,
        font=font,
        fill=fill
        )


def save_chart(results):
    width, height = 1_200, 720
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = _font(30)
    label_font = _font(21)
    tick_font = _font(18)
    value_font = _font(17)
    legend_font = _font(18)

    plot_left = 125
    plot_top = 145
    plot_right = 1_100
    plot_bottom = 610
    plot_height = plot_bottom - plot_top
    y_maximum = 110

    _centred_text(
        draw,
        width / 2,
        24,
        "Retiring the £2,000 rule: what the model would miss",
        title_font,
        "#111111"
        )
    categories = (
        ("frauds_caught_by_both", "Both", "#00796B", "white"),
        (
            "frauds_caught_only_by_model",
            "Model only",
            "#1565C0",
            "white"
            ),
        (
            "frauds_caught_only_by_rule",
            "Rule only",
            "#F9A825",
            "#111111"
            ),
        (
            "frauds_caught_by_neither",
            "Neither",
            "#9E9E9E",
            "#111111"
            )
        )
    legend_x = 180
    for _, label, colour, _ in categories:
        draw.rectangle(
            (legend_x, 86, legend_x + 24, 110),
            fill=colour
            )
        draw.text(
            (legend_x + 34, 85),
            label,
            font=legend_font,
            fill="#222222"
            )
        legend_x += 230

    tick_values = [0, 20, 40, 60, 80, 100, 110]
    for tick_value in tick_values:
        y = plot_bottom - round(
            tick_value / y_maximum * plot_height
            )
        draw.line(
            (plot_left, y, plot_right, y),
            fill="#DDDDDD",
            width=1
            )
        tick_text = str(tick_value)
        tick_box = draw.textbbox((0, 0), tick_text, font=tick_font)
        draw.text(
            (plot_left - 14 - (tick_box[2] - tick_box[0]), y - 10),
            tick_text,
            font=tick_font,
            fill="#333333"
            )

    draw.line(
        (plot_left, plot_top, plot_left, plot_bottom),
        fill="#222222",
        width=2
        )
    draw.line(
        (plot_left, plot_bottom, plot_right, plot_bottom),
        fill="#222222",
        width=2
        )

    bar_width = 135
    available_width = plot_right - plot_left
    spacing = available_width / len(results["results"])
    for bar_index, rate_result in enumerate(results["results"]):
        centre_x = plot_left + spacing * (bar_index + 0.5)
        left = round(centre_x - bar_width / 2)
        right = round(centre_x + bar_width / 2)
        cumulative_value = 0
        outcomes = rate_result["fraud_outcomes"]
        for key, _, colour, text_colour in categories:
            value = outcomes[key]
            bottom_y = plot_bottom - round(
                cumulative_value / y_maximum * plot_height
                )
            cumulative_value += value
            top_y = plot_bottom - round(
                cumulative_value / y_maximum * plot_height
                )
            if value:
                draw.rectangle(
                    (left, top_y, right, bottom_y),
                    fill=colour,
                    outline="white"
                    )
                if (
                    key != "frauds_caught_only_by_rule"
                    and bottom_y - top_y >= 17
                    ):
                    _centred_text(
                        draw,
                        centre_x,
                        (top_y + bottom_y) / 2 - 9,
                        str(value),
                        value_font,
                        text_colour
                        )
        rule_only_value = outcomes[
            "frauds_caught_only_by_rule"
            ]
        rule_only_bottom_value = (
            outcomes["frauds_caught_by_both"]
            + outcomes["frauds_caught_only_by_model"]
            )
        rule_only_centre_value = (
            rule_only_bottom_value + rule_only_value / 2
            )
        rule_only_centre_y = plot_bottom - round(
            rule_only_centre_value / y_maximum * plot_height
            )
        draw.line(
            (
                right,
                rule_only_centre_y,
                right + 7,
                rule_only_centre_y
                ),
            fill="#9A6500",
            width=1
            )
        draw.text(
            (right + 10, rule_only_centre_y - 9),
            f"Rule only: {rule_only_value}",
            font=value_font,
            fill="#6F4800"
            )
        _centred_text(
            draw,
            centre_x,
            plot_bottom + 18,
            f"{rate_result['k_percent']:g}%",
            tick_font,
            "#222222"
            )

    _centred_text(
        draw,
        (plot_left + plot_right) / 2,
        665,
        "Model alert rate (top K% of test transactions)",
        label_font,
        "#111111"
        )
    y_label = "Number of fraudulent test transactions"
    y_label_box = Image.new(
        "RGBA",
        (440, 40),
        (255, 255, 255, 0)
        )
    y_label_draw = ImageDraw.Draw(y_label_box)
    y_label_draw.text(
        (0, 5),
        y_label,
        font=label_font,
        fill="#111111"
        )
    rotated_label = y_label_box.rotate(90, expand=True)
    image.paste(
        rotated_label,
        (20, round((height - rotated_label.height) / 2)),
        rotated_label
        )
    image.save(CHART_PATH, format="PNG")


def _rate_result(results, k_percent):
    return next(
        rate_result
        for rate_result in results["results"]
        if rate_result["k_percent"] == k_percent
        )


def _summary_row(label, outcomes):
    return (
        f"| {label} | {outcomes['total_frauds_in_test']} | "
        f"{outcomes['frauds_caught_by_both']} | "
        f"{outcomes['frauds_caught_only_by_rule']} | "
        f"{outcomes['frauds_caught_only_by_model']} | "
        f"{outcomes['frauds_caught_by_neither']} |"
        )


def save_summary(results):
    one_percent = _rate_result(results, 1.0)
    outcomes = one_percent["fraud_outcomes"]
    reactivation = one_percent["rule_reactivation"]
    rule_only_count = outcomes["frauds_caught_only_by_rule"]
    if rule_only_count:
        catch_label = (
            "fraud catch" if rule_only_count == 1 else "fraud catches"
            )
        conclusion = (
            f"Retiring the rule loses {rule_only_count} {catch_label}; "
            f"keeping it adds {reactivation['extra_alert_count']} alerts "
            f"at {reactivation['extra_alert_precision']:.2%} precision."
            )
    else:
        conclusion = (
            "Retiring the rule loses no fraud catches in this test; "
            f"keeping it adds {reactivation['extra_alert_count']} alerts."
            )

    lines = [
        "# Shadow-mode assessment: retire the £2,000 rule?",
        "",
        "**Question:** If we retire the £2,000 rule, what do we lose?",
        (
            "**Method:** On held-out test transactions, XGBoost alerts "
            "are the top 1% by risk score; the rule is logged in shadow "
            "mode and does not create alerts."
            ),
        "",
        (
            "| K=1% segment | Total frauds | Both | Rule only | "
            "Model only | Neither |"
            ),
        "|---|---:|---:|---:|---:|---:|",
        _summary_row("All fraud", outcomes),
        _summary_row(
            INVESTMENT_SCAM,
            one_percent["by_fraud_type"][INVESTMENT_SCAM]
            ),
        _summary_row(
            MONEY_MULE_LAYERING,
            one_percent["by_fraud_type"][MONEY_MULE_LAYERING]
            ),
        "",
        (
            f"**Operational impact:** Reactivating the rule adds "
            f"{reactivation['extra_alert_count']} alerts, including "
            f"{reactivation['extra_fraud_count']} fraud "
            f"({reactivation['extra_alert_precision']:.2%} precision)."
            ),
        f"**Conclusion:** {conclusion}",
        (
            "**Limitation:** Synthetic data cannot measure deterrence "
            "(whether fraudsters avoid large payments because the rule "
            "exists)."
            )
        ]
    if len(lines) >= 15:
        raise ValueError("Shadow-mode summary must be under 15 lines")
    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
        )


def _print_outcomes(prefix, outcomes):
    print(
        f"  {prefix}: total={outcomes['total_frauds_in_test']}, "
        f"both={outcomes['frauds_caught_by_both']}, "
        f"rule_only={outcomes['frauds_caught_only_by_rule']}, "
        f"model_only={outcomes['frauds_caught_only_by_model']}, "
        f"neither={outcomes['frauds_caught_by_neither']}"
        )


def print_results(results, cache_status, runtime_seconds):
    metadata = results["metadata"]
    print("Shadow-mode rule evaluation")
    print(f"  Evaluation cache: {cache_status}")
    print(
        f"  Test transactions: {metadata['test_transaction_count']}"
        )
    print(f"  Total frauds in test: {metadata['total_test_frauds']}")
    print("  Rule: transaction_amount >= 2000 (shadow only)")
    for rate_result in results["results"]:
        print(
            f"\nK={rate_result['k_percent']:g}%: "
            f"model alerts={rate_result['model_alert_count']}, "
            f"rule flags={rate_result['rule_flag_count']}"
            )
        _print_outcomes("All fraud", rate_result["fraud_outcomes"])
        for fraud_type in FRAUD_TYPES:
            _print_outcomes(
                fraud_type,
                rate_result["by_fraud_type"][fraud_type]
                )
        reactivation = rate_result["rule_reactivation"]
        print(
            "  Rule reactivation: "
            f"extra_alerts={reactivation['extra_alert_count']}, "
            f"extra_frauds={reactivation['extra_fraud_count']}, "
            "extra_alert_precision="
            f"{reactivation['extra_alert_precision']:.6%}"
            )
    print(f"\nSaved results to {RESULTS_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved chart to {CHART_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Saved summary to {SUMMARY_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Runtime: {runtime_seconds:.3f} seconds")


def main():
    start_time = time.perf_counter()
    evaluation_data, cache_status = load_or_build_evaluation_data()
    results = build_shadow_mode_results(evaluation_data)
    save_chart(results)
    save_summary(results)
    runtime_seconds = time.perf_counter() - start_time
    results["run"] = {
        "evaluation_cache": cache_status,
        "runtime_seconds": float(runtime_seconds)
        }
    RESULTS_PATH.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8"
        )
    print_results(results, cache_status, runtime_seconds)


if __name__ == "__main__":
    main()
