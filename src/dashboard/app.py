from dataclasses import asdict

from faker import Faker
import pandas as pd
import random
import streamlit as st

from src.data_generation import DEFAULT_REFERENCE_DATETIME
from src.data_generation.account_generator import AccountGenerator
from src.data_generation.beneficiary_generator import BeneficiaryGenerator
from src.data_generation.customer_generator import CustomerGenerator
from src.data_generation.device_generator import DeviceGenerator
from src.data_generation.fraud_generator import (
    FraudGenerator,
    INVESTMENT_SCAM,
    MONEY_MULE_LAYERING
    )
from src.data_generation.transaction_generator import TransactionGenerator
from src.explainability.shap_explainer import ShapExplainer
from src.feature_engineering.feature_engineer import FeatureEngineer
from src.models.risk_model import RiskModel


RANDOM_SEED = 20260914
REFERENCE_DATETIME = DEFAULT_REFERENCE_DATETIME
CUSTOMER_COUNT = 500
ACCOUNTS_PER_CUSTOMER = 2
DEVICES_PER_CUSTOMER = 2
BENEFICIARIES_PER_CUSTOMER = 6
TRANSACTIONS_PER_ACCOUNT = 10
SCENARIOS_PER_TYPE = 8
BENIGN_SHARED_DEVICE_PAIRS = 20

PAGE_NAMES = (
    "Overview",
    "Alerts List",
    "Transaction Detail"
    )


def _generate_entities(
    investment_customer_indices,
    mule_customer_indices,
    selection_random,
    reference_datetime
    ):
    customer_generator = CustomerGenerator(reference_datetime)
    account_generator = AccountGenerator(reference_datetime)
    device_generator = DeviceGenerator(reference_datetime)
    beneficiary_generator = BeneficiaryGenerator(reference_datetime)

    customers = []
    accounts_by_customer = {}
    devices_by_customer = {}
    beneficiaries_by_customer = {}

    for _ in range(CUSTOMER_COUNT):
        customer = customer_generator.generate_customer()
        customers.append(customer)

        active_accounts = []
        while len(active_accounts) < ACCOUNTS_PER_CUSTOMER:
            account = account_generator.generate_account(
                customer.customer_id
                )
            if account.account_status == "Active":
                active_accounts.append(account)
        accounts_by_customer[customer.customer_id] = active_accounts
        devices_by_customer[customer.customer_id] = [
            device_generator.generate_device()
            for _ in range(DEVICES_PER_CUSTOMER)
            ]
        beneficiaries_by_customer[customer.customer_id] = [
            beneficiary_generator.generate_beneficiary()
            for _ in range(BENEFICIARIES_PER_CUSTOMER)
            ]

    scenario_customer_indices = (
        investment_customer_indices | mule_customer_indices
        )
    non_scenario_customer_indices = [
        customer_index
        for customer_index in range(CUSTOMER_COUNT)
        if customer_index not in scenario_customer_indices
        ]
    selection_random.shuffle(non_scenario_customer_indices)
    shared_customer_indices = non_scenario_customer_indices[
        :BENIGN_SHARED_DEVICE_PAIRS * 2
        ]
    for pair_start in range(
        0,
        len(shared_customer_indices),
        2
        ):
        first_customer = customers[
            shared_customer_indices[pair_start]
            ]
        second_customer = customers[
            shared_customer_indices[pair_start + 1]
            ]
        devices_by_customer[second_customer.customer_id][1] = (
            devices_by_customer[first_customer.customer_id][1]
            )

    mule_shared_devices = [
        device_generator.generate_device()
        for _ in range(SCENARIOS_PER_TYPE // 2)
        ]
    mule_device_by_customer_index = {
        customer_index: mule_shared_devices[position // 2]
        for position, customer_index in enumerate(
            sorted(mule_customer_indices)
            )
        }

    return {
        "customers": customers,
        "accounts_by_customer": accounts_by_customer,
        "devices_by_customer": devices_by_customer,
        "beneficiaries_by_customer": beneficiaries_by_customer,
        "mule_device_by_customer_index": (
            mule_device_by_customer_index
            )
        }


def _generate_transactions(
    entities,
    investment_customer_indices,
    mule_customer_indices,
    reference_datetime
    ):
    transaction_generator = TransactionGenerator(reference_datetime)
    fraud_generator = FraudGenerator()
    transactions = []

    for customer_index, customer in enumerate(entities["customers"]):
        accounts = entities["accounts_by_customer"][
            customer.customer_id
            ]
        devices = entities["devices_by_customer"][
            customer.customer_id
            ]
        beneficiaries = entities["beneficiaries_by_customer"][
            customer.customer_id
            ]

        for account_index, account in enumerate(accounts):
            device = devices[account_index]
            normal_transaction_count = TRANSACTIONS_PER_ACCOUNT

            if (
                account_index == 0
                and customer_index in investment_customer_indices
                ):
                while True:
                    baseline_transactions = [
                        transaction_generator.generate_transaction(
                            customer,
                            account,
                            device,
                            beneficiaries[0]
                            )
                        for _ in range(2)
                        ]
                    try:
                        injected_transactions = (
                            fraud_generator.inject_investment_scam(
                                baseline_transactions
                                )
                            )
                    except ValueError as error:
                        if "timestamp" not in str(error).lower():
                            raise
                    else:
                        transactions.extend(injected_transactions)
                        break
                normal_transaction_count -= len(injected_transactions)

            elif (
                account_index == 0
                and customer_index in mule_customer_indices
                ):
                device = entities[
                    "mule_device_by_customer_index"
                    ][customer_index]
                while True:
                    baseline_transactions = [
                        transaction_generator.generate_transaction(
                            customer,
                            account,
                            device,
                            beneficiary
                            )
                        for beneficiary in beneficiaries[:4]
                        ]
                    try:
                        injected_transactions = (
                            fraud_generator.inject_money_mule_layering(
                                baseline_transactions
                                )
                            )
                    except ValueError as error:
                        if "timestamp" not in str(error).lower():
                            raise
                    else:
                        transactions.extend(injected_transactions)
                        break
                normal_transaction_count -= len(injected_transactions)

            for _ in range(normal_transaction_count):
                transactions.append(
                    transaction_generator.generate_transaction(
                        customer,
                        account,
                        device,
                        random.choice(beneficiaries)
                        )
                    )

    return transactions


@st.cache_resource(
    show_spinner="Generating data, training XGBoost, and calculating SHAP..."
    )
def _build_dashboard_state():
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

    customers = pd.DataFrame([
        asdict(customer)
        for customer in entities["customers"]
        ])
    accounts = pd.DataFrame([
        asdict(account)
        for customer_accounts in entities[
            "accounts_by_customer"
            ].values()
        for account in customer_accounts
        ])
    transaction_data = pd.DataFrame([
        asdict(transaction)
        for transaction in transactions
        ]).set_index("transaction_id")

    features = FeatureEngineer().build_features(
        transaction_data.reset_index(),
        accounts,
        customers
        )
    labels = transaction_data["is_fraud"].astype("int64")
    timestamps = transaction_data["timestamp"]

    risk_model = RiskModel(random_seed=RANDOM_SEED)
    risk_model.train_and_evaluate(
        features,
        labels,
        timestamps
        )

    train_features = features.loc[
        list(risk_model.train_transaction_ids)
        ]
    test_features = features.loc[
        list(risk_model.test_transaction_ids)
        ]
    shap_explainer = ShapExplainer(
        risk_model,
        train_features
        )
    global_feature_importance = (
        shap_explainer.global_feature_importance(test_features)
        )

    monitored_transactions = transaction_data.loc[
        list(risk_model.test_transaction_ids)
        ].join(
            risk_model.evaluation_results,
            how="left",
            validate="one_to_one"
            )
    monitored_transactions["fraud_type"] = (
        monitored_transactions["fraud_type"].fillna("Not labelled fraud")
        )

    return {
        "customers": customers,
        "accounts": accounts,
        "transactions": transaction_data,
        "features": features,
        "risk_model": risk_model,
        "shap_explainer": shap_explainer,
        "global_feature_importance": global_feature_importance,
        "monitored_transactions": monitored_transactions
        }


def _render_overview(state):
    st.title("UK APP Scam & Money Mule Risk Intelligence")
    st.caption(
        "Synthetic banking activity with chronological held-out "
        "model evaluation. Fixed reference clock: "
        f"{REFERENCE_DATETIME.isoformat(sep=' ')}."
        )

    transactions = state["transactions"]
    monitored_transactions = state["monitored_transactions"]
    metrics = state["risk_model"].metrics
    alert_count = int(
        (monitored_transactions["predicted_label"] == 1).sum()
        )
    fraud_count = int(transactions["is_fraud"].sum())

    volume_columns = st.columns(3)
    volume_columns[0].metric(
        "Synthetic transactions",
        f"{len(transactions):,}"
        )
    volume_columns[1].metric(
        "Held-out transactions",
        f"{len(monitored_transactions):,}"
        )
    volume_columns[2].metric(
        "Known fraud transactions",
        f"{fraud_count:,}"
        )

    result_columns = st.columns(4)
    result_columns[0].metric("Model alerts", f"{alert_count:,}")
    result_columns[1].metric(
        "Precision",
        f"{metrics['precision']:.1%}"
        )
    result_columns[2].metric(
        "Recall",
        f"{metrics['recall']:.1%}"
        )
    result_columns[3].metric("F1", f"{metrics['f1']:.3f}")

    st.subheader("Fraud scenario mix")
    scenario_mix = (
        transactions.loc[
            transactions["is_fraud"],
            ["fraud_type"]
            ]
        .value_counts()
        .rename("transaction_count")
        .reset_index()
        )
    st.dataframe(
        scenario_mix,
        hide_index=True,
        width="stretch"
        )

    st.subheader("Global feature importance")
    st.caption(
        "Mean absolute SHAP contribution across the chronological "
        "held-out set. Merchant and merchant category are not model inputs."
        )
    feature_importance = state[
        "global_feature_importance"
        ].copy()
    st.bar_chart(
        feature_importance.set_index("feature"),
        y="mean_absolute_shap_value"
        )
    st.dataframe(
        feature_importance,
        hide_index=True,
        width="stretch",
        key="global_importance_table",
        column_config={
            "mean_absolute_shap_value": st.column_config.NumberColumn(
                "Mean absolute SHAP value",
                format="%.6f"
                )
            }
        )


def _render_alerts(state):
    st.title("Alerts List")
    st.caption(
        "Model-generated alerts from the chronological held-out set."
        )

    risk_model = state["risk_model"]
    minimum_risk = st.slider(
        "Minimum risk score",
        min_value=float(risk_model.decision_threshold),
        max_value=1.0,
        value=float(risk_model.decision_threshold),
        step=0.01,
        key="minimum_risk"
        )
    monitored_transactions = state[
        "monitored_transactions"
        ]
    alerts = monitored_transactions.loc[
        (
            monitored_transactions["predicted_label"] == 1
            )
        & (
            monitored_transactions["risk_score"] >= minimum_risk
            )
        ].copy()
    alerts = alerts.reset_index().sort_values(
        ["risk_score", "transaction_id"],
        ascending=[False, True],
        kind="mergesort"
        )

    st.metric("Matching alerts", f"{len(alerts):,}")
    if alerts.empty:
        st.info("No model alerts meet the selected risk threshold.")
        return

    alert_columns = [
        "transaction_id",
        "timestamp",
        "risk_score",
        "actual_label",
        "fraud_type",
        "customer_id",
        "account_id",
        "beneficiary_id",
        "amount",
        "channel",
        "device",
        "ip_address"
        ]
    st.dataframe(
        alerts.loc[:, alert_columns],
        hide_index=True,
        width="stretch",
        key="alerts_table",
        column_config={
            "amount": st.column_config.NumberColumn(
                "Amount",
                format="£%.2f"
                ),
            "risk_score": st.column_config.NumberColumn(
                "Risk score",
                format="%.4f"
                ),
            "actual_label": "Known fraud label"
            }
        )


def _render_transaction_detail(state):
    st.title("Transaction Detail")
    st.caption(
        "Case review and local SHAP explanation for a held-out transaction."
        )

    monitored_transactions = state[
        "monitored_transactions"
        ]
    ordered_transactions = (
        monitored_transactions
        .reset_index()
        .sort_values(
            ["risk_score", "transaction_id"],
            ascending=[False, True],
            kind="mergesort"
            )
        .set_index("transaction_id")
        )
    transaction_id = st.selectbox(
        "Transaction",
        options=ordered_transactions.index.tolist(),
        key="transaction_selector"
        )
    transaction = ordered_transactions.loc[transaction_id]

    summary_columns = st.columns(4)
    summary_columns[0].metric(
        "Risk probability",
        f"{transaction['risk_score']:.2%}"
        )
    summary_columns[1].metric(
        "Model decision",
        "Alert" if transaction["predicted_label"] else "No alert"
        )
    summary_columns[2].metric(
        "Known fraud label",
        "Fraud" if transaction["actual_label"] else "Not fraud"
        )
    summary_columns[3].metric(
        "Fraud scenario",
        transaction["fraud_type"]
        )

    st.subheader("Transaction data")
    transaction_details = pd.DataFrame({
        "field": [
            "transaction_id",
            "customer_id",
            "account_id",
            "beneficiary_id",
            "sender",
            "receiver",
            "amount",
            "timestamp",
            "channel",
            "merchant",
            "merchant_category",
            "location",
            "device",
            "ip_address",
            "latitude",
            "longitude"
            ],
        "value": [
            transaction_id,
            transaction["customer_id"],
            transaction["account_id"],
            transaction["beneficiary_id"],
            transaction["sender"],
            transaction["receiver"],
            f"£{transaction['amount']:,.2f}",
            transaction["timestamp"].isoformat(sep=" "),
            transaction["channel"],
            transaction["merchant"],
            transaction["merchant_category"],
            transaction["location"],
            transaction["device"],
            transaction["ip_address"],
            f"{transaction['latitude']:.6f}",
            f"{transaction['longitude']:.6f}"
            ]
        })
    st.dataframe(
        transaction_details,
        hide_index=True,
        width="stretch",
        key="transaction_data_table"
        )

    explanation = state["shap_explainer"].explain_transaction(
        state["features"],
        transaction_id
        )
    st.subheader("SHAP explanation")
    st.caption(
        "Signed probability contributions: positive values raise risk "
        "and negative values lower it."
        )
    explanation_columns = st.columns(2)
    explanation_columns[0].metric(
        "Baseline probability",
        f"{explanation['base_value']:.6f}"
        )
    explanation_columns[1].metric(
        "Explained risk probability",
        f"{explanation['risk_score']:.6f}"
        )

    contributions = explanation[
        "feature_contributions"
        ].copy()
    st.bar_chart(
        contributions.set_index("feature"),
        y="shap_value"
        )
    st.dataframe(
        contributions,
        hide_index=True,
        width="stretch",
        key="local_shap_table",
        column_config={
            "feature_value": st.column_config.NumberColumn(
                "Feature value",
                format="%.6f"
                ),
            "shap_value": st.column_config.NumberColumn(
                "Signed SHAP value",
                format="%.6f"
                ),
            "absolute_shap_value": st.column_config.NumberColumn(
                "Absolute SHAP value",
                format="%.6f"
                )
            }
        )


def main():
    st.set_page_config(
        page_title="UK APP Scam Risk Intelligence",
        page_icon="🔎",
        layout="wide"
        )
    state = _build_dashboard_state()

    with st.sidebar:
        st.header("Risk Intelligence")
        page_name = st.radio(
            "Page",
            PAGE_NAMES,
            key="page_navigation"
            )
        st.caption(
            "XGBoost risk scores with probability-space SHAP explanations."
            )

    if page_name == "Overview":
        _render_overview(state)
    elif page_name == "Alerts List":
        _render_alerts(state)
    else:
        _render_transaction_detail(state)


if __name__ == "__main__":
    main()
