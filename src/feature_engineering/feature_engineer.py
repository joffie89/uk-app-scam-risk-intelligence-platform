from collections import defaultdict, deque
from datetime import timedelta
import math

import pandas as pd

from src.config.cities import UK_CITY_COORDINATES


FEATURE_COLUMNS = (
    "transaction_amount",
    "prior_transaction_count_1h",
    "prior_transaction_count_24h",
    "prior_transaction_amount_24h",
    "amount_to_prior_7d_mean_ratio",
    "minutes_since_previous_transaction",
    "fan_in_24h",
    "fan_out_24h",
    "is_first_payment_to_beneficiary",
    "prior_beneficiary_payment_count_30d",
    "prior_other_customers_on_device_30d",
    "prior_other_customers_on_ip_30d",
    "account_age_days",
    "distance_from_home_km",
    "distance_anomaly_flag"
    )

DISTANCE_ANOMALY_THRESHOLD_KM = 100
NO_PREVIOUS_TRANSACTION_MINUTES = 10_080


def _remove_expired(history, cutoff):
    while history and history[0][0] <= cutoff:
        history.popleft()


def _haversine_distance(
    latitude,
    longitude,
    home_latitude,
    home_longitude
    ):
    earth_radius_km = 6_371
    latitude_delta = math.radians(home_latitude - latitude)
    longitude_delta = math.radians(home_longitude - longitude)
    latitude = math.radians(latitude)
    home_latitude = math.radians(home_latitude)

    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude)
        * math.cos(home_latitude)
        * math.sin(longitude_delta / 2) ** 2
        )
    central_angle = 2 * math.asin(
        math.sqrt(min(1, max(0, haversine)))
        )
    return earth_radius_km * central_angle


class FeatureEngineer:
    def build_features(self, transactions, accounts, customers):
        """Build numeric features using only strictly prior transactions."""
        if not isinstance(transactions, pd.DataFrame):
            raise TypeError("transactions must be a pandas DataFrame")
        if not isinstance(accounts, pd.DataFrame):
            raise TypeError("accounts must be a pandas DataFrame")
        if not isinstance(customers, pd.DataFrame):
            raise TypeError("customers must be a pandas DataFrame")

        transaction_columns = {
            "transaction_id",
            "customer_id",
            "account_id",
            "beneficiary_id",
            "sender",
            "receiver",
            "amount",
            "timestamp",
            "device",
            "ip_address",
            "latitude",
            "longitude"
            }
        account_columns = {
            "account_id",
            "customer_id",
            "opened_date"
            }
        customer_columns = {
            "customer_id",
            "city"
            }

        missing_transaction_columns = (
            transaction_columns - set(transactions.columns)
            )
        if missing_transaction_columns:
            raise ValueError(
                "transactions is missing columns: "
                f"{sorted(missing_transaction_columns)}"
                )
        missing_account_columns = account_columns - set(accounts.columns)
        if missing_account_columns:
            raise ValueError(
                "accounts is missing columns: "
                f"{sorted(missing_account_columns)}"
                )
        missing_customer_columns = customer_columns - set(customers.columns)
        if missing_customer_columns:
            raise ValueError(
                "customers is missing columns: "
                f"{sorted(missing_customer_columns)}"
                )

        transactions = transactions.copy()
        accounts = accounts.copy()
        customers = customers.copy()

        if transactions["transaction_id"].duplicated().any():
            raise ValueError("transaction_id values must be unique")
        if accounts["account_id"].duplicated().any():
            raise ValueError("account_id values must be unique")
        if customers["customer_id"].duplicated().any():
            raise ValueError("customer_id values must be unique")

        if transactions[list(transaction_columns)].isnull().any().any():
            raise ValueError("transactions contains missing required values")
        if accounts[list(account_columns)].isnull().any().any():
            raise ValueError("accounts contains missing required values")
        if customers[list(customer_columns)].isnull().any().any():
            raise ValueError("customers contains missing required values")

        try:
            transactions["timestamp"] = pd.to_datetime(
                transactions["timestamp"],
                errors="raise"
                )
            accounts["opened_date"] = pd.to_datetime(
                accounts["opened_date"],
                errors="raise"
                )
            for column in ["amount", "latitude", "longitude"]:
                transactions[column] = pd.to_numeric(
                    transactions[column],
                    errors="raise"
                    )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "dates and numeric transaction values must be valid"
                ) from error

        if transactions["timestamp"].isnull().any():
            raise ValueError("transaction timestamps must be valid")
        if accounts["opened_date"].isnull().any():
            raise ValueError("account opened dates must be valid")

        identifier_columns = [
            "transaction_id",
            "customer_id",
            "account_id",
            "beneficiary_id",
            "sender",
            "receiver",
            "device",
            "ip_address"
            ]
        if any(
            not isinstance(value, str) or not value
            for column in identifier_columns
            for value in transactions[column]
            ):
            raise ValueError(
                "transaction identifiers must be non-empty strings"
                )

        numeric_columns = ["amount", "latitude", "longitude"]
        if any(
            not math.isfinite(float(value))
            for column in numeric_columns
            for value in transactions[column]
            ):
            raise ValueError("transaction numeric values must be finite")
        if (transactions["amount"] <= 0).any():
            raise ValueError("transaction amounts must be positive")
        if (
            (transactions["latitude"] < -90).any()
            or (transactions["latitude"] > 90).any()
            or (transactions["longitude"] < -180).any()
            or (transactions["longitude"] > 180).any()
            ):
            raise ValueError("transaction coordinates are out of range")

        account_lookup = {
            row.account_id: {
                "customer_id": row.customer_id,
                "opened_date": row.opened_date
                }
            for row in accounts[
                ["account_id", "customer_id", "opened_date"]
                ].itertuples(index=False)
            }
        customer_lookup = {
            row.customer_id: row.city
            for row in customers[
                ["customer_id", "city"]
                ].itertuples(index=False)
            }

        for transaction in transactions.itertuples(index=False):
            if transaction.account_id not in account_lookup:
                raise ValueError(
                    f"Unknown account_id: {transaction.account_id}"
                    )
            account = account_lookup[transaction.account_id]
            if account["customer_id"] != transaction.customer_id:
                raise ValueError(
                    "Transaction customer_id does not own its account_id"
                    )
            if transaction.customer_id not in customer_lookup:
                raise ValueError(
                    f"Unknown customer_id: {transaction.customer_id}"
                    )
            customer_city = customer_lookup[transaction.customer_id]
            if customer_city not in UK_CITY_COORDINATES:
                raise ValueError(
                    f"Customer city has no coordinates: {customer_city}"
                    )
            sender_is_account = (
                transaction.sender == transaction.account_id
                )
            receiver_is_account = (
                transaction.receiver == transaction.account_id
                )
            if sender_is_account == receiver_is_account:
                raise ValueError(
                    "Each transaction account_id must be exactly one endpoint"
                    )
            if (
                sender_is_account
                and transaction.receiver != transaction.beneficiary_id
                ):
                raise ValueError(
                    "Outbound receiver must match beneficiary_id"
                    )
            if (
                receiver_is_account
                and transaction.sender != transaction.beneficiary_id
                ):
                raise ValueError(
                    "Inbound sender must match beneficiary_id"
                    )
            if (
                transaction.timestamp.date()
                < account["opened_date"].date()
                ):
                raise ValueError(
                    "Transaction timestamp cannot predate account opening"
                    )

        integer_features = [
            "prior_transaction_count_1h",
            "prior_transaction_count_24h",
            "fan_in_24h",
            "fan_out_24h",
            "is_first_payment_to_beneficiary",
            "prior_beneficiary_payment_count_30d",
            "prior_other_customers_on_device_30d",
            "prior_other_customers_on_ip_30d",
            "account_age_days",
            "distance_anomaly_flag"
            ]
        if transactions.empty:
            empty_features = pd.DataFrame({
                feature: pd.Series(
                    dtype=(
                        "int64"
                        if feature in integer_features
                        else "float64"
                        )
                    )
                for feature in FEATURE_COLUMNS
                })
            empty_features.index = pd.Index(
                [],
                name="transaction_id"
                )
            return empty_features

        original_transaction_ids = transactions[
            "transaction_id"
            ].tolist()
        ordered_transactions = transactions.sort_values(
            ["timestamp", "transaction_id"],
            kind="mergesort"
            )

        account_history_1h = defaultdict(deque)
        account_history_24h = defaultdict(deque)
        account_history_7d = defaultdict(deque)
        beneficiary_history_30d = defaultdict(deque)
        device_history_30d = defaultdict(deque)
        ip_history_30d = defaultdict(deque)
        previous_account_timestamp = {}
        seen_customer_beneficiaries = set()
        feature_rows = []

        one_hour = timedelta(hours=1)
        twenty_four_hours = timedelta(hours=24)
        seven_days = timedelta(days=7)
        thirty_days = timedelta(days=30)

        for timestamp, timestamp_group in ordered_transactions.groupby(
            "timestamp",
            sort=False
            ):
            for transaction in timestamp_group.itertuples(index=False):
                history_1h = account_history_1h[
                    transaction.account_id
                    ]
                history_24h = account_history_24h[
                    transaction.account_id
                    ]
                history_7d = account_history_7d[
                    transaction.account_id
                    ]
                _remove_expired(history_1h, timestamp - one_hour)
                _remove_expired(
                    history_24h,
                    timestamp - twenty_four_hours
                    )
                _remove_expired(history_7d, timestamp - seven_days)

                prior_transaction_count_1h = len(history_1h)
                prior_transaction_count_24h = len(history_24h)
                prior_transaction_amount_24h = round(
                    sum(amount for _, amount, _, _ in history_24h),
                    2
                    )
                if history_7d:
                    prior_mean_amount = sum(
                        amount for _, amount in history_7d
                        ) / len(history_7d)
                    amount_to_prior_7d_mean_ratio = round(
                        transaction.amount / prior_mean_amount,
                        6
                        )
                else:
                    amount_to_prior_7d_mean_ratio = 1.0

                if transaction.account_id in previous_account_timestamp:
                    previous_timestamp = previous_account_timestamp[
                        transaction.account_id
                        ]
                    minutes_since_previous_transaction = min(
                        (
                            timestamp - previous_timestamp
                            ).total_seconds() / 60,
                        NO_PREVIOUS_TRANSACTION_MINUTES
                        )
                else:
                    minutes_since_previous_transaction = (
                        NO_PREVIOUS_TRANSACTION_MINUTES
                        )
                minutes_since_previous_transaction = round(
                    minutes_since_previous_transaction,
                    2
                    )

                inbound_counterparties = {
                    sender
                    for _, _, sender, receiver in history_24h
                    if receiver == transaction.account_id
                    }
                outbound_counterparties = {
                    receiver
                    for _, _, sender, receiver in history_24h
                    if sender == transaction.account_id
                    }
                fan_in_24h = len(inbound_counterparties)
                fan_out_24h = len(outbound_counterparties)

                customer_beneficiary = (
                    transaction.customer_id,
                    transaction.beneficiary_id
                    )
                is_outbound = (
                    transaction.sender == transaction.account_id
                    )
                is_first_payment_to_beneficiary = int(
                    is_outbound
                    and customer_beneficiary
                    not in seen_customer_beneficiaries
                    )
                if is_outbound:
                    beneficiary_history = beneficiary_history_30d[
                        customer_beneficiary
                        ]
                    _remove_expired(
                        beneficiary_history,
                        timestamp - thirty_days
                        )
                    prior_beneficiary_payment_count_30d = len(
                        beneficiary_history
                        )
                else:
                    prior_beneficiary_payment_count_30d = 0

                device_history = device_history_30d[transaction.device]
                ip_history = ip_history_30d[transaction.ip_address]
                _remove_expired(
                    device_history,
                    timestamp - thirty_days
                    )
                _remove_expired(
                    ip_history,
                    timestamp - thirty_days
                    )
                prior_other_customers_on_device_30d = len({
                    customer_id
                    for _, customer_id in device_history
                    if customer_id != transaction.customer_id
                    })
                prior_other_customers_on_ip_30d = len({
                    customer_id
                    for _, customer_id in ip_history
                    if customer_id != transaction.customer_id
                    })

                account = account_lookup[transaction.account_id]
                account_age_days = (
                    timestamp.date()
                    - account["opened_date"].date()
                    ).days
                customer_city = customer_lookup[
                    transaction.customer_id
                    ]
                home_latitude, home_longitude = (
                    UK_CITY_COORDINATES[customer_city]
                    )
                distance_from_home_km = round(
                    _haversine_distance(
                        float(transaction.latitude),
                        float(transaction.longitude),
                        home_latitude,
                        home_longitude
                        ),
                    3
                    )
                distance_anomaly_flag = int(
                    distance_from_home_km
                    > DISTANCE_ANOMALY_THRESHOLD_KM
                    )

                feature_rows.append({
                    "transaction_id": transaction.transaction_id,
                    "transaction_amount": float(transaction.amount),
                    "prior_transaction_count_1h": (
                        prior_transaction_count_1h
                        ),
                    "prior_transaction_count_24h": (
                        prior_transaction_count_24h
                        ),
                    "prior_transaction_amount_24h": (
                        prior_transaction_amount_24h
                        ),
                    "amount_to_prior_7d_mean_ratio": (
                        amount_to_prior_7d_mean_ratio
                        ),
                    "minutes_since_previous_transaction": (
                        minutes_since_previous_transaction
                        ),
                    "fan_in_24h": fan_in_24h,
                    "fan_out_24h": fan_out_24h,
                    "is_first_payment_to_beneficiary": (
                        is_first_payment_to_beneficiary
                        ),
                    "prior_beneficiary_payment_count_30d": (
                        prior_beneficiary_payment_count_30d
                        ),
                    "prior_other_customers_on_device_30d": (
                        prior_other_customers_on_device_30d
                        ),
                    "prior_other_customers_on_ip_30d": (
                        prior_other_customers_on_ip_30d
                        ),
                    "account_age_days": account_age_days,
                    "distance_from_home_km": distance_from_home_km,
                    "distance_anomaly_flag": distance_anomaly_flag
                    })

            for transaction in timestamp_group.itertuples(index=False):
                history_values = (
                    timestamp,
                    float(transaction.amount)
                    )
                account_history_1h[
                    transaction.account_id
                    ].append(history_values)
                account_history_24h[
                    transaction.account_id
                    ].append((
                        timestamp,
                        float(transaction.amount),
                        transaction.sender,
                        transaction.receiver
                        ))
                account_history_7d[
                    transaction.account_id
                    ].append(history_values)
                previous_account_timestamp[
                    transaction.account_id
                    ] = timestamp

                if transaction.sender == transaction.account_id:
                    customer_beneficiary = (
                        transaction.customer_id,
                        transaction.beneficiary_id
                        )
                    seen_customer_beneficiaries.add(
                        customer_beneficiary
                        )
                    beneficiary_history_30d[
                        customer_beneficiary
                        ].append((timestamp,))

                device_history_30d[
                    transaction.device
                    ].append((
                        timestamp,
                        transaction.customer_id
                        ))
                ip_history_30d[
                    transaction.ip_address
                    ].append((
                        timestamp,
                        transaction.customer_id
                        ))

        features = pd.DataFrame(feature_rows).set_index(
            "transaction_id"
            )
        features = features.reindex(original_transaction_ids)
        features = features.loc[:, FEATURE_COLUMNS]
        features[integer_features] = features[
            integer_features
            ].astype("int64")
        float_features = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in integer_features
            ]
        features[float_features] = features[
            float_features
            ].astype("float64")

        if features.isnull().any().any():
            raise ValueError("Engineered features contain missing values")
        if any(
            not math.isfinite(float(value))
            for feature in FEATURE_COLUMNS
            for value in features[feature]
            ):
            raise ValueError("Engineered features must be finite")

        return features
