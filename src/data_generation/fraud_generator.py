from dataclasses import replace
from datetime import datetime, timedelta

import random


INVESTMENT_SCAM = "investment_scam"
MONEY_MULE_LAYERING = "money_mule_layering"


def _random_timestamps(
    ordered_transactions,
    minimum_gap_seconds,
    maximum_gap_seconds,
    earliest_timestamp=None,
    latest_timestamp=None
    ):
    gap_count = len(ordered_transactions) - 1
    if earliest_timestamp is None and latest_timestamp is None:
        gap_seconds = [
            random.randint(
                minimum_gap_seconds,
                maximum_gap_seconds
                )
            for _ in range(gap_count)
            ]
        timestamps = [ordered_transactions[-1].timestamp]
        for seconds in reversed(gap_seconds):
            timestamps.append(
                timestamps[-1] - timedelta(seconds=seconds)
                )
        return list(reversed(timestamps))
    if earliest_timestamp is None or latest_timestamp is None:
        raise ValueError(
            "Both scenario timestamp bounds must be supplied"
            )
    if (
        not isinstance(earliest_timestamp, datetime)
        or not isinstance(latest_timestamp, datetime)
        ):
        raise TypeError("Scenario timestamp bounds must be datetimes")
    if (
        earliest_timestamp.tzinfo is not None
        or latest_timestamp.tzinfo is not None
        ):
        raise ValueError(
            "Scenario timestamp bounds must not include timezones"
            )

    available_scenario_seconds = int((
        latest_timestamp - earliest_timestamp
        ).total_seconds())
    minimum_scenario_seconds = minimum_gap_seconds * gap_count
    if available_scenario_seconds < minimum_scenario_seconds:
        raise ValueError("Scenario timestamp range is too short")

    gap_seconds = []
    remaining_scenario_seconds = available_scenario_seconds
    for gap_index in range(gap_count):
        remaining_gap_count = gap_count - gap_index - 1
        maximum_current_gap_seconds = min(
            maximum_gap_seconds,
            (
                remaining_scenario_seconds
                - minimum_gap_seconds * remaining_gap_count
                )
            )
        current_gap_seconds = random.randint(
            minimum_gap_seconds,
            maximum_current_gap_seconds
            )
        gap_seconds.append(current_gap_seconds)
        remaining_scenario_seconds -= current_gap_seconds

    scenario_duration_seconds = sum(gap_seconds)
    latest_start_timestamp = (
        latest_timestamp
        - timedelta(seconds=scenario_duration_seconds)
        )
    available_start_seconds = int((
        latest_start_timestamp - earliest_timestamp
        ).total_seconds())
    scenario_start = earliest_timestamp + timedelta(
        seconds=random.randint(0, available_start_seconds)
        )
    timestamps = [scenario_start]
    for seconds in gap_seconds:
        timestamps.append(
            timestamps[-1] + timedelta(seconds=seconds)
            )
    return timestamps


class FraudGenerator:
    def inject_investment_scam(
        self,
        transactions,
        earliest_timestamp=None,
        latest_timestamp=None
        ):
        """Return fraud replacements for selected baseline transactions."""
        transactions = list(transactions)
        if not 3 <= len(transactions) <= 6:
            raise ValueError(
                "Investment scams require three to six transactions"
                )
        if len({
            transaction.transaction_id
            for transaction in transactions
            }) != len(transactions):
            raise ValueError(
                "Investment scam transaction IDs must be unique"
                )
        if len({
            transaction.timestamp
            for transaction in transactions
            }) != len(transactions):
            raise ValueError(
                "Investment scam transaction timestamps must be unique"
                )
        if any(
            transaction.is_fraud
            or transaction.fraud_type is not None
            for transaction in transactions
            ):
            raise ValueError(
                "Transactions must not already have fraud labels"
                )

        reference_transaction = transactions[0]
        shared_fields = [
            "customer_id",
            "account_id",
            "beneficiary_id",
            "device",
            "ip_address"
            ]
        if any(
            getattr(transaction, field)
            != getattr(reference_transaction, field)
            for transaction in transactions
            for field in shared_fields
            ):
            raise ValueError(
                "Investment scam transactions must share one source, "
                "beneficiary, and device"
                )
        if any(
            transaction.sender != transaction.account_id
            or transaction.receiver != transaction.beneficiary_id
            for transaction in transactions
            ):
            raise ValueError(
                "Investment scam inputs must be outbound transactions"
                )

        ordered_transactions = sorted(
            transactions,
            key=lambda transaction: transaction.timestamp
            )
        scenario_timestamps = _random_timestamps(
            ordered_transactions,
            minimum_gap_seconds=24 * 60 * 60,
            maximum_gap_seconds=7 * 24 * 60 * 60,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
            )
        if (
            earliest_timestamp is None
            and scenario_timestamps[0] < ordered_transactions[0].timestamp
            ):
            raise ValueError(
                "Investment scam timestamp range is too short"
                )

        first_amount = round(random.uniform(50, 500), 2)
        minimum_final_amount = max(
            500,
            first_amount + 25 * (len(ordered_transactions) - 1)
            )
        final_amount = round(
            random.uniform(minimum_final_amount, 8_000),
            2
            )
        growth_weights = [
            random.uniform(0.75, 1.25)
            for _ in range(len(ordered_transactions) - 1)
            ]
        total_growth_weight = sum(growth_weights)
        cumulative_growth_weight = 0.0
        amounts = [first_amount]
        for index, growth_weight in enumerate(growth_weights):
            cumulative_growth_weight += growth_weight
            if index == len(growth_weights) - 1:
                amounts.append(final_amount)
            else:
                amounts.append(round(
                    first_amount
                    + (final_amount - first_amount)
                    * cumulative_growth_weight
                    / total_growth_weight,
                    2
                    ))

        injected_transactions = []
        for index, transaction in enumerate(ordered_transactions):
            injected_transactions.append(
                replace(
                    transaction,
                    amount=amounts[index],
                    timestamp=scenario_timestamps[index],
                    merchant="Investment Platform",
                    merchant_category="Investment",
                    is_fraud=True,
                    fraud_type=INVESTMENT_SCAM
                    )
                )

        if len({
            transaction.timestamp
            for transaction in injected_transactions
            }) != len(injected_transactions):
            raise ValueError(
                "Injected investment scam timestamps must be unique"
                )

        return injected_transactions

    def inject_money_mule_layering(
        self,
        transactions,
        earliest_timestamp=None,
        latest_timestamp=None
        ):
        """Return fraud replacements for selected baseline transactions."""
        transactions = list(transactions)
        if len(transactions) < 4:
            raise ValueError(
                "Money-mule layering requires at least four transactions"
                )
        if len({
            transaction.transaction_id
            for transaction in transactions
            }) != len(transactions):
            raise ValueError(
                "Money-mule transaction IDs must be unique"
                )
        if len({
            transaction.timestamp
            for transaction in transactions
            }) != len(transactions):
            raise ValueError(
                "Money-mule transaction timestamps must be unique"
                )
        if any(
            transaction.is_fraud
            or transaction.fraud_type is not None
            for transaction in transactions
            ):
            raise ValueError(
                "Transactions must not already have fraud labels"
                )

        reference_transaction = transactions[0]
        shared_fields = [
            "customer_id",
            "account_id",
            "device",
            "ip_address"
            ]
        if any(
            getattr(transaction, field)
            != getattr(reference_transaction, field)
            for transaction in transactions
            for field in shared_fields
            ):
            raise ValueError(
                "Money-mule transactions must share one account and device"
                )
        if len({
            transaction.beneficiary_id
            for transaction in transactions
            }) != len(transactions):
            raise ValueError(
                "Money-mule transactions require distinct beneficiaries"
                )
        if any(
            transaction.sender != transaction.account_id
            or transaction.receiver != transaction.beneficiary_id
            for transaction in transactions
            ):
            raise ValueError(
                "Money-mule inputs must be outbound transactions"
                )

        ordered_transactions = sorted(
            transactions,
            key=lambda transaction: transaction.timestamp
            )
        inbound_transactions = ordered_transactions[:-2]
        outbound_transactions = ordered_transactions[-2:]
        scenario_timestamps = _random_timestamps(
            ordered_transactions,
            minimum_gap_seconds=5 * 60,
            maximum_gap_seconds=180 * 60,
            earliest_timestamp=earliest_timestamp,
            latest_timestamp=latest_timestamp
            )
        if (
            earliest_timestamp is None
            and any(
                timestamp < transaction.timestamp
                for transaction, timestamp in zip(
                    ordered_transactions,
                    scenario_timestamps
                    )
                )
            ):
            raise ValueError(
                "Money-mule timestamp range is too short"
                )

        inbound_amounts = [
            round(random.uniform(200, 3_000), 2)
            for _ in inbound_transactions
            ]
        total_inbound = round(sum(inbound_amounts), 2)
        total_outbound = round(
            total_inbound * random.uniform(0.90, 0.97),
            2
            )
        first_outbound = round(
            total_outbound * random.uniform(0.40, 0.60),
            2
            )
        outbound_amounts = [
            first_outbound,
            round(total_outbound - first_outbound, 2)
            ]

        injected_transactions = []
        for index, transaction in enumerate(inbound_transactions):
            injected_transactions.append(
                replace(
                    transaction,
                    sender=transaction.beneficiary_id,
                    receiver=transaction.account_id,
                    amount=inbound_amounts[index],
                    timestamp=scenario_timestamps[index],
                    merchant="Account Transfer",
                    merchant_category="Personal Transfer",
                    is_fraud=True,
                    fraud_type=MONEY_MULE_LAYERING
                    )
                )

        for index, transaction in enumerate(outbound_transactions):
            transaction_index = len(inbound_transactions) + index
            injected_transactions.append(
                replace(
                    transaction,
                    sender=transaction.account_id,
                    receiver=transaction.beneficiary_id,
                    amount=outbound_amounts[index],
                    timestamp=scenario_timestamps[transaction_index],
                    merchant="Account Transfer",
                    merchant_category="Personal Transfer",
                    is_fraud=True,
                    fraud_type=MONEY_MULE_LAYERING
                    )
                )

        if len({
            transaction.timestamp
            for transaction in injected_transactions
            }) != len(injected_transactions):
            raise ValueError(
                "Injected money-mule timestamps must be unique"
                )

        return injected_transactions
