from dataclasses import replace
from datetime import timedelta

import random


INVESTMENT_SCAM = "investment_scam"
MONEY_MULE_LAYERING = "money_mule_layering"


class FraudGenerator:
    def inject_investment_scam(self, transactions):
        """Return fraud replacements for selected baseline transactions."""
        transactions = list(transactions)
        if len(transactions) < 2:
            raise ValueError(
                "Investment scams require at least two transactions"
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
        scenario_end = ordered_transactions[-1].timestamp
        base_amount = max(
            2_500,
            max(
                transaction.amount
                for transaction in ordered_transactions
                )
            )
        amount_increase = random.uniform(1_000, 2_500)

        injected_transactions = []
        for index, transaction in enumerate(ordered_transactions):
            desired_timestamp = scenario_end - timedelta(
                days=2 * (len(ordered_transactions) - index - 1)
                )
            timestamp = max(
                transaction.timestamp,
                desired_timestamp
                )
            amount = round(
                base_amount + amount_increase * index,
                2
                )
            injected_transactions.append(
                replace(
                    transaction,
                    amount=amount,
                    timestamp=timestamp,
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

    def inject_money_mule_layering(self, transactions):
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
        scenario_end = ordered_transactions[-1].timestamp

        inbound_amounts = [
            round(random.uniform(1_000, 5_000), 2)
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
            desired_timestamp = scenario_end - timedelta(
                minutes=10 * (len(ordered_transactions) - index - 1)
                )
            timestamp = max(
                transaction.timestamp,
                desired_timestamp
                )
            injected_transactions.append(
                replace(
                    transaction,
                    sender=transaction.beneficiary_id,
                    receiver=transaction.account_id,
                    amount=inbound_amounts[index],
                    timestamp=timestamp,
                    merchant="Account Transfer",
                    merchant_category="Personal Transfer",
                    is_fraud=True,
                    fraud_type=MONEY_MULE_LAYERING
                    )
                )

        for index, transaction in enumerate(outbound_transactions):
            transaction_index = len(inbound_transactions) + index
            desired_timestamp = scenario_end - timedelta(
                minutes=(
                    10
                    * (len(ordered_transactions) - transaction_index - 1)
                    )
                )
            timestamp = max(
                transaction.timestamp,
                desired_timestamp
                )
            injected_transactions.append(
                replace(
                    transaction,
                    sender=transaction.account_id,
                    receiver=transaction.beneficiary_id,
                    amount=outbound_amounts[index],
                    timestamp=timestamp,
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
