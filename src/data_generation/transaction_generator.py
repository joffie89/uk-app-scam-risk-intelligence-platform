from dataclasses import dataclass
from datetime import datetime, timedelta
import math

from faker import Faker
import random

from src.config.cities import UK_CITY_COORDINATES
from src.data_generation import DEFAULT_REFERENCE_DATETIME


_AMOUNT_DISTRIBUTIONS = {
    # minimum, median, maximum, lognormal sigma
    "Groceries": (2, 35, 350, 0.65),
    "Retail": (5, 60, 2_000, 1.00),
    "Utilities": (10, 90, 600, 0.55),
    "Transport": (2, 15, 400, 0.85),
    "Food & Dining": (4, 25, 300, 0.65),
    "Entertainment": (3, 15, 250, 0.60),
    "Travel": (20, 180, 3_000, 1.00),
    "Telecoms": (5, 35, 250, 0.55),
    "Personal Transfer": (5, 100, 1_900, 1.05),
    "Other Transfer": (5, 80, 1_900, 1.10)
    }
_LARGE_GENUINE_TRANSFER_CATEGORIES = {
    "Personal Transfer",
    "Other Transfer"
    }
_LARGE_GENUINE_TRANSFER_RATE = 0.03
_LARGE_GENUINE_TRANSFER_DISTRIBUTION = (
    2_000,
    5_500,
    15_000,
    0.55
    )


def _generate_amount(merchant_category):
    amount_distribution = _AMOUNT_DISTRIBUTIONS[merchant_category]
    if (
        merchant_category in _LARGE_GENUINE_TRANSFER_CATEGORIES
        and random.random() < _LARGE_GENUINE_TRANSFER_RATE
        ):
        amount_distribution = (
            _LARGE_GENUINE_TRANSFER_DISTRIBUTION
            )

    (
        minimum_amount,
        median_amount,
        maximum_amount,
        sigma
        ) = amount_distribution
    amount = random.lognormvariate(
        math.log(median_amount),
        sigma
        )
    amount = min(max(amount, minimum_amount), maximum_amount)
    return round(amount, 2)


@dataclass
class Transaction:
    transaction_id: str
    customer_id: str
    account_id: str
    beneficiary_id: str
    sender: str
    receiver: str
    amount: float
    timestamp: datetime
    channel: str
    merchant: str
    merchant_category: str
    location: str
    device: str
    ip_address: str
    latitude: float
    longitude: float
    is_fraud: bool = False
    fraud_type: str | None = None


class TransactionGenerator:
    def __init__(
        self,
        reference_datetime=DEFAULT_REFERENCE_DATETIME
        ):
        if not isinstance(reference_datetime, datetime):
            raise TypeError(
                "reference_datetime must be a datetime"
                )
        if reference_datetime.tzinfo is not None:
            raise ValueError(
                "reference_datetime must not include a timezone"
                )

        self.fake = Faker("en_GB")
        self.transaction_counter = 1
        self.device_ip_addresses = {}
        self.reference_datetime = reference_datetime

    def generate_transaction(
        self,
        customer,
        account,
        device,
        beneficiary,
        timestamp=None
        ):
        if account.customer_id != customer.customer_id:
            raise ValueError(
                "Account customer_id must match the supplied customer"
                )
        if customer.city not in UK_CITY_COORDINATES:
            raise ValueError(
                "Customer city must have configured coordinates"
                )

        transaction_id = f"TXN{self.transaction_counter:06d}"
        customer_id = customer.customer_id
        account_id = account.account_id
        beneficiary_id = beneficiary.beneficiary_id
        sender = account_id
        receiver = beneficiary_id

        if device.device_type == "Desktop":
            channel = "Online Banking"
        else:
            channel = "Mobile Banking"

        business_merchants = {
            "Groceries": ["Tesco", "Sainsbury's"],
            "Retail": ["Amazon UK", "John Lewis"],
            "Utilities": ["British Gas", "Octopus Energy"],
            "Transport": ["Transport for London", "National Rail"],
            "Food & Dining": ["Deliveroo", "Just Eat"],
            "Entertainment": ["Netflix"],
            "Travel": ["Premier Inn"],
            "Telecoms": ["Vodafone", "BT"]
            }

        if beneficiary.relationship == "Business":
            merchant_category = random.choice(
                list(business_merchants.keys())
                )
            merchant = random.choice(
                business_merchants[merchant_category]
                )
        elif beneficiary.relationship in ["Family", "Friend"]:
            merchant_category = "Personal Transfer"
            merchant = f"{beneficiary.relationship} Transfer"
        else:
            merchant_category = "Other Transfer"
            merchant = "Other Payee"

        amount = _generate_amount(merchant_category)

        start_timestamp, end_timestamp = self._timestamp_bounds(
            customer,
            account,
            beneficiary
            )
        if timestamp is None:
            timestamp = self.fake.date_time_between(
                start_date=start_timestamp,
                end_date=end_timestamp
                )
        else:
            if not isinstance(timestamp, datetime):
                raise TypeError("timestamp must be a datetime")
            if timestamp.tzinfo is not None:
                raise ValueError("timestamp must not include a timezone")
            if not start_timestamp <= timestamp <= end_timestamp:
                raise ValueError(
                    "timestamp must be within the valid transaction window"
                    )

        if random.random() < 0.85:
            location = customer.city
        else:
            location = random.choice(
                list(UK_CITY_COORDINATES.keys())
                )
        base_latitude, base_longitude = UK_CITY_COORDINATES[location]
        latitude = round(
            base_latitude + random.uniform(-0.05, 0.05),
            6
            )
        longitude = round(
            base_longitude + random.uniform(-0.05, 0.05),
            6
            )

        if device.device_id not in self.device_ip_addresses:
            self.device_ip_addresses[
                device.device_id
                ] = self.fake.ipv4_public()
        ip_address = self.device_ip_addresses[device.device_id]

        transaction = Transaction(
            transaction_id=transaction_id,
            customer_id=customer_id,
            account_id=account_id,
            beneficiary_id=beneficiary_id,
            sender=sender,
            receiver=receiver,
            amount=amount,
            timestamp=timestamp,
            channel=channel,
            merchant=merchant,
            merchant_category=merchant_category,
            location=location,
            device=device.device_id,
            ip_address=ip_address,
            latitude=latitude,
            longitude=longitude
            )
        device.last_seen = max(device.last_seen, timestamp)
        self.transaction_counter += 1
        return transaction

    def generate_transaction_burst(
        self,
        customer,
        account,
        device,
        beneficiary,
        transaction_count
        ):
        """Generate two to five genuine payments within three hours."""
        if (
            isinstance(transaction_count, bool)
            or not isinstance(transaction_count, int)
            or not 2 <= transaction_count <= 5
            ):
            raise ValueError(
                "transaction_count must be an integer from two to five"
                )

        start_timestamp, end_timestamp = self._timestamp_bounds(
            customer,
            account,
            beneficiary
            )
        gaps = [
            timedelta(minutes=random.randint(5, 45))
            for _ in range(transaction_count - 1)
            ]
        burst_duration = sum(gaps, timedelta())
        latest_start_timestamp = end_timestamp - burst_duration
        if start_timestamp > latest_start_timestamp:
            raise ValueError(
                "Linked entities do not have enough history for a burst"
                )
        burst_start_timestamp = self.fake.date_time_between(
            start_date=start_timestamp,
            end_date=latest_start_timestamp
            )
        burst_timestamps = [burst_start_timestamp]
        for gap in gaps:
            burst_timestamps.append(burst_timestamps[-1] + gap)

        return [
            self.generate_transaction(
                customer,
                account,
                device,
                beneficiary,
                timestamp=burst_timestamp
                )
            for burst_timestamp in burst_timestamps
            ]

    def _timestamp_bounds(
        self,
        customer,
        account,
        beneficiary
        ):
        now = self.reference_datetime
        earliest_date = max(
            customer.customer_since,
            account.opened_date,
            beneficiary.added_date
            )
        earliest_timestamp = datetime.combine(
            earliest_date,
            datetime.min.time()
            )
        recent_timestamp = now - timedelta(days=365)
        start_timestamp = max(
            earliest_timestamp,
            recent_timestamp
            )
        if start_timestamp > now:
            raise ValueError(
                "Linked entities must not have future relationship dates"
                )
        return start_timestamp, now
