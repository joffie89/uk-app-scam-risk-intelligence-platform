from dataclasses import dataclass
from datetime import datetime, timedelta

from faker import Faker
import random

from src.config.cities import UK_CITY_COORDINATES


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


class TransactionGenerator:
    def __init__(self):
        self.fake = Faker("en_GB")
        self.transaction_counter = 1
        self.device_ip_addresses = {}

    def generate_transaction(
        self,
        customer,
        account,
        device,
        beneficiary
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

        amount_ranges = {
            "Groceries": (5, 200),
            "Retail": (10, 800),
            "Utilities": (20, 350),
            "Transport": (2, 250),
            "Food & Dining": (5, 150),
            "Entertainment": (5, 120),
            "Travel": (50, 1_500),
            "Telecoms": (10, 150),
            "Personal Transfer": (10, 2_000),
            "Other Transfer": (10, 1_000)
            }
        minimum_amount, maximum_amount = amount_ranges[
            merchant_category
            ]
        amount = round(
            random.uniform(minimum_amount, maximum_amount),
            2
            )

        now = datetime.now()
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
        timestamp = self.fake.date_time_between(
            start_date=start_timestamp,
            end_date=now
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
