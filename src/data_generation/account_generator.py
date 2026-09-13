from dataclasses import dataclass
from datetime import date

from faker import Faker
import random


@dataclass
class Account:
    account_id: str
    customer_id: str
    account_number: str
    sort_code: str
    iban: str
    account_type: str
    account_status: str
    currency: str
    opened_date: date
    balance: float
    overdraft_limit: float


class AccountGenerator:
    def __init__(self):
        self.fake = Faker("en_GB")
        self.account_counter = 1

    def generate_account(self, customer_id):
        account_id = f"ACC{self.account_counter:06d}"
        iban = self.fake.iban()
        account_number = iban[14:]
        sort_code = (
            f"{iban[8:10]}-"
            f"{iban[10:12]}-"
            f"{iban[12:14]}"
            )
        account_type = random.choice([
            "Current",
            "Savings"
            ])
        account_status = random.choice([
            "Active",
            "Dormant",
            "Frozen",
            "Closed"
            ])
        currency = "GBP"
        opened_date = self.fake.date_between(
            start_date="-15y",
            end_date="today"
            )

        if account_type == "Current":
            overdraft_limit = random.choice([
                0.00,
                500.00,
                1_000.00,
                2_000.00
                ])
        else:
            overdraft_limit = 0.00

        if account_status == "Closed":
            balance = 0.00
        else:
            balance = round(
                random.uniform(-overdraft_limit, 50_000),
                2
                )

        account = Account(
            account_id=account_id,
            customer_id=customer_id,
            account_number=account_number,
            sort_code=sort_code,
            iban=iban,
            account_type=account_type,
            account_status=account_status,
            currency=currency,
            opened_date=opened_date,
            balance=balance,
            overdraft_limit=overdraft_limit
            )
        self.account_counter += 1
        return account
