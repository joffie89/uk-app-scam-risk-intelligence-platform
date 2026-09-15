from dataclasses import dataclass
from datetime import date, datetime

from dateutil.relativedelta import relativedelta
from faker import Faker
import random

from src.data_generation import DEFAULT_REFERENCE_DATETIME


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
        self.account_counter = 1
        self.reference_datetime = reference_datetime

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
        reference_date = self.reference_datetime.date()
        opened_date = self.fake.date_between_dates(
            date_start=(
                reference_date - relativedelta(years=15)
                ),
            date_end=reference_date
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
