from dataclasses import dataclass
from datetime import date, datetime

from dateutil.relativedelta import relativedelta
from faker import Faker
import random

from src.data_generation import DEFAULT_REFERENCE_DATETIME


@dataclass
class Beneficiary:
    beneficiary_id: str
    relationship: str
    added_date: date
    trusted_flag: bool
    country: str


class BeneficiaryGenerator:
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
        self.beneficiary_counter = 1
        self.reference_datetime = reference_datetime

    def generate_beneficiary(self):
        beneficiary_id = f"BEN{self.beneficiary_counter:06d}"
        relationship = random.choice([
            "Family",
            "Friend",
            "Business",
            "Unknown"
            ])
        reference_date = self.reference_datetime.date()
        added_date = self.fake.date_between_dates(
            date_start=(
                reference_date - relativedelta(years=5)
                ),
            date_end=reference_date
            )
        trusted_flag = random.choice([True, False])
        country = random.choice([
            "United Kingdom",
            "Ireland",
            "France",
            "Germany",
            "Spain",
            "United States"
            ])

        beneficiary = Beneficiary(
            beneficiary_id=beneficiary_id,
            relationship=relationship,
            added_date=added_date,
            trusted_flag=trusted_flag,
            country=country
            )
        self.beneficiary_counter += 1
        return beneficiary
