from dataclasses import dataclass
from datetime import date

from faker import Faker
import random


@dataclass
class Beneficiary:
    beneficiary_id: str
    relationship: str
    added_date: date
    trusted_flag: bool
    country: str


class BeneficiaryGenerator:
    def __init__(self):
        self.fake = Faker("en_GB")
        self.beneficiary_counter = 1

    def generate_beneficiary(self):
        beneficiary_id = f"BEN{self.beneficiary_counter:06d}"
        relationship = random.choice([
            "Family",
            "Friend",
            "Business",
            "Unknown"
            ])
        added_date = self.fake.date_between(
            start_date="-5y",
            end_date="today"
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
