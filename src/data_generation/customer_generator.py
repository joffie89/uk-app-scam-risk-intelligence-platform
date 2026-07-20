from dataclasses import dataclass
from datetime import date

from faker import Faker
import random

from src.config.occupations import OCCUPATION_INCOME_RANGES
from src.config.cities import UK_CITY_POSTCODES
from src.config.risk_profiles import RISK_SEGMENTS

@dataclass
class Customer:
    customer_id: str
    first_name: str
    last_name: str
    gender: str
    date_of_birth: date
    age: int
    occupation: str
    annual_income: float
    city: str
    postcode: str
    customer_since: date
    risk_segment: str
    email: str
    phone_number: str    


class CustomerGenerator:
    def __init__(self):
        self.fake= Faker("en_GB")
        self.customer_counter= 1

    def generate_customer(self):
        customer_id = f"CUST{self.customer_counter:06d}"
        gender = random.choice(["Male","Female"])
        if gender == "Male":
            first_name= self.fake.first_name_male()
        else:
            first_name=self.fake.first_name_female()

        last_name= self.fake.last_name()
        today= date.today()

        date_of_birth= self.fake.date_of_birth(
            minimum_age=18,
            maximum_age=80
        )
        
        age = (
            today.year
              - date_of_birth.year
              - (
                  (today.month, today.day)
                    < (date_of_birth.month, date_of_birth.day)
                      )
                      )
        customer_since = self.fake.date_between(
            start_date="-15y",
            end_date="today"
            )

        occupation = random.choice(
             list(OCCUPATION_INCOME_RANGES.keys())
          )
        min_income, max_income = OCCUPATION_INCOME_RANGES[occupation]
        annual_income = round(random.uniform(min_income, max_income),2)
        risk_segment = random.choice(RISK_SEGMENTS)

        city = random.choice(
            list(UK_CITY_POSTCODES.keys())
            )
        postcode_prefix = random.choice(
            UK_CITY_POSTCODES[city]
            )
        postcode = (
            f"{postcode_prefix} "
            f"{random.randint(1,9)}"
            f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
            f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}"
            )
        email = self.fake.email()
        phone_number = self.fake.phone_number()

        customer = Customer(
            customer_id=customer_id,
            first_name=first_name,
            last_name=last_name,
            gender=gender,
            date_of_birth=date_of_birth,
            age=age,
            occupation=occupation,
            annual_income=annual_income,
            city=city,
            postcode=postcode,
            customer_since=customer_since,
            risk_segment=risk_segment,
            email=email,
            phone_number=phone_number
            )
        self.customer_counter += 1
        return customer
        

