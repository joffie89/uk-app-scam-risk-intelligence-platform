from dataclasses import dataclass
from datetime import datetime

from dateutil.relativedelta import relativedelta
from faker import Faker
import random

from src.data_generation import DEFAULT_REFERENCE_DATETIME


@dataclass
class Device:
    device_id: str
    device_type: str
    browser: str
    OS: str
    trusted_device: bool
    device_fingerprint: str
    last_seen: datetime


class DeviceGenerator:
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
        self.device_counter = 1
        self.reference_datetime = reference_datetime

    def generate_device(self):
        device_id = f"DEV{self.device_counter:06d}"
        device_type = random.choice([
            "Mobile",
            "Desktop",
            "Tablet"
            ])

        operating_systems = {
            "Mobile": ["Android", "iOS"],
            "Desktop": ["Windows", "macOS", "Linux"],
            "Tablet": ["Android", "iPadOS"]
            }
        operating_system = random.choice(
            operating_systems[device_type]
            )

        browsers = {
            "Android": ["Chrome", "Firefox"],
            "iOS": ["Safari", "Chrome"],
            "Windows": ["Chrome", "Edge", "Firefox"],
            "macOS": ["Safari", "Chrome", "Firefox"],
            "Linux": ["Chrome", "Firefox"],
            "iPadOS": ["Safari", "Chrome"]
            }
        browser = random.choice(browsers[operating_system])
        trusted_device = random.choice([True, False])
        device_fingerprint = self.fake.sha256()
        last_seen = self.fake.date_time_between_dates(
            datetime_start=(
                self.reference_datetime - relativedelta(years=1)
                ),
            datetime_end=self.reference_datetime
            )

        device = Device(
            device_id=device_id,
            device_type=device_type,
            browser=browser,
            OS=operating_system,
            trusted_device=trusted_device,
            device_fingerprint=device_fingerprint,
            last_seen=last_seen
            )
        self.device_counter += 1
        return device
