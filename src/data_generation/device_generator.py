from dataclasses import dataclass
from datetime import datetime

from faker import Faker
import random


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
    def __init__(self):
        self.fake = Faker("en_GB")
        self.device_counter = 1

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
        last_seen = self.fake.date_time_between(
            start_date="-1y",
            end_date="now"
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
