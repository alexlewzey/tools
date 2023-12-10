from datetime import datetime

import requests


def is_weekday():
    today = datetime.now()
    return 0 <= today.weekday() < 5


def get_holiday() -> None:
    r = requests.get("www.somesite.com")
    if r.status_code == 200:
        return r.json()
