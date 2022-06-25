import pytz
from datetime import datetime

IST = pytz.timezone('Asia/Kolkata')


def get_datetime_from_millis(ms: int):
    return datetime.fromtimestamp(ms // 1000, tz=IST)


def get_millisec_from_datetime(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)
