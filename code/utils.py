import pytz
from datetime import datetime


def korea_now():
    now = datetime.now()
    korea = pytz.timezone('Asia/Seoul')
    korea_dt = korea.normalize(now.astimezone(korea))
    return str(korea_dt).split('.')[0]
