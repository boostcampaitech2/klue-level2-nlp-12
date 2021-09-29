import pytz
from datetime import datetime


def korea_now():
    """
    현재시간을 알려주는 함수
    혹시 파일생성시 파일명 안 겹치게 저장하려고 만들었습니다
    """
    now = datetime.now()
    korea = pytz.timezone("Asia/Seoul")
    korea_dt = korea.normalize(now.astimezone(korea))
    return str(korea_dt).split(".")[0]
