class ErrorCode():
    '''
    에러 코드 관리
    '''
    NOT_FOUND = '404'
    BAD_REQUEST = '400'

class ErrorMsg():
    '''
    에러 메시지 관리
    '''
    WRONG_INPUT = '입력이 잘못되었습니다.'

class CustomError(Exception):
    '''
    커스텀 에러 코드 클래스
    '''
    def __init__(self, code, msg):
        self.code = code # from ErrorCode
        self.msg = msg # from ErrorMsg

    def __str__(self):
        '''
        에러 메시지 출력부
        '''
        return f'[ {self.code} ] {self.msg}'


# 예시
tmp = None
try:
    if tmp is None:
        raise CustomError(ErrorCode.BAD_REQUEST, ErrorMsg.WRONG_INPUT)
except CustomError as e:
    print(e)