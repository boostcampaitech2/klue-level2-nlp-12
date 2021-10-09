class ErrorCode():
    '''
    A constant class for managing error codes
    '''
    NOT_FOUND = '404'
    BAD_REQUEST = '400'

class ErrorMsg():
    '''
    A constant class for managing error messages
    '''
    WRONG_INPUT = '입력값 형태가 올바르지 않습니다.'
    ALREADY_EXIST = '이미 존재하는 명칭입니다.'

class CustomError(Exception):
    '''
    A error class for customized error code and messages

    Args:
        code (str): A customized error code
        msg (str): A customized error messages
    '''
    def __init__(self, code, msg):
        self.code = code    # from ErrorCode class
        self.msg = msg      # from ErrorMsg class

    def __str__(self):
        return f'[ {self.code} ] {self.msg}'

# example
tmp = None
try:
    if tmp is None:
        raise CustomError(ErrorCode.BAD_REQUEST, ErrorMsg.WRONG_INPUT)
except CustomError as e:
    print(e)