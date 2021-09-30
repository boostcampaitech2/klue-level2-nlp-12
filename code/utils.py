import pytz
from datetime import datetime

from sklearn.utils.validation import column_or_1d
from load_data import *

TRAIN_DATA_PATH = '/opt/ml/dataset/train/train.csv'

def korea_now():
    """
    현재시간을 알려주는 함수
    혹시 파일생성시 파일명 안 겹치게 저장하려고 만들었습니다
    """
    now = datetime.now()
    korea = pytz.timezone("Asia/Seoul")
    korea_dt = korea.normalize(now.astimezone(korea))
    return str(korea_dt).split(".")[0]

def switch_sub_obj():
    config = {
        "change_entity": {"subject_entity": "object_entity", "object_entity": "subject_entity"},

        "possible_label_list": ['no_relation', 'org:members', 'org:alternate_names',
                              'per:children', 'per:alternate_names', 'per:other_family', 
                              'per:colleagues', 'per:siblings', 'per:spouse',
                              'org:member_of', 'per:parents'],
        
        "opposite_label_list": {"org:member_of": "org:members", 
                          "org:members": "org:member_of", 
                          "per:parents": "per:children", 
                          "per:children": "per:parents" },
        
        "cols": ['id', 'sentence', 'subject_entity', 'object_entity', 'label']
    }
    # 1. 훈련 데이터 불러와서 subject_entity 와 object_entitiy의 column name 만 바꾼다.
    data = load_data(TRAIN_DATA_PATH).rename(columns=config['change_entity'])
    # 2. subject_entity와 object_entity를 바꿀 수 있는 라벨만 남긴다.
    data = data[data['label'].isin(config['possible_label_list'])]
    # 3. column name을 정렬해준다. (1번에서 column name만 obj, sub 순으로 바꿔놨기 때문에 다시 sub, obj 순으로 정리하면 열 전체가 바뀐다.)
    data = data[config['cols']]
    # 4. 서로 반대되는 뜻을 가진 label을 바꿔준다.
    data = data.replace({'label':config['opposite_label_list']})

    return data
