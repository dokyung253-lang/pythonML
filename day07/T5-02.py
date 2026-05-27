# 한국어로 된 샘플데이터
text_data = [
    "사과 바나나 과일 식사",
    "포도 멜론 단맛 과일",
    "삼성 스마트폰 아이폰 출시 갤럭시 전자기기", 
    "컴퓨터 모니터 마우스 키보드 전자기기 제품", 
    "과일 딸기 주스 음료 디저트",
    "전자기기 노트북 인공지능 그래픽카드 컴퓨터 성능"
]
import pandas as pd
df = pd.DataFrame({'자연어_문장': text_data})

# [*] 한국어 대한 군집은 범주형이라서 힘들다. kmodes 모델 사용하거나 *DL모델* 사용.
# [1] 범주형 -> 수치형(벡터화) 변경, 텍스트(자연어) 수치형 벡터로 변환
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer( ) # 객체 생성
vector = tv.fit_transform( df['자연어_문장']) # 객체 벡터화

# [2] 벡터화 스케일링 , 각 문장별 벡터 최대 길이를 1로 고정
from sklearn.preprocessing import normalize
vector = normalize( vector )
print( vector )

# [3] k-mean 분집 모델, 키워드 빈도군집 # 한계점 : 자료부재, TfidfVectorizer(문맥/의미 이해안함), 형태소분석불가 ==> LLM(BERT)딥러닝 모델 필요
from sklearn.cluster import KMeans
km = KMeans( n_clusters= 2,random_state= 42)
km.fit( vector ) # 벡터화된 문자열을 학습
df['예측군집'] = km.labels_

print( df ) # 0: 전자기기 , 1: 과일

# 벡터 구성 예시
# [1] 2개의 문장 존재,                                "사과 바나나 과일 식사", "포도 멜론 단맛 과일",
# [2] 전체 문장 나열하여 총 7개 키워드/단어 추출,       사과,바나나,과일,식사,포도,멜론,단맛
# [3] 첫번째 문장은 7가지 키워드[2]에 존재하는지 검토 , 존재 : 1, 존재X : 0
#       사과,바나나,과일,식사,포도,멜론,단맛 --> 1, 1, 1, 1, 0, 0, 0
# [4] 두번째 문장은 7가지 키워드에 존재하는지 검토2, 
#       사과,바나나,과일,식사,포도,멜론,단맛 --> 0, 0, 1, 0, 1, 1, 1
# 스케일링
# [5] 1, 1, 1, 1, 0, 0, 0, 제곱의 합으로 총합을 1으로 만든다.
