
# [1] 여러가지 특성에 따른 어종 분류 모델
import pandas as pd
df = pd.read_csv('./day01/Fish.csv')
# 어종 7개, Species
fish_target = df['Species']
# 특성 6개, Weight,Length1,Length2,Length3,Height,Width
fish_input = df[ [ 'Weight', 'Length1' , 'Length2', 'Length3', 'Height', 'Width'] ]
# 훈련 / 테스트 분리 
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( fish_input, fish_target, test_size= 0.25, random_state= 42 )
# 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit( train_input )
train_scaled = ss.transform( train_input )
test_scaled = ss.transform( test_input )

# 로지스틱 회귀 = 이진분류 = 시그모이드 함수(공식)
# 선형 방정식의 출력값을 0과 1(확률/분류) 사이의 값으로 변환해주는 공식/함수
# 예시] 암 환자의 확률 / 스팸 메일 분류 등 (임계점(중앙값)을 기준으로 모 아니면 도)
# 즉] 컴퓨터는 수치상의 150 또는 -82.3 (수치)값으로 확률 어렵다, 확률이란? 항상 0(0%)에서 1(100%) 사이여야 하기 때문
import numpy as np
import matplotlib.pyplot as plt
z = np.arange( -5, 5, 0.1 ) # -5부터 5까지 0.1씩 증가하는 리스트 
phi = 1 /( 1 + np.exp( -z ) ) # 시그모이드 공식
plt.plot( z, phi )
# plt.show()


# [2] 이진 분류 # 로지스틱 회귀 모델 # 시그모이드 함수
# 이진분류는 0 또는 1 분류하는 방법
indexs = ( train_target == 'Bream' ) | ( train_target == 'Smelt' ) # 도미와 빙어만 추출
print ( indexs )
train_bream_smelt = train_scaled[ indexs ]
target_bream_smelt = train_target[ indexs ]
print( target_bream_smelt )
# 이진분류 모델 구현
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit( train_bream_smelt , target_bream_smelt ) # 도미와 빙어만 학습
# 이진분류 모델 예측 
print( lr.predict( train_bream_smelt [: 3] ) ) # 위에서 3개만 예측 # ['Bream' 'Smelt' 'Bream']
print( lr.predict_proba( train_bream_smelt[: 3 ] ) ) # 3개만 예측 확률 [[ 도미확률, 빙어확률]] # 총합의 확률은 1(100%)
# 임계값은 0.5 기준으로 0.5% 이상이면 도미 예측하고, 0.5 미만이면 빙어 예측한다
# [[0.99793611 0.00206389]
#  [0.02391315 0.97608685]
#  [0.99575505 0.00424495]]

# [3] 다중분류 # 로지스틱 회귀 모델 # 소프트맥스
# 하이퍼파라미터
# C: 규제를 완화하여 릿지/라쏘 모델처럼 정확도 설정 가능하다. # 모델의 성능 향상하기 위해서 가중치 값들을 자동 조정
# max_iter : 다중분류 계산 횟수 # (생략 시) 기본값 100으로 최적의 정확도를 찾을 때까지 계산 반복횟수 조정 # 넉넉하게 넣기( 1000 )
lr = LogisticRegression( C = 20 , max_iter= 1000 )
lr.fit( train_scaled, train_target ) # 모든 어종 학습
# 모델 예측
print( lr. predict( test_scaled[:3] ) ) # 3개만 예측 # ['Perch' 'Smelt' 'Pike']
print( lr.predict_proba( test_scaled [:3 ] ) ) # 3개만 예측 # 분류개수 만큼의 확률
# 모델 평가
print( lr.score( test_scaled, test_target ) ) # 0.85
# 소프트 맥스
from scipy.special import softmax # 소프트맥스 : 
decision = lr.decision_function( test_scaled [ : 3 ] )
print( decision )
print( softmax( decision ) )                            # 소프트맥스 라는 함수로 결과값을 확인했을 때 predict 동일하게 출력된다.
print( np.round( softmax( decision), decimals= 3 ) )    # np.round( 값, decimals = 소수점 )
#[[0.    0.001 0.041 0.    0.007 0.    0.   ]
#  [0.    0.001 0.031 0.    0.005 0.688 0.   ]
#  [0.    0.    0.007 0.212 0.002 0.006 0.   ]]
# 다중분류의 확률 검증할 때는 .classes_ 종속변수들의 순서 확인
print( lr.classes_ ) # 종속변수들 출력 : ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']