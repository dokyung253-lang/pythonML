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

# [*] 경사하강법
# fit() 모델학습에서는 정답(target)도 같이 학습중이다. 예측(y)값과 실제 정답간의 오차 ( 기울기/가중치 조절 오차 줄이기 ) 측정
# 예] 산꼭대기에서 내려가는 방법 중에 가장 최적의 경로로 내려오는 방법 = 경사하강법( 수많은 경우의 수 계산하여 판단 )
# (1) 전통적인 경사하강법 : 정확도 좋지만 학습속도 느리다. 
# VS (2) 확률 경사 하강법(SGD) : 정확도 낮지만 학습속도 빠르다 # 미니배치
#       모든 데이터 사용하지 않고, 미니배치(대표데이터)로 오차 찾기

# [*] 로그 로스 / 손실함수, 손실( 예측과 정답의 전체 차이 )
# 로그 로스 함수는 0과 1의 확률값이 아닌 *오차값*을 측정

# [*] 에포크 : 학습 횟수

# [2] SGDClassifier, 분류모델
from sklearn.linear_model import SGDClassifier
# loss= 'log_loss' : 로스함수
# random_state : SGD가 전체 데이터 학습이 아닌 일부자료(미니배치)를 가지고 학습하는데 사용되는 분리기준(난수값)
# max_iter : (반복)계산횟수 # 미니배치이므로 전체 데이터셋을 '10'이면 10 반복학습하여 모델 성공 향상/ 최적의 정확도에서 멈춤(에포크)
# tol = None : 최적의 정확도를 찾아도, 계속 반복학습 설정
sc = SGDClassifier( loss= 'log_loss', random_state= 42 , max_iter= 10, tol=None )    # 모델 객체 생성
sc.fit( train_scaled, train_target )  # 10번 학습                       # 모델학습
print( sc.score( test_scaled, test_target) ) 
print( sc.predict( test_scaled[:3] ) )                      # ['Perch' 'Smelt' 'Pike']

# 점진적 학습( 중간학습 가능 )
sc.partial_fit( train_scaled, train_target ) # 1번 학습 , 총 11번
print( sc.score( test_scaled, test_target ) )

# [4] 최적의 에포크(반복학습횟수) 찾기
sc = SGDClassifier( loss='log_loss', random_state= 42 ) # max_iter 생략 시 1번 학습

#
train_score = [ ] # 학습용 정확도
test_score = [ ] # 테스트용 정확도

# 정답지의 중복제거한 고유 정답만 추출
import numpy as np
classes = np.unique( train_target ) 

for i in range( 0, 150 ) : # 300번 반복
    sc.partial_fit( train_scaled, train_target , classes = classes ) # 1학습

    train_score.append( sc.score( train_scaled, train_target ) )
    test_score.append( sc.score( test_scaled, test_target) ) 

# 정확도 시각화 # 과대적합: 둘이 너무 떨어진 시점  # 과소적합: 두 차트간에 너무 붙어있는 시점 
# 최적의 에포크(반복횟수)는 학습용과 테스트용이 고르게 오르는 시점
import matplotlib.pyplot as plt
plt.plot( train_score )              
plt.plot( test_score )
plt.show()         

# [5] hinge / SVM( 서포트 벡터 머신 )
# alpha= 0.0001 : 기본값(0.0001) , 힌지 함수는 경계면(애매/아슬)에 있는 자료들을 찾는 기준점/크기
sc = SGDClassifier( loss = 'hinge', max_iter = 100, random_state= 42, alpha=0.0001 )
sc.fit( train_scaled, train_target )

print( sc.score( train_scaled, train_target ) ) #0.9159663865546218
print( sc.score( test_scaled, test_target ) ) # 0.925

# 로지스틱 회귀 : 확률 이용한 분류('log_loss')
# SGD(확률 경사하강법 / 미니배치 ) : loss='log_loss' vs loss='hinge'
# 경사하강법 : 손실( 예측과 정답 오차 ) 0에 가깝게 처리하기 위한 반복 계산 ( * 딥러닝 * )
# loss = 'log_loss'
    # 도미일 확률이 51%일 때 기울기 , y절편을 끊임없이 조정하여 확률 100% 만드는 방법( 경사 하강법 )
# loss = 'hinge'
    # *전체데이터가 아닌* 도미 확률이 50%, 0인 지점 근처인 애매한 (경계선) 자료만 가지고 확률(오차) 조정하는 방법( 경사 하강법 )