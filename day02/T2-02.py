# [1]
import pandas as pd
df=pd.read_csv("./day02/Fish.csv")
df.info()

# [2] 필요한 어종 추출: 조건식 대신에 .isin() 특정 값만 추출, isna() 결측치만추출
target_fish=df[df["Species"].isin(["Bream", "Smelt"])]
print(target_fish)

# [3] 필요한 특성 추출 : Length2, Weight
# 넘파이 # np.column_stack(( 리스트1 ,리스트 2 )) : 두 리스트 간에 동일한 요소로 2차원리스트 구성
# T2-01.py [6] zip 함수 대신에 2차원 리스트 구성 방법
import numpy as np
fish_data = np.column_stack(( target_fish['Length2'], target_fish['Weight'] ))
print( fish_data ) # [[ 길이 무게 ][ 길이 무게 ]]

# [4] 모델 학습 하기 위한 정답지 만들기, 도미 35마리, 빙어 14마리
# np.ones( 개수 ) : 개수만큼 1 채워진 리스트 반환, np.zeros( 개수 ) : 개수만큼 0 채워진 리스트 반환
# concatenate( 리스트, 리스트 ) : 두 리스트 연결
fish_target = np.concatenate( ( np.ones( 35 ) , np.zeros( 14 ) ) )
print( fish_target )
# [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 
# 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# [5] 학습 모델 만들기 전 학습용,테스트용 분리 # 방대한 자료(억단위 이상) 학습용과 테스트용 구분하여 모델 구성하며 테스트 한다.
from sklearn.model_selection import train_test_split
# 학습용자료, 테스트용자료, 학습용정답지, 테스트용정답지 = train_test_split(학습자료, 정답지, test_size = 테스트자료비율 ) 
# 4개의 반환타입을 갖는다.
train_input, test_input, train_target , test_target = train_test_split( fish_data , fish_target, test_size = 0.3 ) # 학습용7 : 테스트용3 비율로 분할
print( train_input.shape ) # (34, 2) # 49개 중 학습용 7에 해당하는 개수가 34개
print( test_input.shape ) # (15, 2) # 49개 중 테스트용 3에 해당하는 개수가 15개

# [6] 학습모델 : k-최근접 이웃 분류기 모델
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier() # 모델 객체 생성 # new 없음
kn.fit( train_input , train_target ) # 모델 (지도) 학습
print( kn.score(test_input, test_target) ) # 모델 평가( 1:100% )

# [7] 임의의 값으로 학습모델 예측하기
# 길이: 25, 무게 : 150의 물고기가 도미[1]인지?빙어[0]인지? 예측하기
print( kn.predict ([[25, 150]]) ) # 모델 예측 # [0.], 빙어 # 잘못된 예측

# [8] 예측값 시각화
import matplotlib.pyplot as plt
# train_input[:,0] : [행슬라이싱, 열슬라이싱], 모든 행의 0번째 열만 추출 # 즉] 길이만 추출
# train_input[:,1] : [행슬라이싱, 열슬라이싱 ], 모든 행의 1번쨰 열만 추출 # 즉] 무게만 추출
plt.scatter( train_input[:,0], train_input[:,1]) # 학습용
plt.scatter( 25, 150 ) # 예측값
plt.show()

# [9] 예측하기 위한 이웃들 확인,  .kn.kneighbors( [ 예측값 ] ), 예측에 사용된 이웃들을 반환
dist, indexs = kn.kneighbors( [[25, 150]] )
plt.scatter( train_input[:,0], train_input[:,1])                # 학습용
plt.scatter( 25, 100 )                                          # 예측값     
plt.scatter( train_input[indexs, 0], train_input[indexs, 1] )   # 예측에 사용된 이웃자료 # 문제 발견
plt.show()