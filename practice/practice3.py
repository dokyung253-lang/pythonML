# PythonML Practice3: K-최근접 이웃 회귀와 과소적합 해결 
# https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 확인
# 파일명: ./Iris.csv
import pandas as pd
df = pd.read_csv( './day01/Iris.csv')

# [단계 2] 특정 품종 추출 (데이터 필터링)
# 전체 데이터 중 Species가 'Iris-setosa'인 데이터만 추출하세요.
target_iris = df[df['Species'].isin(['Iris-setosa'])]
# print( target_iris ) # 49개

# [단계 3] 특성 데이터 및 정답 데이터 추출
# 추출한 붓꽃의 꽃받침 길이('SepalLengthCm')를 특성 데이터(iris_length)로, 
# 꽃받침 너비('SepalWidthCm')를 정답 데이터(iris_width)로 추출하세요.
iris_length = target_iris['SepalLengthCm'].values
iris_width = target_iris['SepalWidthCm'].values
# print( iris_length, iris_width )

# [단계 4] 훈련용 / 테스트용 데이터 분리
# train_test_split() 함수를 사용하여 학습용 데이터와 테스트용 데이터를 분리하세요.
# test_size는 0.3, random_state는 42로 설정하세요.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(iris_length, iris_width, test_size= 0.3, random_state =42)

# [단계 5] 데이터 차원 변환 (Reshape)
# 사이킷런 모델 학습을 위해 1차원 배열인 train_input과 test_input을  [개수, 1] 형태의 2차원 배열로 변환하세요.
train_input = train_input.reshape( -1, 1 )
test_input = test_input.reshape( -1, 1 )

# [단계 6] KNeighborsRegressor 모델 생성 및 학습
# KNeighborsRegressor 객체를 생성하고 훈련용 데이터로 모델을 학습하세요.
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit( train_input, train_target ) # 모델 학습

# [단계 7] 초기 모델 평가 및 결정계수(R^2) 확인
# 훈련 세트와 테스트 세트의 정확도(score)를 각각 출력하기
print( knr.score( test_input, test_target ) ) # 0.60355587808418

# [단계 8] 이웃 개수 변경에 따른 회귀선 시각화
# 이웃의 개수(n_neighbors)가 1, 3, 5, 10으로 변화할 때, 꽃받침 길이 4.0부터 6.0까지의 구간에 대한 모델의 예측 회귀선을 각각 시각화하세요.
import numpy as np
import matplotlib.pyplot as plt
knr = KNeighborsRegressor() 
x = np.arange( 4, 6 ).reshape( -1, 1 )
print( x )

for k in [1, 3, 5, 10] :
    knr.n_neighbors = k
    knr.fit( train_input, train_target )
    print( knr.score(test_input, test_target) )    
    pred = knr.predict(x)
    # 시각화
    plt.scatter( train_input, train_target )
    plt.plot( x, pred )
    plt.title( f'k = {k}') 
    plt.show()
# [단계 9] 단계8 에서 가장 적합한 이웃의 개수( 1, 3, 5, 10 중에)ㄴ를 주석에 작성 하시오. 
# k가 5인 회귀선이 가장 적합. score가 0.60355587808418로 가장 크기 때문.