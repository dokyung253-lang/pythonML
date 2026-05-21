# PythonML Practice4: 다항 회귀를 이용한 비선형 데이터 예측
# 데이터 출처: https://www.kaggle.com/code/anshigupta01/iris-flower-classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv( './day01/Iris.csv')

# [단계 1] 데이터 로드 및 특정 품종 추출
target_iris = df[df['Species'].isin(['Iris-setosa'])]
iris_length = target_iris['SepalLengthCm'].values
iris_width = target_iris['SepalWidthCm'].values

# [단계 2] 훈련용 / 테스트용 데이터 분리
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(iris_length, iris_width, test_size=0.3, random_state= 42 )

# [단계 3] 데이터 차원 변환 (Reshape)
train_input = train_input.reshape( -1, 1 )
test_input = test_input.reshape( -1, 1 )


# [단계 4] 단순 선형 회귀(Linear Regression) 모델 학습 및 평가
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit( train_input , train_target ) 
print( lr.score( train_input, train_target ) ) # 0.5395298831723283
print( lr.score( test_input, test_target) ) # 0.5423687498422736
print( lr.coef_ ) # [0.76182638]
print( lr.intercept_ ) # -0.3876310655937556


# [단계 5] 다항 회귀를 위한 데이터 전처리 (특성 추가)
import numpy as np
train_poly = np.column_stack(( train_input **2 , train_input))
test_poly = np.column_stack(( test_input**2, test_input ))

# [단계 6] 다항 회귀 모델 학습 및 결정계수(R^2) 확인
lr = LinearRegression()
lr.fit(train_poly, train_target)
print( lr.score( train_poly, train_target ) ) # 0.5444662337785278
print( lr.score( test_poly, test_target ) ) # 0.5413125141187329

# [단계 7] 다항 회귀 모델을 통한 임의의 값 예측
point = np.arange( 4, 6, 0.1 )
point_poly = np.column_stack( ( point**2, point ) )
print( lr.predict(point_poly) )
#[[16  4]
# [25  5]]

# [단계 8] 다항 회귀 곡선 시각화
# 1) 4.0부터 6.0까지 0.1 간격으로 증가하는 배열(point)을 생성하고, 이를 다항 특성(point_poly)으로 변환하세요.
# 2) 산점도(scatter)를 이용해 원래의 train_input과 test_input 데이터를 시각화하세요.
# 3) 앞서 생성한 point와 다항 회귀 모델의 예측값(predict)을 사용하여 2차 방정식 형태의 곡선(회귀선)을 그리세요.
import matplotlib.pyplot as plt
plt.scatter( train_input, train_target )
plt.scatter( test_input, test_target )
dd = plt.plot( point, lr.predict( point_poly ) )
plt.show() 
# [단계 9] 단순 선형 회귀(1차 방정식)와 비교했을 때, 특성을 제곱한 다항 회귀(2차 방정식)가 가지는 수학적/표현적 장점을 주석으로 서술하시오.
# 기울기를 결정하는 'SepalLengthCm' 가 커질 수록 'SepalWidthCm'가 배로 늘어남