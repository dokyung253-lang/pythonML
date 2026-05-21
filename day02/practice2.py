# PythonML Practice2: 훈련용/테스트용 분리 + 스케일링
# https://www.kaggle.com/code/anshigupta01/iris-flower-classification

# [단계 1] 데이터 로드 및 확인
# 파일명: ./Iris.csv
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv( './day01/Iris.csv')
print( df.head() )
df.info()

# [단계 2] 특정 품종 추출
target_iris = df[df['Species'].isin(["Iris-setosa", "Iris-versicolor"])]

# [단계 3] 특성 데이터와 Species 함께 유지
target_iris = target_iris[["PetalLengthCm", "PetalWidthCm", "Species"]]

# [단계 4] 학습 데이터 구성
length_Width = target_iris[["PetalLengthCm", "PetalWidthCm"]].values

# [단계 5] 정답 데이터(Target) 생성
iris_data = []
for species in target_iris['Species']:
    if species == "Iris-setosa":
        iris_data.append(1)
    else:
        iris_data.append(0)
        
# [단계 6] 훈련용 / 테스트용 데이터 분리
# train_test_split() 함수를 사용하여
# 학습용 데이터와 테스트용 데이터를 분리하세요.
# test_size 옵션을 설정하세요.
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(length_Width, iris_data, test_size=0.3, random_state=42)

# [단계 7] KNeighborsClassifier 모델 생성 및 학습
# KNeighborsClassifier 객체를 생성하고
# 훈련용 데이터로 모델을 학습하세요.
from sklearn.neighbors import KNeighborsClassifier
knr = KNeighborsClassifier()
print (knr.fit( train_input , train_target )) # 길이에 따른 무게 학습
# [단계 8] 모델 평가
# 테스트용 데이터를 사용하여
# 모델의 정확도(score)를 출력하세요.
print( '테스트정확도', knr.score( test_input, test_target ) )

# [단계 9] 새로운 데이터 예측
# [꽃잎 길이 2.0, 꽃잎 너비 0.5] 데이터를
# 어떤 품종으로 예측하는지 확인하세요.
import numpy as np
x = np.array( [[2.0, 0.5]])
pred = knr.predict(x)
print( '품종 예측', knr.predict(x) ) 

# [단계 10] 데이터 시각화
# 산점도(Scatter plot)를 사용하여
# 훈련용 데이터와 예측 데이터를 시각화하세요.
plt.scatter( train_input[:,0], train_input[:,1] )
plt.scatter(x[:,0], x[:,1], c=pred)          # 새로운 데이터 예측 결과
plt.xlabel('PetalLengthCm')
plt.ylabel('PetalWidthCm')
plt.show()

# [단계 11] 최근접 이웃 확인
# kneighbors() 함수를 사용하여
# 예측에 사용된 최근접 이웃 데이터를 확인하고 시각화하세요.
new_point = [2.0, 0.5]

# kneighbors로 이웃(거리, 인덱스) 확인
neighbors = knr.kneighbors(x, n_neighbors=5)
print("이웃 인덱스:\n", neighbors[1])
print("이웃 거리:\n", neighbors[0])

# 시각화 (스케일링 전)
plt.figure()
plt.scatter(train_input[:,0], train_input[:,1], c=train_target)
plt.scatter(new_point[0], new_point[1], marker='^', s=100, c='k')
# neighbors[1]은 shape (1, n_neighbors) 이므로 [0]으로 인덱싱
plt.scatter(train_input[ neighbors[1][0], 0 ], train_input[ neighbors[1][0], 1 ])
plt.xlabel("Length2")
plt.ylabel("Weight")
plt.show()

# [단계 12] 스케일링(StandardScaler) 적용
# StandardScaler 객체를 생성하세요.
# 훈련용 데이터를 기준으로 fit() 하세요.
# transform()을 사용하여 훈련용 데이터를 스케일링하세요.
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(train_input)
print( scaler.mean_)
print(scaler.scale_)
train_scaled = scaler.transform(train_input)
print("스케일링된 훈련 데이터(일부):", train_scaled[:5])

# [단계 13] 스케일링 이후 재학습
# 스케일링된 훈련용 데이터로
# 모델을 다시 학습하세요.
knr.fit(train_scaled, train_target)  # 표준화된 자료로 재학습

# [단계 14] 새로운 데이터 스케일링 후 예측
# [2.0, 0.5] 데이터도 동일하게 스케일링하여
# 품종을 다시 예측하세요.
new_scaled = scaler.transform(x)  # [2.0, 0.5] 동일하게 스케일링
pred = knr.predict(new_scaled)

# [단계 15] 스케일링 이후 최근접 이웃 시각화
# 스케일링된 데이터 기준으로
# 최근접 이웃들을 다시 시각화하세요.
dist, indexs = knr.kneighbors(new_scaled)

plt.figure()
plt.scatter(train_scaled[:,0], train_scaled[:,1], c=train_target)
plt.scatter(new_scaled[0,0], new_scaled[0,1], marker='^', s=100, c='k')
plt.scatter(train_scaled[ indexs[0], 0 ], train_scaled[ indexs[0], 1 ]) 
plt.xlabel("Length2 (scaled)")
plt.ylabel("Weight (scaled)")
plt.show()
