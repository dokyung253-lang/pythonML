# 용어정리
# 모델 : 데이터(자료)를 학습하는 프로그램
    # K-NN모델 : 가까운 이웃 기준의 예측
        # KNeighborsClassifier : k최근접이웃 분류 (타겟이 범주형, 예: 물고기 종류)
        # KNeighborsRegressor  : k최근접이웃 회귀 (타겟이 연속형, 예: 물고기 무게 예측)
            # 하이퍼파라미터(k) : 이웃개수(k) 직접 설정하여 최적의 K찾기 (모델 학습 전에 사람이 직접 정하는 값)
            # 학습 특성의 형태는 2차원 배열만 가능 
                # 변환예시 : T2-01( zip ), T1-02( column_stack ), T2-02( reshape ) 
# ---------------------------------------------------------------------------------
# 학습 : 데이터(자료)의 규칙 찾는 과정
# 예측 : 학습된 모델로 새로운 데이터(결과) 추론하는 과정
# 특성 : 학습에 입력 정보           # 물고기의 '길이', '무게'
# 타겟 : 학습의 정답 정보           # 물고기의 '종류'
# 표준화(스케일링) : 0 ~ 1 사이로 크기 맞춤
    # StandardScaler() #.transform()
# 과소적합    : 너무 데이터가 단순한 경우      # 이웃이 너무 많아서 기준 애매
# 과대적합    : 한 쪽 특성에 너무 암기된 경우  # 이웃이 너무 많아서 특정 이웃 학습
# ---------------------------------- day 1-2 --------------------------------------

# [1] 숭어의 '길이' , '무게' : '길이'(특성)에 따른 '무게'(타겟) 예측
import pandas as pd
df = pd.read_csv( './day01/Fish.csv')
fish_data = df[ df[ 'Species' ].isin(['Perch'])]
perch_length = fish_data['Length2'].values
perch_weight = fish_data['Weight'].values
print( perch_length, perch_weight ) # 확인

# [2] 훈련셋과 테스트셋 분리
from sklearn.model_selection import train_test_split
# train_test_split( 특성, 타겟, test_size= 테스트비율, random_state= 분리기준난수 )
train_input , test_input , train_target, test_target = train_test_split( perch_length , perch_weight, test_size= 0.2, random_state= 42 )

# [3] 학습하기 전에 사이킷런 모델들은 2차원 배열만 가능하다   # [ 1, 2, 3 ] ---> [ [1] [2] [3] ] 
train_input = train_input.reshape( -1, 1 ) # reshape( 행개수, 열개수 )   # -1 행은 자동으로 설정, 1열은 1개 설정
test_input = test_input.reshape( -1, 1 )

# [4] k-최근접이웃 회귀모델 훈련
from sklearn.neighbors import KNeighborsRegressor # 회귀
knr = KNeighborsRegressor() # 모델객체 생성
knr.fit( train_input , train_target ) # 모델 학습
print( knr.score( test_input, test_target ) ) # 모델 평가  # 0.9932626838364674

# [5] 임의의 값으로 예측하기
print( knr.predict( [[50]]) ) # [1010.] # 임의의 물고기 길이 50일 때 무게 예측
print( knr.predict([[100]]) ) # [1010.] # 임의의 물고기 길이 100일 때 무게 예측

# 문제점 : k-최근접이웃의 문제점은 단순한 주변 이웃의 평균으로 예측하기 때문에 최댓값을 벗어나면 항상 동일한 값으로 예측한다.
# 즉] 소규모 또는 간단한 예측 프로그램에서만 사용된다.

# [1] (단순)선형회귀 모델 # 1차 방정식
from sklearn.linear_model import LinearRegression # 선형회귀 모델
lr = LinearRegression() # 모델 객체 생성
lr.fit( train_input, train_target ) # 모델 학습
print( lr.score( test_input, test_target) ) # 모델 평가 
print( lr.predict( [[50]] )) # 모델 예측 # [1238.3175398]
print( lr.predict( [[100]] )) # [3191.00026354]
# 직선 공식(1차 방정식) : y(예측값) = w(가중치)x(특성) + b(절편)
# 즉] (물고기)무게 = 가중치 * (물고기)길이 + 절편
print( lr.coef_ )       # 기울기값 반환  [39.05365447]  # 직선의 기울기(특성의 가중치)
print( lr.intercept_ )  # y절편 반환 -714.3651839448922 # 편향 # x(물고기길이)가 0일 때 y의 값
# x와 y가 직선관계이며, 실자료들은 물고기 길이가 1씩 증가할 때 무조건 무게 비례증가는 아님. (상관관계 부정확)
# 즉] 초반에는 길이에 따라 무게가 3배 증가하다가 중후반에는 무게가 2배 혹은 1배 증가할 수 있다. # 사람 : 어릴수록 키 증가 폭 큼, 어느 순간 증가 폭 고정

# [2] 시각화
import matplotlib.pyplot as plt
plt.scatter( train_input, train_target ) # x축: 길이, y축 : 무게
plt.scatter( 50, 1238 ) # 예측된 결과 # 길이가 50일 때 무게는 1238일 것이다.
plt.scatter( 100, 3191 )

plt.plot( [15, 100], lr.predict([[15], [100]]) ) # 회귀선 그리기 # 15 = (물고기 길이의)시작점, 100 길이의 끝점
plt.show()

# [3] ( 다항 : 여러개 항 )선형회귀 모델 # 2차 방정식
#
