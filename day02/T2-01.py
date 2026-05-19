# [1] Fist csv 가져오기
import pandas as pd
df=pd.read_csv("./day01/Fish.csv")

# [2] Perch(농어)만 추출
target_fish = df[ df['Species'].isin(['Perch'])]
target_fish.info() # 56마리

# 농어의 길이/무게 추출
perch_length = target_fish['Length2'].values
perch_weight = target_fish['Weight'].values
print( perch_length, perch_weight ) # 길이, 무게

# '농어' 길이에 따른 무게 예측
import matplotlib.pyplot as plt
plt.scatter( perch_length, perch_weight)
plt.show()

# [3] 학습모델 만들기, (1) 준비 : 학습용과 테스트용 분리. 왜? 모델평가에 사용됨
from sklearn.model_selection import train_test_split
# random_state : 난수값에 따라 분리. 고정값 넣어주면 항상 동일한 분리값 넣을 수 있다. 0~32억 사이
# train_test_split( 학습자료, 정답자료, test_size= 분리비율, random_state= 분리기준난수 ) 
train_input , test_input, train_target, test_target = train_test_split( perch_length, perch_weight,  test_size= 0.3, random_state= 42 ) # 42는 관례적 수치

# (2) 준비 : 자료형식(모양)구성, 대부분 2차원 사용한다.
import numpy as np
array = np.array([1, 2, 3, 4])
print( array.shape ) # shape 배열의 모양 반환 : (행, 열), (4,)
array2 = np.array( [ [ 1, 2,], [3, 4], [5, 6] ] )
print( array2.shape ) # (3, 2)

print( train_input.shape) # (39, ) 1차원배열 --> 사이킷런 모델들은 1차원배열 학습이 불가능하다.
print( train_input )      # 1차원으로 구성된 '농어'길이
# [17.4 36.  25.  40.  39. ~~ ] 
# T2-01( zip 활용 ), T1-02( column_stack 활용 ), T2-02( reshape ) : 1차원 -> 2차원

# [4] .reshape( 행개수, 열개수 ) : 행개수에는 -1 넣어서 자동 뜻(자료개수만 자동), 열개수는 1개 , 
train_input = train_input.reshape( -1 , 1 )
print( train_input ) #[[17.4] [36. ] [25. ] [40. ] [39. ] ~ ]
print( train_input.shape ) #(39, 1) 2차원 배열 
# train_target = train_target.reshape(-1, 1)        
test_input = test_input.reshape( -1, 1 )            # 테스트 학습 2차원
# np이용해서 농어의 길이 1차원에서 2차원으로 만듦. 왜? 사이킷런 사용하려고

# [5] 모델 학습
from sklearn.neighbors import KNeighborsClassifier  # k최근접이웃 찾기
from sklearn.neighbors import KNeighborsRegressor   # k최근접이웃 회귀
knr = KNeighborsRegressor()                         # 모델 객체 생성
knr.fit( train_input, train_target )                # 모델 학습 # (길이, 무게) # '길이'에 따른 '무게' 학습
print( knr.score( test_input, test_target ) )       # 모델 평가 # 0.9929281790592219 # 회귀모델에서는 결정계수라고 한다.
print( knr.predict( test_input ) )                  # 모델 예측 # [  61.4   78.   248.   117.   139. ] ~~

# [6] k최근접이웃 회귀는 이웃의 평균으로 예측한다. 하이퍼파라미터(k) 조절
# k =  이웃 개수 정하기
knr = KNeighborsRegressor() # 모델객체 생성
# 임의의 길이 생성, 임의의 물고기 길이 5부터 45까지 생성 ( 45개 임의값 )
x = np.arange( 5, 45 ).reshape( -1, 1 )
print (x) # 5 ~ 44 까지의 임의의 값

for k in [1, 3, 5, 10 ] : # 이웃개수를 4가지(1,3,5,10) 모델 학습
    knr.n_neighbors = k             # 현재 모델의 이웃개수 대입
    knr.fit( train_input, train_target ) # 총 4번 학습 예정
    print( knr.score(test_input, test_target) )            # 총 4번 학습 평가
    pred = knr.predict( x )   # 임의의 값으로 예측
    print( pred )           # 총 45개의 물고기길이의 무게 예측한다.
    # 시각화
    plt.scatter( train_input, train_target )
    plt.plot( x, pred )     # plot ( 선차트 이면서 회귀(예측선)) # x는 길이, y는 무게(예측)
    plt.title( f'k = {k}')
    plt.show()

# 차트 해석 : k는 이웃의 개수. k최근접 회귀는 이웃의 평균으로 예측한다.
# k가 1일 때 0.9918926744767643     # 특정한 자료에 튀는 데이터(노이즈/이상치)까지 적용될 수 있으므로 예측이 망가질 수 있다. # 과대 적합 훈련 # 너무 사용된 데이터 적음
# k가 3일 때 0.9766857219041255
# k가 5일 때 0.9929281790592219
# k가 10일 때 0.9742254836937329   # 많은 자료에 둔담하고 단순화된 자료까지 적용될 수 있으므로 예측이 망가질 수 있다. # 과소 적합 훈련 # 너무 사용된 데이터 많음

# k가 5일 때 가장 균형적인 추세 표현한다. 회귀선이 너무 꺾이거나 완만한 일직선(----)이 아니다.
# 결론] 튜닝 :  머신러닝에서는 가장 최적의 파라미터 찾는 과정