# wine.csv , alcohol,sugar,pH,class

# [1]
import pandas as pd
df = pd.read_csv( './day06/wine.csv')
data = df[['alcohol', 'sugar', 'pH']]      # 와인들의 속성 3개
target = df['class']                                # 1:화이트와인 0:레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( data, target, random_state= 42 )

# [2] 결정트리 ( 분류모델 )
from sklearn.tree import DecisionTreeClassifier # 의사결정 트리 분류
dt = DecisionTreeClassifier() # 모델객체생성
dt.fit( train_input, train_target ) # 모델학습
print( dt.score( train_input, train_target)) # 모델 정확도 # 0.9973316912972086
print( dt.score( test_input, test_target)) # 0.8498461538461538
print( dt.predict( test_input[:5] )) # 모델 5개만 예측 # [1. 0. 1. 1. 1.]
# [3] 결정트리 시각화
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree                 # 트리시각화
plot_tree( dt, max_depth= 1, feature_names= ['alcohol', 'sugar', 'pH'], class_names= ['Red Wine', 'White Wine'], filled= True ) # plot_tree (트리모델, max_depth = 가지,)
plt.show(  )
# 트리 : 전체적인 구조
# 노드 : 사각형 상자 하나하나 의미, 가장위에 있는 노드: 루트(root)노드
# 노드 속성  
    # - value = [예측타겟수] # [85, 2097] 0으로 예측하는 수 : 85개 , 1으로 예측하는 수 : 2097
    # - gini = 불순도 # 0.075 
        # 0(순수한 값: 특정예측값으로 모여있음) ~ 0.3 (혼란한 값: 예측값이 섞임)
    # - sugar = 특성 # sugar <= 4.15 보다 작으면 true(왼쪽 노드로 이동), false(오른쪽 노드로 이동)

# [4] 특성중요도
# 각 특성이 트리모델에 얼마나 중요한 역할 하는지 수치 # 합은 1
print( dt.feature_importances_) # [0.23677575 0.51889745 0.2443268 ]
print( dt.feature_importances_[0]) # alchol # sugar # pH

# [5] 최소한의 불순도( gini )설정
dt = DecisionTreeClassifier( random_state= 42, min_impurity_decrease= 0.0005 )
dt.fit( train_input, train_target )
print( dt.score( train_input, train_target)) # 0.8975779967159278
print( dt.score( test_input, test_target)) # 0.8590769230769231 # 과대적합 최소화