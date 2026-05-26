# wine.csv , alcohol,sugar,pH,class

# [1]
import pandas as pd
df = pd.read_csv( './day06/wine.csv')
data = df[['alcohol', 'sugar', 'pH']]      # 와인들의 속성 3개
target = df['class']                                # 1:화이트와인 0:레드와인

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split( data, target, random_state= 42 )

# [2] 결정트리
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit( train_input, train_target )
print( dt.score( test_input, test_target ) ) # 0.8541538461538462

# [3] 교차 검증
from sklearn.model_selection import cross_validate
# cross_validate( 학습모델, 학습세트, 정답세트)
# 교차검증은 전체 데이터를 N등분(폴드)하여 돌아가면서 검증한다. 기본값은 5등분
# 즉] 데이터를 여러조각으로 나누어 학습하는 방법
scores = cross_validate( dt, train_input, train_target )
print( scores )
# {'fit_time': array([0.00681114, 0.00433445, 0.0046401 , 0.00430298, 0.00401664]), 
# 'score_time': array([0.00115323, 0.00087929, 0.00201917, 0.00089478, 0.00100923]),  
# 'test_score': array([0.85538462, 0.85128205, 0.88090349, 0.85420945, 0.84086242])}

import numpy as np
print( np.mean( scores['test_score'] ) ) # 5등분 학습의 평균 검증 점수 # 0.8571429474016743

# 
from sklearn.model_selection import StratifiedKFold
# n_splits= N등분 # 데이터를 N등분으로 하여 교차검증 수행한다.
# shuffle= 
splits = StratifiedKFold( n_splits= 10, shuffle= True, random_state= 42)
scores = cross_validate( dt, train_input, train_target , cv = splits )
print( scores )
# 'test_score': array([0.85245902, 0.85655738, 0.8788501 , 0.84188912, 0.87679671,
#        0.85010267, 0.85010267, 0.86036961, 0.84394251, 0.87268994])}
print( np.mean( scores['test_score'] ) ) # 10등분 학습의 평균 검증 점수 # 0.858375971993133 # 조금 증가

# [4] 그리드 서치, 최적의 파러미터(학습에 필요한 변수) 찾기
from sklearn.model_selection import GridSearchCV
#(1) 여러개 '최소불순도' 설정, 불순도란? 0에 가까울수록 예측값이 명확하다.(과대적합) 0.5에 가까울수록 예측값이 애매하다.
params = {'min_impurity_decrease' : [0.0001, 0.0002, 0.0003, 0.0004, 0.0005 ]}
# (2)
# GridSearchCV( 트리모델 , {파라미터들}, n_jobs = -1)
# n_jobs = -1 : 컴퓨터내 모든 CPU 코어(연산흐름단위=사람뇌) 사용하여 병렬(쓰레드) 연산, 즉] CPU 최대 사용
gs = GridSearchCV( DecisionTreeClassifier( random_state= 42), params, n_jobs = -1 )
# (3) 그리드 서치 학습
gs.fit( train_input, train_target ) # 기본값으로 교차검증 5번
dt = gs.best_estimator_ # 최적의 파라미터로 학습 결과
print( dt.score( test_input, test_target ) ) # 최적의 파라미터로 학습점수 # 0.8670769230769231 # 조금 증가
print( gs.best_score_ ) # 0.8731517927657558
print( gs.best_params_ ) # {'min_impurity_decrease': 0.0003}
# print( gs.cv_results_ ) # 기본값으로 교차검증 5가 적용됨

# [5] 다중 파라미터 
