# 회귀분석/분류분석/k-최근점 -> 지도학습(정답있음)
# 군집분석 -> 비지도학습(정답 없음)
# ==========================================
# 데이터 준비 (3차원 특성: 무게, 당도, 단단함)
# ==========================================
import pandas as pd
data = {  
    'weight': [110, 160, 130, 320, 370, 300, 55, 65, 60, 210, 220, 200, 90, 80, 100, 190, 180, 170, 100, 90,
               140, 280, 320, 130, 200, 140, 250, 150, 70, 80, 200, 300, 220, 140, 180, 230, 220, 250],
    'sweetness': [6.2, 7.2, 6.8, 8.1, 8.6, 8.1, 5.2, 5.7, 6.1, 7.2, 7.6, 6.7, 7.3, 6.9, 7.3, 7.5, 7.4, 7.3, 7.0, 6.8,
                  6.9, 8.0, 8.1, 6.7, 7.0, 6.6, 7.8, 7.1, 6.7, 6.5, 7.0, 7.6, 7.3, 7.0, 7.2, 7.5, 7.4, 7.7],
    'hardness': [7.8, 6.5, 7.1, 4.2, 3.5, 3.9, 8.9, 8.4, 8.1, 5.8, 5.2, 6.1, 7.3, 7.5, 7.0, 5.9, 6.2, 6.4, 7.2, 7.6,
                 6.8, 4.5, 4.1, 7.0, 5.7, 6.9, 4.9, 6.6, 8.2, 8.5, 5.8, 4.0, 5.3, 6.7, 6.1, 5.0, 5.2, 4.7]
}
df = pd.DataFrame(data)

# 테스트용
newDf = pd.DataFrame({'weight': [110], 'sweetness': [7.0], 'hardness': [7.5]})
features = ['weight', 'sweetness', 'hardness']

# [1] k-Means : 정해진(k)개수 만큼의 그룹/군집 , 중심점의 평균 계산
from sklearn.cluster import KMeans
# n_clusters= k, 그룹 갯수 설정 , 2이면 2개의 그룹으로 군집화한다.
# random_state , 그룹/군집/클러스터 설정하기 위한 초기 중심점 무작위 난수 생성값(시드)
km = KMeans( n_clusters= 2 , random_state=42 )      # 모델 객체 생성
km.fit( df[ features ])                             # 모델 (비지도) 학습 # target(정답/타겟)이 없다
print( km.predict(newDf[features]) )                # 모델 예측(클러스터/ 군집화) , 0 : 그룹 A에 소속된다.
print( km.labels_ )                                 # 행 마다의 군집번호, 0: 그룹A, 1:그룹B
# 시각화
import matplotlib.pyplot as plt
plt.scatter( df['weight'], df['sweetness'], c= km.labels_ )
plt.scatter( newDf['weight'], newDf['sweetness'] , marker= '^' )
plt.show()
# 특성들 간의 서로다른 단위의 의미 -> 스케일링
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaledDf = ss.fit_transform( df[features]) # fit + transform
scaledNewDf = ss.transform( newDf[features] )
# 스케일링 이후 시각화
plt.scatter( scaledDf[:,0], scaledDf[:,1], c=km.labels_ )
plt.scatter( scaledNewDf[:,0], scaledNewDf[:,1], marker='^')
plt.show()

# [2] 최적의 K(그룹수) 찾기, 엘보우 방법( 오차 측정 )
sse = [] # 오차들을 저장하는 리스트
for k in range( 1, 11 ) : # 1~ 10 개의 클러스터
    km =  KMeans( n_clusters= k, random_state= 42 ) # k 개수만큼 클러스터가 존재하는 모델 생성
    km.fit( scaledDf ) # 스케일링된 자료 학습
    sse.append( km.inertia_ ) # 군집(클러스터) 내 자료들간 오차의 제곱 합 측정
print( sse ) # 클러스터가 많아지면 오차의 제곱합이 줄어든다. 더 촘촘해짐
# 오차 시각화
plt.plot( range( 1, 11 ), sse , marker = 'o')
plt.show()
# 엘보우 포인트 : SSE(오차의제곱합) 급격하게 줄어든 포인트 => 최적의 K : 2~3