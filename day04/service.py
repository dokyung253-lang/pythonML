# 서비스
import pandas as pd
class Service :
    def __init__(self) :
        self.model = None

    def 학습요청( self, carList ) :
        # 2
        df = pd.DataFrame( carList )
        train_full = df[['평균 연비', '누적 주행거리(km)', '출고 후 경과 월수', '사고 감가 건수', '소유자 변경 횟수']] 
        train_target = df['매매 가격(단위: 만 원)']
        from sklearn.model_selection import train_test_split
        train_input, test_input, train_target, test_target = train_test_split( train_full, train_target, test_size= 0.1, random_state= 42)
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LinearRegression, Ridge, Lasso

        optimization = []

        return True
    # 2. 
    def 예측요청( self, car) :
        return 10000
service = Service()
