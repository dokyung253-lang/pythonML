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
        for degree in [1,2,3,4,5] : 
            poly = PolynomialFeatures( degree= degree, include_bias = False )
            poly.fit( train_input )
            train_poly = poly.transform( train_input )
            test_poly = poly.transform( test_input )
            print( f'{degree}')
            lr = LinearRegression()
            lr.fit( train_poly, train_target )
            r2 = lr.score( test_poly, test_target )
            optimization.append( {'r2': r2, 'model':lr, 'poly':poly, 'degree':degree, 'scaler':None, 'alpha': None })
            ss = StandardScaler()
            ss.fit( train_poly )
            train_scaled = ss.transform( train_poly )
            test_scaled = ss.transform( test_poly )
            for alpha in[0.01, 0.1, 1, 10, 100] :
                ridge = Ridge( alpha= alpha)
                ridge.fit( train_scaled, train_target )
                r2 = ridge.score( test_scaled, test_target )
                print( f'{degree} 차수의 릿지 강도 : {alpha}의 결정계수 : {r2}')
                optimization.append( {'r2': r2, 'model':ridge, 'poly':poly, 'degree':degree, 'scaler':ss, 'alpha': alpha })

                lasso = Lasso( alpha= alpha )
                lasso.fit( train_scaled, train_target )
                r2 = lasso.score( test_scaled, test_target )
                print( f'{degree} 차수의 릿지 강도 : {alpha}의 결정계수 : {r2}')
                optimization.append( {'r2': r2, 'model':ridge, 'poly':poly, 'degree':degree, 'scaler':ss, 'alpha': alpha })
        list = []
        return True
    # 2. 
    def 예측요청( self, car) :
        return 10000
service = Service()
