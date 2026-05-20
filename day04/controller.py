# app.py 연결하는 라우터 설정
from fastapi import APIRouter, Request
router = APIRouter( prefix= "/api/model" )


# 서비스 호출
from service import service

# 매핑 2개 만들기
# http://localhost:8000/api/model/learn
@router.post("/learn") # 매핑주소 설정
async def 학습요청( request : Request ) : # Request 객체
    list = await request.json() # request 객체 body값을 직접 json 변환
    print( list )
    return service.학습요청(list)

@router.post("/predict")
# http://localhost:8000/api/model/predict
async def 예측요청(car : dict) : # body정보를 딕셔너리 타입으로 받기
    print( car )
    return service.예측요청(car)