# 파이썬 통신 주고받을 수 있는 FASTAPI
from fastapi import FastAPI
import uvicorn
app = FastAPI()

if __name__ == '__main__' :
    uvicorn.run( 'app:app', host = '127.0.0.1',
                port = 8000, reload = True )

import controller
app.include_router( controller.router )