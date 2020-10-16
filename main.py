from fastapi import Depends, FastAPI,Header, HTTPException
from com_stock_api.korea_covid.korea_covid_api import KoreaApi
from com_stock_api.kospi_pred.kospi_pred_api import KospiApi
from com_stock_api.naver_finance.stock_api import StocksApi
from com_stock_api.naver_news.news_api import NewsApi

app = FastAPI()

async def get_token_header(x_token: str = Header(...)):
    if x_token != 'fake-super-secret-token':
        raise HTTPException(statues_code=400, detail="X-Token header incalid")

app.include_router(
    api = StocksApi().get_router(),
    prefix='/stocks',
    tags=['stocks']
)

app.include_router(
    
)