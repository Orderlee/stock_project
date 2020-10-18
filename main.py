from flask import Flask
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes
from com_stock_api.korea_covid.korea_covid_api import KoreaApi
from com_stock_api.kospi_pred.kospi_pred_api import KospiApi
from com_stock_api.naver_finance.stock_api import StocksApi
from com_stock_api.naver_news.news_api import NewsApi

app = Flask(__name__)

print('========= url ===========')
print(url)
app.config['SQLALCHEMY_DATABASE_URL'] =url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api= Api(app)

@app.before_first_request

def create_tables():
    db.create_all()

with app.app_context():
    db.create_all()