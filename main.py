from flask import Flask
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes
from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

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

initialize_routes(api)

with app.app_context():
    db.init_app(app)
    print(f'db created ... ')
    db.create_all()