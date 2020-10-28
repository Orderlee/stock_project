from flask import Flask
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes
from com_stock_api.resource.korea_news import NewsDao
from com_stock_api.resource.korea_covid import KoreaDao
from com_stock_api.resource.korea_finance import StockDao
from com_stock_api.resource.korea_finance_recent import RecentStockDao
from com_stock_api.resource.korea_news_recent import RecentNewsDao
from com_stock_api.resource.kospi_pred import KospiDao
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={'r/api/*': {"origins":"*"}})

app.config['SQLALCHEMY_DATABASE_URI'] =url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api= Api(app)


with app.app_context():
    db.create_all()
    #print(f'db created ... ')
with app.app_context():
    news_count = NewsDao.count()
    print(f'****** News Total Count is {news_count} *******')
    if news_count[0] == 0:
        #NewsDao()
        n = NewsDao()
        n.bulk()

    covid_count = KoreaDao.count()
    print(f'***** Covid Count is {covid_count} *******')
    if covid_count[0] == 0:
        #KoreaDao()
        k = KoreaDao()
        k.bulk()

    stock_count = StockDao.count()
    print(f'**** Stock Count is {stock_count} **********')
    if stock_count[0] == 0:
        #StockDao()
        s = StockDao()
        s.bulk()

    recent_stock_count = RecentStockDao.count()
    print(f'**** Recent Stock Count is {recent_stock_count} ****')
    if recent_stock_count[0] == 0:
        RecentStockDao()
        #rs = RecentStockDao()
        #rs.bulk()
    
    recent_news_count = RecentNewsDao.count()
    print(f'******* Recent News Count is {recent_news_count}*****')
    if recent_news_count[0] == 0:
        RecentNewsDao()
        #rn = RecentNewsDao()
        #rn.bulk()

    

    # pred_count = KospiDao.count()
    # print(f'***** Pred Count is {pred_count} *********')
    # if pred_count[0] == 0:
    #     #KospiDao()
    #     kp = KospiDao()
    #     kp.bulk()
    




@app.route('/api/test')
def test():
    return{'test':'SUCCESS'}

initialize_routes(api)
