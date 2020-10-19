from com_stock_api.home.api import Home
from com_stock_api.korea_covid.api import KoreaCovid,KoreaCovids
from com_stock_api.kospi_pred.api import Kospi,Kospis
from com_stock_api.naver_finance.api import Stock,Stocks
from com_stock_api.naver_news.api import News,News_

def initialize_routes(api):
    api.add_resource(Home,'/api')
    api.add_resource(KoreaCovid,'/api/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/api/koreacovids')
    api.add_resource(Kospi,'/api/kospi/<string:id>')
    api.add_resource(Kospis,'/api/kospis')
    api.add_resource(Stock,'/api/stock/<string:id>')
    api.add_resource(Stocks,'/api/stocks')
    api.add_resource(News,'/api/news/<string:id>')
    api.add_resource(News_,'/api/news_')



