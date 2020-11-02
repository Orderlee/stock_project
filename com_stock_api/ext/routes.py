from com_stock_api.resources.korea_covid import KoreaCovid,KoreaCovids
from com_stock_api.resources.kospi_pred import Kospi,Kospis,lgchem_pred,lginnotek_pred
from com_stock_api.resources.korea_finance import Stock,Stocks,lgchem,lginnotek
from com_stock_api.resources.korea_news import News,News_
from com_stock_api.resources.korea_news_recent import RNews,RNews_

# from com_stock_api.resources.home import Home

# from com_stock_api.member.member_api import MemberApi,Members
# from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPredApi,MemberChurnPreds

# from com_stock_api.nasdaq_pred.prediction_api import Prediction,Predictions
# from com_stock_api.recommend_stock.recommend_stock_api import RecommendStockApi,RecommendStocks

# from com_stock_api.trading.trading_api import TradingApi,Tradings
# from com_stock_api.us_covid.us_covid_api import USCovid,USCovids

# from com_stock_api.yhfinance.yhfinance_api import YHFinance,YHFinances
# from com_stock_api.yhnews.yhnews_api import YHNews,YHNewses

def initialize_routes(api):
    # api.add_resource(Home,'/kospi')
    api.add_resource(KoreaCovid,'/api/koreacovid/<string:id>')
    api.add_resource(KoreaCovids,'/api/koreacovids')
    api.add_resource(RNews, '/api/rnews')
    api.add_resource(RNews_, '/api/rnews_') 
    api.add_resource(lgchem,'/kospi/lgchem')
    api.add_resource(lginnotek,'/kospi/lginnotek')
    api.add_resource(lgchem_pred, '/kospi/lgchem_pred')
    api.add_resource(lginnotek_pred, '/kospi/lginnotek_pred')
    api.add_resource(News,'/api/news/<string:id>')
    api.add_resource(News_,'/api/news_')


    # api.add_resource(MemberApi,'/api/memberapi/<string:id>')
    # api.add_resource(Members,'/api/members')
    # api.add_resource(MemberChurnPredApi,'/api/memberchurnpredapi/<string:id>')
    # api.add_resource(MemberChurnPreds,'/api/memberchurnpreds')
    # api.add_resource(Prediction,'/api/predictioin/<string:id>')
    # api.add_resource(Predictions,'/api/predictions')
    # api.add_resource(RecommendStockApi,'/api/recommendstockapi/<string:id>')
    # api.add_resource(TradingApi,'/api/tradingapi')
    # api.add_resource(USCovid,'/api/uscovid')
    # api.add_resource(USCovids,'/api/uscovids')
    # api.add_resource(YHFinance,'/api/yhfinance')
    # api.add_resource(YHFinances,'/api/yhfiances')
    # api.add_resource(YHNews,'/api/yhnews')
    # api.add_resource(YHNewses,'/api/yhnewses')






