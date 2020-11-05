from com_stock_api.resources.korea_covid import KoreaCovid,KoreaCovids
from com_stock_api.resources.kospi_pred import Kospi,Kospis,lgchem_pred,lginnotek_pred
from com_stock_api.resources.korea_finance import Stock,Stocks,lgchem,lginnotek
from com_stock_api.resources.korea_news import News,News_,Lgchem_Label,Lginnotek_Label
from com_stock_api.resources.korea_news_recent import RNews,RNews_, lgchemNews,lginnoteknews
#from com_stock_api.resources.home import Home

# from com_stock_api.member.member_api import MemberApi,Members
# from com_stock_api.memberChurn_pred.memberChurn_pred_api import MemberChurnPredApi,MemberChurnPreds

# from com_stock_api.nasdaq_pred.prediction_api import Prediction,Predictions
# from com_stock_api.recommend_stock.recommend_stock_api import RecommendStockApi,RecommendStocks

# from com_stock_api.trading.trading_api import TradingApi,Tradings
from com_stock_api.resources.uscovid import USCovid, USCovids
from com_stock_api.resources.yhfinance import YHFinance,YHFinances,AppleGraph,TeslaGraph
from com_stock_api.resources.investingnews import Investing, AppleSentiment, TeslaSentiment

def initialize_routes(api):
    #api.add_resource(Home,'/kospi')
    api.add_resource(KoreaCovid,'/kospi/koreacovid')

    api.add_resource(lgchemNews,'/kospi/lgchemNews')
    api.add_resource(lginnoteknews,'/kospi/lginnoteknews')
    api.add_resource(Lgchem_Label, '/kospi/lgchem_label')
    api.add_resource(Lginnotek_Label, '/kospi/lginnotek_label')

    api.add_resource(lgchem,'/kospi/lgchem')
    api.add_resource(lginnotek,'/kospi/lginnotek')

    api.add_resource(lgchem_pred, '/kospi/lgchem_pred')
    api.add_resource(lginnotek_pred, '/kospi/lginnotek_pred')
    


    api.add_resource(AppleSentiment, '/nasdaq/apple_sentiment')
    api.add_resource(TeslaSentiment, '/nasdaq/tesla_sentiment')
    api.add_resource(USCovid, '/nasdaq/uscovid')






