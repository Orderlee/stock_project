from com_stock_api.naver_finance.stock_api import StocksApi
#from com_stock_api.home.home_api import HomeApi

def initialize_routes(api):
    api.add_resource(StocksApi,'/api/kospi')
    #api.add_resource(HomeApi,'/api/')
    #api.add_resource()
    #api.add_resource()