from com_stock_api.naver_finance.stock_api import StocksApi

def initialize_routes(api):
    api.add_resource(StocksApi,'api/kospi')