from flask_restful import Resource
from flask import Response, jsonify
from com_stock_api.naver_finance.stock_dao import StockDao

class StocksApi(Resource):
    
    def __init__(self):
        self.dao = StockDao

    def get(self):
        stocks = self.dao.select_stocks()
        return jsonify(stocks[0])