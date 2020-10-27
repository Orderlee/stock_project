from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.naver_finance.dao import StockDao
from com_stock_api.naver_finance.dto import StockDto, StockVo
import json
from flask import jsonify

parser = reqparse.RequestParser()
parser.add_argument('id',type=str, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('open',type=int, required=True,help='This field should be a open')
parser.add_argument('close',type=int, required=True,help='This field should be a close')
parser.add_argument('high',type=int, required=True,help='This field should be a high')
parser.add_argument('low',type=int, required=True,help='This field should be a low')
parser.add_argument('amount',type=int, required=True,help='This field should be a amount')
parser.add_argument('stock',type=str, required=True,help='This field should be a stock')



class Stock(Resource):

    @staticmethod
    def Stock():
        args = parser.parse_args()
        print(f'Stock {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'

        params_str =''
        for key in params.keys():
            params_str += 'key: {}, value:{} <br>'.format(key,params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def get(id):
        print(f'Stock {id} added')
        try:
            stock = StockDao.find_by_id(id)
            if stock:
                return stock.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'Stock {args["id"]} updated')
        return {'code':0, 'message': 'SUCCESS'}, 200

    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Stock {args["id"]} deleted')
        return {'code':0, 'message': 'SUCCESS'}, 200

class Stocks(Resource):
    
    def get(self):
        sd = StockDao()
        sd.insert('naver_finance')

    def get(self):
        print('========stock==========')
        data = StockDao.find_all()
        return data, 200

class Auth(Resource):

    def post(self):
        body = request.get_json()
        stock = StockDto(**body)
        StockDto.save(stock)
        id = stock.id

        return {'id': str(id)}, 200

class Access(Resource):

    def __init__(self):
        print('=========stock2===========')

    def post(self):
        print('=======stock3=======')
        args = parser.parse_argse()
        stock = StockVo()
        stock.id = args.id 
        sstock.date = args.date
        print(stock.id)
        print(stock.date)
        data = StockDao.login(stock)
        return data[0], 200
