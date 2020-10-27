from typing import List
from flask import request
from flask_restful import Resource, reqparse

from com_stock_api.kospi_pred.dao import KospiDao
from com_stock_api.kospi_pred.dto import KospiDto, KospiVo
import json
from flask import jsonify

parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field should be a id')
parser.add_argument('date', type=str, required=True, help='TThis field should be a date')
parser.add_argument('covid_id', type=int, required=True, help='This field should be a covid_date')
parser.add_argument('stock_id', type=int, required=True, help='This field should be a stock_date')
parser.add_argument('news_id', type=int, required=True, help='This field should be a news_date')
parser.add_argument('ticker', type=str, required=True, help='This field should be a ticker')
parser.add_argument('price', type=int, required=True, help='This field should be a price')

class Kospi(Resource):

    @staticmethod
    def Kospi(self):
        args = parser.parse_args()
        print(f'Kospi {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'
        params_str=''
        for key in params.keys():
            params_str += 'key:{},value:{}<br>'.format(key, params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def post(id):
        print(f'Kospi{id} added')
        try:
            kospi = KospiDao.find_by_id(id)
            if kospi:
                return kospi.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
            
    @staticmethod
    def update():
            args = parser.arse_args()
            print(f'Kospi {args["id"]} updated')
            return {'code':0, 'message':'SUCCESS'}, 200
            
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Kospi {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class Kospis(Resource):
    def get(self):
        kd = KospiDao()
        kd.insert_many('kospi_pred')

    def get(self):
        print('======kc========')
        data = KospiDao.find_all()
        return data, 200

class Auth(Resource):

    def post(self):
        body = request.get_json()
        kospi = KospiDto(**body)
        KospiDao.save(kospi)
        id = kospi.id

        return {'id': str(id)}, 200

class Access(Resource):
    
    def __init__(self):
        print('=======kc2=========')

    def post(self):
        print('=======kc3===========')
        args = parser.parse_args()
        kospi = KospiVo()
        kospi.id = args.id
        kospi.date = args.date
        print(kospi.id)
        print(kospi.date)
        data = KospiDao.login(kospi)
        return data[0], 200