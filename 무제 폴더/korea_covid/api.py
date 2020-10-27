from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.korea_covid.dao import KoreaDao
from com_stock_api.korea_covid.dto import KoreaDto, KoreaVo
import json
from flask import jsonify

parser = reqparse.RequestParser()
parser.add_argument('id',type=str, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('seoul_cases',type=int, required=True,help='This field should be a seoul_cases')
parser.add_argument('seoul_death',type=int, required=True,help='This field should be a seoul_deate')
parser.add_argument('total_cases',type=int, required=True,help='This field should be a total_cases')
parser.add_argument('total_death',type=int, required=True,help='This field should be a password')


class KoreaCovid(Resource):
    
    @staticmethod
    def Covid():
        args = parser.parse_args()
        print(f'Covid {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'
        params_str = ''
        for key in params.keys():
            params_str += 'key: {}, value:{}<br>'.format(key, params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
        
    
    @staticmethod
    def post(id):
        print(f'Covid {id} added')
        try:
            covid = KoreaDao.find_by_id(id)
            if covid:
                return covid.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'Covid {args["id"]} updated')
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Covid {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class KoreaCovids(Resource):

    def get(self):
        kd = KoreaDao()
        kd.insert_many('korea_covid')

    def get(self):
        print('======kc========')
        data = KoreaDao.find_all()
        return data, 200

class Auth(Resource):

    def post(self):
        body = request.get_json()
        covid = KoreaDto(**body)
        KoreaDao.save(covid)
        id = covid.id

        return {'id': str(id)}, 200

class Access(Resource):

    def __init__(self):
        print('=======kc2=========')

    def post(self):
        print('=======kc3===========')
        args = parser.parse_args()
        covid = KoreaVo()
        covid.id = args.id
        covid.date = args.date
        print(covid.id)
        print(covid.date)
        data = KoreaDao.login(covid)
        return data[0], 200
