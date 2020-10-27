from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.naver_news.dao import NewsDao
from com_stock_api.naver_news.dto import NewsDto, NewsVo
import json
from flask import jsonify








parser = reqparse.RequestParser()
parser.add_argument('id', type=str, required=True, help='This field should be a id')
parser.add_argument('date', type=str, required=True, help='This field should be a date')
parser.add_argument('headline', type=str, required=True, help='This field should be a headline')
parser.add_argument('contents', type=str, required=True, help='This field should be a contents')
parser.add_argument('url', type=str, required=True, help='This field should be a url')
parser.add_argument('stock', type=str, required=True, help='This field should be a stock')
parser.add_argument('label', type=float, required=True, help='This field should be a label')


class News(Resource):

    @staticmethod
    def News():
        args = parser.parse_args()
        print(f'News {args["id"]} added')
        parmas = json.loads(request.get_data(), encoding='utf-8')
        if len (parmas) == 0:
            return 'No parameter'
        
        params_str=''
        for key in parmas.keys():
            params_str += 'key:{}, value:{}<br>'.format(key, parmas[key])
        return {'code':0, 'message': 'SUCCESS'}, 200
    
    @staticmethod
    def post(id):
        print(f'News {id} added')
        try:
            news = NewsDao.find_by_id(id)
            if news:
                return news.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'News {args["id"]} updated')
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'News {args["id"]} deleted')
        return {'code':0, 'message':'SUCCESS'}, 200

class News_(Resource):
    
    def get(self):
        nd = NewsDao()
        nd.insert('naver_news')

    def get(self):
        print('=======news==========')
        data = NewsDao.find_all()
        return data, 200

class Auth(Resource):

    def post(self):
        body = request.get_json()
        news = NewsDto(**body)
        NewsDao.save(news)
        id = news.id

        return {'id': str(id)}, 200

class Access(Resource):

    def __init__(self):
        print('========== news2 ==========')

    def post(self):
        print('========= news3 =========')
        args = parser.parse_args()
        news = NewsVo()
        news.id = args.id
        news.headline = args.headline
        print(news.id)
        print(news.headline)
        data = NewsDao.login(news)
        return data[0], 200