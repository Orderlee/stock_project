from flask_restful import Resource
from flask import Response, jsonify
from com_stock_api.naver_news.news_dao import NewsDao

class NewsApi(Resource):

    def __init__(self):
        self.dao = NewsDao()

    def get(self):
        news = self.dao.self_news()
        return jsonify(news[0])