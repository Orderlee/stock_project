from flask_restful import Resource
from flask import Response, jsonify
from com_stock_api.korea_covid.korea_covid_dao import koreaDao

class KoreaApi(Resource):

    def __init__(self):
        self.dao = koreaDao

    def get(self):
        covid = self.dao.select_covid()
        return jsonify(covid[0])

