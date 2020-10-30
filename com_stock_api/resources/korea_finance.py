import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
import math
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class StockDto(db.Model):
    __tablename__ = 'korea_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    open : int = db.Column(db.String(30))
    close : int = db.Column(db.String(30))
    high : int = db.Column(db.String(30))
    low :int = db.Column(db.String(30))
    volume : int = db.Column(db.String(30))
    ticker : str = db.Column(db.String(30))
  
    def __init__(self,id,date, open, close, high, low, volume, ticker):
        self.id = id
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.ticker = ticker
    
    def __repr__(self):
        return f'id={self.id}, date={self.date}, open={self.open},\
            close={self.close}, high={self.high}, low={self.low}, volume={self.volume}, ticker={self.ticker}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'ticker' : self.ticker
        }

class StockVo:
    id: int = 0
    date: str=''
    open: int =''
    close: int =''
    high: int =''
    low: int =''
    volume: int =''
    ticker: str=''


Session = openSession()
session= Session()


class StockDao(StockDto):

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")


    def bulk(self):
        path = self.data
        companys = ['lgchem','lginnotek']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            print(df.head())
            session.bulk_insert_mappings(StockDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(StockDto.id)).one()

    @staticmethod
    def save(stock):
        db.session.add(stock)
        db.sessio.commit()

    @staticmethod
    def update(stock):
        db.session.add(stock)
        db.session.commit()

    @classmethod
    def delete(cls,open):
        data = cls.query.get(open)
        db.session.delete(data)
        db.sessio.commit()
    
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_date(cls,date):
        return session.query.filter(StockDto.date.like(date)).one()

    @classmethod
    def find_by_id(cls, open):
        return session.query.filter(StockDto.open.like(open)).one()

    @classmethod
    def find_by_close(cls,close):
        return session.query.filter(StockDto.close.like(close)).one()
    
    @classmethod
    def find_by_low(cls,low):
        return session.query.filter(StockDto.low.like(low)).one()

    @classmethod
    def find_by_volume(cls,volume):
        return session.query.filter(StockDto.volume.like(volume)).one()

    @classmethod
    def find_by_ticker(cls,ticker):
        return session.query.filter(StockDto.ticker.like(ticker)).one()

    @classmethod
    def find_by_high(cls,high):
        return session.query.filter(StockDto.high.like(high)).one()

    @classmethod
    def find_By_id(cls,id):
        return session.query.filter(StockDto.id.like(id)).one()




# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================






parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('open',type=int, required=True,help='This field should be a open')
parser.add_argument('close',type=int, required=True,help='This field should be a close')
parser.add_argument('high',type=int, required=True,help='This field should be a high')
parser.add_argument('low',type=int, required=True,help='This field should be a low')
parser.add_argument('volume',type=int, required=True,help='This field should be a amount')
parser.add_argument('ticker',type=str, required=True,help='This field should be a stock')



class Stock(Resource):

    @staticmethod
    def post():
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
    
    @staticmethod
    def get():
        sd = StockDao()
        sd.insert('korea_finance')
    
    @staticmethod
    def get():
        data = StockDao.find_all()
        return data, 200

class Auth(Resource):
    
    @staticmethod
    def post():
        body = request.get_json()
        stock = StockDto(**body)
        StockDto.save(stock)
        id = stock.id

        return {'id': str(id)}, 200

class Access(Resource):

    @staticmethod
    def post():
        args = parser.parse_argse()
        stock = StockVo()
        stock.id = args.id 
        sstock.date = args.date
        print(stock.id)
        print(stock.date)
        data = StockDao.login(stock)
        return data[0], 200


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================



