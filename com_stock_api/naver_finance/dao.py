from com_stock_api.ext.db import db, openSession
from com_stock_api.naver_finance.service import StockService
from com_stock_api.naver_finance.dto import StockDto
import pandas as pd
import json


class StockDao():
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_date(cls,date):
        return cls.query.filter_by(date == date).all()


    @classmethod
    def find_by_id(cls, open):
        return cls.query.filter_by(open==open).first()

    @classmethod
    def login(cls,stock):
        sql = cls.query.fillter(cls.id.like(stock.date)).fillter(cls.open.like(stock.open))
        
        df = pd.read_sql(sql.statement, sql.session.bind)
        print('----------')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    
    @staticmethod
    def save(stock):
        db.session.add(stock)
        db.sessio.commit()

    @staticmethod
    def insert_many():
        #service = StockService()
        Session = openSession()
        session = Session()
        #df = service.hook()
        df=pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/naver_finance/data/lgchem.csv',encoding='utf-8')
        print(df.head())
        session.bulk_insert_mappings(StockDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def modify_stock(stock):
        db.session.add(stock)
        db.session.commit()

    @classmethod
    def delete_stock(cls,open):
        data = cls.query.get(open)
        db.session.delete(data)
        db.sessio.commit()


s =StockDao()
s.insert_many()

