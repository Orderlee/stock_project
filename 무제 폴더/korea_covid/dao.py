from com_stock_api.ext.db import db, openSession
from com_stock_api.korea_covid.service import CovidService
from com_stock_api.korea_covid.dto import KoreaDto
import pandas as pd
import json

class KoreaDao():

    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_id(cls,id):
        return cls.query.filter_by(id == id).all()


    @classmethod
    def find_by_date(cls, date):
        return cls.query.filter_by(date == date).first()

    @classmethod
    def login(cls,covid):
        sql = cls.query.fillter(cls.id.like(covid.id)).fillter(cls.date.like(covid.date))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('======================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(covid):
        db.session.add(covid)
        db.session.commit()
    
    @staticmethod
    def insert_many():
        #service = CovidService()
        Session = openSession()
        session = Session()
        #df = service.hook()
        df=pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/korea_covid/data/kor&seoul.csv',encoding='UTF-8')
        print(df.head())
        session.bulk_insert_mappings(KoreaDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def modify_covid(covid):
        db.session.add(covid)
        db.session.commit()

    @classmethod
    def delete_covid(cls,date):
        data = cls.qeury.get(date)
        db.session.delete(data)
        db.sessio.commit()


k =KoreaDao
k.insert_many()

        