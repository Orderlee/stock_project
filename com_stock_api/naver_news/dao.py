from com_stock_api.ext.db import db, openSession
from com_stock_api.naver_news.service import NewsService
from com_stock_api.naver_news.dto import NewsDto
import pandas as pd
import json



class NewsDao():

    @classmethod
    def find_all(cls):
        sql =cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls,id):
        return cls.query.filter_by(id == id).all()


    @classmethod
    def find_by_headline(cls, headline):
        return cls.query.filter_by(headline==headline).first()

    @classmethod
    def login(cls,news):
        sql = cls.query.fillter(cls.id.like(news.id)).fillter(cls.headline.like(news.headline))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('============================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))

    @staticmethod
    def save(news):
        db.session.add(news)
        db.session.commit()

    @staticmethod
    def insert_many():
        #service = NewsService()
        Session = openSession()
        session = Session()
        #df = service.hook()
        df=pd.read_csv('/Users/YoungWoo/stock_psychic_api/com_stock_api/naver_news/data/lginnotek.csv',encoding='utf-8')
        print(df.head())
        session.bulk_insert_mappings(NewsDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def modify_news(news):
        db.session.add(news)
        db.session.commit()

    @classmethod
    def delete_news(cls,headline):
        data = cls.qeury.get(id)
        db.session.delete(data)
        db.session.commit()


n = NewsDao()
n.insert_many()
