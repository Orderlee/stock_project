from com_stock_api.ext.db import db, openSession
from com_stock_api.kospi_pred.service import KospiService
from com_stock_api.korea_covid.dto import KospiDto
import pandas as pd
import json

class KospiDao():

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
    def login(cls,kospi):
        sql = cls.query.fillter(cls.id.like(kospi.id)).fillter(cls.date.like(kospi.date))

        df = pd.read_sql(sql.statement, sql.session.bind)
        print('==================')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_sjon(orient='records'))

    @staticmethod
    def save(kospi):
        db.session.add(kospi)
        db.session.commit()

    @staticmethod
    def insert_many():
        service = KospiService()
        Session = openSession()
        session = Session()
        df = service.hook()
        print(df.head())
        session.bulk_insert_mappings(KospiDto, df.to_dict(orient='records'))
        session.commit()
        session.close()

    @staticmethod
    def modify_kospi(kospi):
        db.session.add(kospi)
        db.session.commit()

    @classmethod
    def delete_kospi(cls,date):
        data = cls.query.get(date)
        db.session.delete(data)
        db.session.commit()

# ko = KospiDao
# ko.insert_many()