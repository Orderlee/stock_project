from com_stock_api.ext.db import db 
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from com_stock_api.korea_covid.service import CovidService

config = {
    'user': 'root',
    'password': 'root',
    'host': '127.0.0.1',
    'port': '3306',
    'database': 'mariadb'
}

charset = {'utf8': 'utf8'}
url = f'mysql+mysqlconnector://{config["user"]}:{config["password"]}@{config["host"]}:{config["port"]}/{config["database"]}?charset=utf8'
engine = create_engine(url)


class KoreaDto(db.Model):
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    seoul_cases :int = db.Column(db.String(30))
    seoul_death :int = db.Column(db.String(30))
    total_cases : int = db.Column(db.String(30))
    total_death : int = db.Column(db.String(30))
    
    def __init__(self, id,date, seoul_cases, seoul_death, total_cases, total_death):
        self.date = date
        self.seoul_cases = seoul_cases
        self.seoul_death = seoul_death
        self.total_cases = total_cases
        self.total_death = total_death
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, seoul_cases={self.seoul_cases},\
            seoul_death={self.seoul_death},total_cases={self.total_cases},total_deatb={self.total_death}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'seoul_cases' : self.seoul_cases,
            'seoul_death' : self.seoul_death,
            'total_cases' : self.total_cases,
            'total_death' : self.total_death
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


service = CovidService()
Session = sessionmaker(bind=engine)
s = Session()
df = service.hook()
print(df.head())
s.bulk_insert_mappings(KoreaDto, df.to_dict(orient="records"))
s.commit()
s.close()


  