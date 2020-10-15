from com_stock_api.ext.db import Base
from sqlalchemy import Column,Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT

class News(Base):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf_general_ci'}

    date = Column(DATE, primary_key = True, index = True)
    symbol = Column(VARCHAR(30))
    headline = Column(VARCHAR(30))
    url = Column(VARCHAR(30))


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8',encoding='utf8',echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(News(date='2020.02.02',symbol='lg화학',headline='dkdif',url='hhtpt;//dkdn'))
query = session.query(News)
print(query)
for i in query:
    print(i)

session.commit()