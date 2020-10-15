from com_stock_api.ext.db import Base
from sqlalchemy import Column,Integer, String, ForeignKey, create_engine, DATE
from sqlalchemy.orm import sessionmaker
from sqlalchemy.dialects.mysql import DECIMAL, VARCHAR, LONGTEXT

class KoreaCovid(Base):
    __tablename__ = 'korea_covid'
    __table_args__ = {'mysql_collate':'utf_general_ci'}

    date = Column(DATE, primary_key = True, index = True)
    seoul_cases = Column(VARCHAR(30))
    seoul_death = Column(VARCHAR(30))
    total_cases = Column(VARCHAR(30))
    total_death = Column(VARCHAR(30))


engine = create_engine('mysql+mysqlconnector://root:root@127.0.0.1/mariadb?charset=utf8',encoding='utf8',echo=True)
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()
session.add(KoreaCovid(date='2020-02-12',seoul_cases='222',seoul_death='333',total_cases='222',total_death='333'))
query = session.query(KoreaCovid)
print(query)
for i in query:
    print(i)

session.commit()