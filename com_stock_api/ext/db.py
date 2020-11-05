from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
Base = declarative_base()


db = SQLAlchemy()
config = {
    'user' : 'root',
    'password' : 'root',
    'host': '127.0.0.1',
    'port' : '3306',
    'database' : 'stockdb'

    # 'user':'stockpsychic',
    # 'password':'stockpsychic',
    # 'host':'stockpsychic.c2hh6gib8xaa.ap-northeast-2.rds.amazonaws.com',
    # 'port':'3306',
    # 'database':'stockpsychic'
    
}
charset ={'utf8':'utf8'}
url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"
# url = 'mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8'
Base = declarative_base()
engine = create_engine(url, pool_size=20, max_overflow=0)

def openSession():
     return sessionmaker(bind=engine)