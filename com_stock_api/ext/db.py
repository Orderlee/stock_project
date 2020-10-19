from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

db = SQLAlchemy()
config = {
    'user':'root',
    'password':'root',
    'host':'127.0.0.1',
    'port':'8080',
    'database':'stockdb'
}
charset ={'utf8':'utf8'}
#url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"
url = 'mysql+mysqlconnector://root:root@127.0.0.1/stockdb?charset=utf8'

def openSession():
    ...