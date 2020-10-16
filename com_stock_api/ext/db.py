from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.ext.declarative import declarative_base
Base = declarative_base()

db = SQLAlchemy()
config = {
    'user':'root',
    'password':'root',
    'host':'localhost',
    'port':'3306',
    'database':'stockdb'
}

def openSession():
    ...