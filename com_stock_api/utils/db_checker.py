from sqlalchemy import create_engine
from sqlalchemy import desc, or_
from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.mysql import \
BIGINT, BINARY, BIT, BLOB, BOOLEAN, CHAR, DATE, \
DATETIME, DECIMAL, DECIMAL, DOUBLE, ENUM, FLOAT, INTEGER, \
LONGBLOB, LONGTEXT, MEDIUMBLOB, MEDIUMINT, MEDIUMTEXT, NCHAR, \
NUMERIC, NVARCHAR, REAL, SET, SMALLINT, TEXT, TIME, TIMESTAMP, \
TINYBLOB, TINYINT, TINYTEXT, VARBINARY, VARCHAR, YEAR

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

engine = create_engine('mysql+pymysql://root:root@127.0.0.1/stockdb?charset=utf8', encoding='utf8', echo=True)

Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    __table_args__ = {'mysql_collate': 'utf8_general_ci'}

    id = Column(Integer, primary_key=True)
    name = Column(VARCHAR(255), index=True)
    fullname = Column(VARCHAR(255))
    password = Column(VARCHAR(255))
    content = Column(LONGTEXT)

    def __repr__(self):
        return "<User(id='%s', name='%s', fullname='%s', password='%s', content='%s')>" % \
        (self.id, self.name, self.fullname, self.password, self.content)

Base.metadata.create_all(engine)

Session = sessionmaker(bind=engine)
session = Session()

'''
# create
session.add(User(name='ed1', fullname='Ed Jones1', password='edspassword1', content='hello'))
session.add(User(name='ed1', fullname='Ed Jones1', password='edspassword1', content='hello'))
session.add(User(name='ed1', fullname='Ed Jones1', password='edspassword1', content='hello'))
session.add(User(name='ed1', fullname='Ed Jones1', password='edspassword1', content='hello'))
session.add(User(name='ed1', fullname='Ed Jones1', password='edspassword1', content='hello'))

# update
upt = session.query(User).filter(User.id == 3).update({"content":"update this content"})
print(upt)

# delete
de = session.query(User).filter(User.id == 1).delete()
print(de)
print(session.query(User).count())
'''
# read# query = session.query(User).order_by(desc(User.id)).limit(1)
query = session.query(User).filter(or_(User.id == 1, User.id == 2)).order_by(desc(User.id)).limit(1)
print(query)
for each in query:
  print(each)

session.commit()