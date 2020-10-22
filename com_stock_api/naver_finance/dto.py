from com_stock_api.ext.db import db 
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine
from com_stock_api.naver_finance.service_test import StockService


config = {
    'user' : 'root',
    'password' : 'root',
    'host': '127.0.0.1',
    'port' : '3306',
    'database' : 'stockdb'
}
charset ={'utf8':'utf8'}
url = f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}?charset=utf8"

engine = create_engine(url)


class StockDto(db.Model):
    __tablename__ = 'naver_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    open : int = db.Column(db.String(30))
    close : int = db.Column(db.String(30))
    high : int = db.Column(db.String(30))
    low :int = db.Column(db.String(30))
    amount : int = db.Column(db.String(30))
    stock : str = db.Column(db.String(30))
  
    def __init__(self,id,date, open, close, high, low, amount, stock):
        self.id = id
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.amount = amount
        self.stock = stock
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, open={self.open},\
            close ={self.close},high ={self.high},low ={self.low},amount ={self.amount},stock ={self.stock}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'finance': self.date,
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'amount': self.amount,
            'stock' : self.stock
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()


service = StockService()
Session = sessionmaker(bind = engine)
s = Session()
df_temp = service.stockcrawl()
print(df_temp.head())
s.bulk_insert_mappings(StockDto, df_temp.to_dict(orient="records"))
s.commit()
s.close()




  