from com_stock_api.ext.db import db 
# from com_stock_api.korea_covid.dto import KoreaDto
# from com_stock_api.naver_finance.dto import StockDto
# from com_stock_api.naver_news.dto import NewsDto

class StockDto(db.Model):
    __tablename__ = 'naver_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    stock_id : int = db.Column(db.String(30), primary_key = True, index=True)
    date : date = db.Column(db.DATE)
    open : int = db.Column(db.VARCHAR(30))
    close : int = db.Column(db.VARCHAR(30))
    high : int = db.Column(db.VARCHAR(30))
    low :int = db.Column(db.VARCHAR(30))
    amount : int = db.Column(db.VARCHAR(30))
    stock : str = db.Column(db.VARCHAR(30))
    
    def __init__(self, stock_id, date, open, close, high, low, amount, stock):
        self.stock_id = stock_id
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.amount = amount
        self.stock = stock
    
    def __repr__(self):
        return f'stock_id={self.stock_id}, date={self.date}, open={self.open},\
            close ={self.close},high ={self.high},low ={self.low},amount ={self.amount},stock ={self.stock}'
            
    @property
    def json(self):
        return {
            'stock_id': self.stock_id,
            'date': self.date,
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




  