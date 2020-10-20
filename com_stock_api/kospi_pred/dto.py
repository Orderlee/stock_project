from com_stock_api.ext.db import db 
from com_stock_api.korea_covid.dto import KoreaDto
from com_stock_api.naver_finance.dto import StockDto
from com_stock_api.naver_news.dto import NewsDto

class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    kospi_date : int = db.Column(db.DATETIME, primary_key = True, index=True)
    stock :int = db.Column(db.VARCHAR(30))
    price : int = db.Column(db.VARCHAR(30))

    covid_date: int = db.Column(db.DATETIME, db.ForeignKey(KoreaDto.date))
    stock_date: int = db.Column(db.DATETIME, db.ForeignKey(StockDto.date))
    news_date: int = db.Column(db.DATETIME, db.ForeignKey(NewsDto.date))


    def __init__(self,kospi_date, covid_date,stock_date,news_date,stock, price):
        self.kospi_date = kospi_date
        self.covid_date = covid_date
        self.stock_date = stock_date
        self.news_date= snews_date
        self.stock = stock
        self.price = price
    
    def __repr__(self):
        return f'kospi_date={self.kospi_date},covid_date={self.covid_date},stock_date={self.stock_date},news_date={self.news_date},  stock={self.stock},\
            price={self.price}'
            
    @property
    def json(self):
        return {
            'kospi_date': self.kospi_date,
            'covid_date': self.covid_date,
            'stock_date': self.stock_date,
            'news_date': self.news_date,
            'stock' : self.stock,
            'price' : self.price
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()




  