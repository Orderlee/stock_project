from com_stock_api.ext.db import db 
# from com_stock_api.kospi_pred.dto import KospiDto
# from com_stock_api.korea_covid.dto import KoreaDto
# from com_stock_api.naver_finance.dto import StockDto

class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    news_id : int = db.Column(db.String(30), primary_key = True, index=True)
    date : date = db.Column(db.DATE)
    symbol :str = db.Column(db.VARCHAR(30))
    headline :str = db.Column(db.VARCHAR(30))
    url: str = db.Column(db.VARCHAR(30))
    
    def __init__(self, news_id, date, symbol, headline, url):
        self.news_id = news_id
        self.date = date
        self.symbol = symbol
        self.headline = headline
        self.url = url
    
    def __repr__(self):
        return f'news_id={self.news_id}, date={self.date}, symbol={self.symbol},\
            headline={self.headline},url={self.url}'
            
    @property
    def json(self):
        return {
            'news_id': self.news_id,
            'date': self.date,
            'symbol' : self.symbol,
            'headline' : self.headline,
            'url' : self.url
        }

    def save(self):
        db.session.add(self)
        db.session.commit()

    def delete(self):
        db.session.delete(self)
        db.session.commit()