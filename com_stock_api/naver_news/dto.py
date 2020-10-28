from com_stock_api.ext.db import db 

class NewsDto(db.Model):
    __tablename__ = 'naver_news'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}

    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATETIME)
    headline : str = db.Column(db.String(255))
    contents : str = db.Column(db.String(10000))
    url :str = db.Column(db.String(500))
    stock : str = db.Column(db.String(30))
    label : float = db.Column(db.Float)


    
    def __init__(self, id, date, headline, contents, url, stock, label):
        self.id = id
        self.date = date
        self.headline = headline
        self.contents = contents
        self.url = url
        self.stock = stock
        self.label = label
        
    
    def __repr__(self):
        return f'id={self.id},date={self.date}, headline={self.headline},\
            contents={self.contents},url={self.url},stock={self.stock},label={self.label}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'headline':self.headline,
            'contents':self.contents,
            'url':self.url,
            'stock':self.stock,
            'label':self.label
        }

class NewsVo:
    id : str =''
    date: str =''
    headline: str=''
    contents: str=''
    url: str =''
    stock: str =''
    label: float =''
