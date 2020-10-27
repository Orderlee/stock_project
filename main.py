from flask import Flask
from flask_restful import Api
from com_stock_api.ext.db import url, db
from com_stock_api.ext.routes import initialize_routes
from com_stock_api.resource.korea_news import NewsDao
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={'r/api/*': {"origins":"*"}})

app.config['SQLALCHEMY_DATABASE_URI'] =url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db.init_app(app)
api= Api(app)


with app.app_context():
    db.create_all()
    #print(f'db created ... ')
    news_count = NewsDao.count()
    print(f'****** News Total Count is {news_count} *******')
    if news_count[0] == 0:
        #NewsDao()
        n = NewsDao()
        n.bulk()




@app.route('/api/test')
def test():
    return{'test':'SUCCESS'}

initialize_routes(api)
