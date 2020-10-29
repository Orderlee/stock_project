import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
import math
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class StockDto(db.Model):
    __tablename__ = 'korea_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    open : int = db.Column(db.String(30))
    close : int = db.Column(db.String(30))
    high : int = db.Column(db.String(30))
    low :int = db.Column(db.String(30))
    volume : int = db.Column(db.String(30))
    ticker : str = db.Column(db.String(30))
  
    def __init__(self,id,date, open, close, high, low, volume, ticker):
        self.id = id
        self.date = date
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.ticker = ticker
    
    def __repr__(self):
        return f'id={self.id}, date={self.date}, open={self.open},\
            close={self.close}, high={self.high}, low={self.low}, volume={self.volume}, ticker={self.ticker}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'open': self.open,
            'close': self.close,
            'high': self.high,
            'low': self.low,
            'volume': self.volume,
            'ticker' : self.ticker
        }

class StockVo:
    id: int = 0
    date: str=''
    open: int =''
    close: int =''
    high: int =''
    low: int =''
    volume: int =''
    ticker: str=''


Session = openSession()
session= Session()


class StockDao(StockDto):

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")

    #@staticmethod
    def bulk(self):
        #service = StockService()
        #df = service.hook()
        path = self.data
        companys = ['lgchem','lginnotek']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            print(df.head())
            session.bulk_insert_mappings(StockDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(StockDto.id)).one()

    @staticmethod
    def save(stock):
        db.session.add(stock)
        db.sessio.commit()

    @staticmethod
    def update(stock):
        db.session.add(stock)
        db.session.commit()

    @classmethod
    def delete(cls,open):
        data = cls.query.get(open)
        db.session.delete(data)
        db.sessio.commit()
    
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))


    @classmethod
    def find_by_date(cls,date):
        return cls.query.filter_by(date == date).all()


    @classmethod
    def find_by_id(cls, open):
        return cls.query.filter_by(open == open).first()

    @classmethod
    def login(cls,stock):
        sql = cls.query.fillter(cls.id.like(stock.date)).fillter(cls.open.like(stock.open))
        
        df = pd.read_sql(sql.statement, sql.session.bind)
        print('----------')
        print(json.loads(df.to_json(orient='records')))
        return json.loads(df.to_json(orient='records'))
    

if __name__ == "__main__":
    #StockDao.bulk()
    # s = StockDao()
    # s.bulk()
    ...

    




# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================






parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field should be a id')
parser.add_argument('date',type=str, required=True,help='This field should be a date')
parser.add_argument('open',type=int, required=True,help='This field should be a open')
parser.add_argument('close',type=int, required=True,help='This field should be a close')
parser.add_argument('high',type=int, required=True,help='This field should be a high')
parser.add_argument('low',type=int, required=True,help='This field should be a low')
parser.add_argument('volume',type=int, required=True,help='This field should be a amount')
parser.add_argument('ticker',type=str, required=True,help='This field should be a stock')



class Stock(Resource):

    @staticmethod
    def post():
        args = parser.parse_args()
        print(f'Stock {args["id"]} added')
        params = json.loads(request.get_data(), encoding='utf-8')
        if len (params) == 0:
            return 'No parameter'

        params_str =''
        for key in params.keys():
            params_str += 'key: {}, value:{} <br>'.format(key,params[key])
        return {'code':0, 'message':'SUCCESS'}, 200
    
    @staticmethod
    def get(id):
        print(f'Stock {id} added')
        try:
            stock = StockDao.find_by_id(id)
            if stock:
                return stock.json()
        except Exception as e:
            return {'message': 'Item not found'}, 404
    
    @staticmethod
    def update():
        args = parser.arse_args()
        print(f'Stock {args["id"]} updated')
        return {'code':0, 'message': 'SUCCESS'}, 200

    
    @staticmethod
    def delete():
        args = parser.parse_args()
        print(f'Stock {args["id"]} deleted')
        return {'code':0, 'message': 'SUCCESS'}, 200

class Stocks(Resource):
    
    @staticmethod
    def get():
        sd = StockDao()
        sd.insert('korea_finance')
    
    @staticmethod
    def get():
        data = StockDao.find_all()
        return data, 200

class Auth(Resource):
    
    @staticmethod
    def post():
        body = request.get_json()
        stock = StockDto(**body)
        StockDto.save(stock)
        id = stock.id

        return {'id': str(id)}, 200

class Access(Resource):

    @staticmethod
    def post():
        args = parser.parse_argse()
        stock = StockVo()
        stock.id = args.id 
        sstock.date = args.date
        print(stock.id)
        print(stock.date)
        data = StockDao.login(stock)
        return data[0], 200


# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================
# ==============================================================



import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class Graph():

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
        
    def closeprice(self):
        
        path = self.data
        companys = ['lgchem']#['lgchem','lginnotek']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            df = pd.DataFrame(df)
        #print(df.columns)

        from bokeh.io import show, output_file
        from bokeh.plotting import figure, show, output_notebook
        from bokeh.layouts import gridplot
        from bokeh.models.formatters import NumeralTickFormatter

        output_notebook()
        inc = df.close >= df.open
        dec = df.open > df.close

        # 캔들 만들기
        p_candlechart = figure(plot_width=1050, plot_height=300, x_range=(-1, len(df)), tools=['xpan, crosshair, xwheel_zoom, reset, hover, box_select, save'])
        p_candlechart.segment(df.index[inc], df.high[inc], df.index[inc], df.low[inc], color="green") #서양은 양봉을 초록 음봉을 빨강
        p_candlechart.segment(df.index[dec], df.high[dec], df.index[dec], df.low[dec], color="red") #한국은 양봉을 빨간 음봉을 파랑
        p_candlechart.vbar(df.index[inc], 0.5, df.open[inc], df.close[inc], fill_color="green", line_color="green")
        p_candlechart.vbar(df.index[dec], 0.5, df.open[dec], df.close[dec], fill_color="red", line_color="red")

        p_volumechart = figure(plot_width=1050, plot_height=200, x_range=p_candlechart.x_range, tools="crosshair")
        p_volumechart.vbar(df.index, 0.5, df.volume, fill_color="white", line_color="white")
        p_volumechart.background_fill_color = '#121212'

        # 날짜 표시를 위한 작업
        major_label = {i: date.strftime('%Y-%m-%d') for i, date in enumerate(pd.to_datetime(df["date"]))}
        major_label.update({len(df): ''})
        p_volumechart.xaxis.major_label_overrides = major_label
        p_volumechart.yaxis[0].formatter = NumeralTickFormatter(format='0,0')


        # 그리기
        p = gridplot([[p_candlechart],[p_volumechart]], toolbar_location=None)
        show(p)


        
        

        
        # 반응형 차트 그리는 코드
        #fig = px.line(df, x='date', y='close', title='{}의 종가(close) Time Series'.format(com))
        # fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,row_heights = [0.7, 0.3],)
        
        # fig.add_trace(go.Line(x=df["date"], y=df["close"], name="거래금액"), row=1, col=1)
        # fig.add_trace(go.Bar(x=df["date"], y=df["volume"], name="거래량"), row=2, col=1)
        # fig.update_layout(height=300*2, width=350*2,title_text="{}의 종가(close)".format(com))

        # fig.update_xaxes(
            
        #     rangeselector=dict(
        #         buttons=list([
        #             dict(count=1, label="1d", step="day", stepmode="backward"),
        #             dict(count=3, label="3d", step="day", stepmode="backward"),
        #             dict(count=6, label="6d", step="day", stepmode="backward"),
        #             dict(count=1, label="1m", step="month", stepmode="backward"),
        #             dict(count=3, label="3m", step="month", stepmode="backward"),
        #             dict(count=6, label="6m", step="month", stepmode="backward"),
        #             dict(step="all")
        #         ])
        #     )
        # )
        # fig.show()




        # plt.figure(figsize=(10,4))
        # plt.plot(df['date'], df['close'])
        # plt.xlabel('')
        # plt.ylabel('close')
        # plt.tick_params(
        #     axis='x',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False) # labels along the bottom edge are off
        # #plt.savefig(company + ".png")
        # plt.show()

if __name__ == "__main__":
    g=Graph()
    g.closeprice()






# class StockService:

#     x_train: object = None
#     y_train: object = None
#     x_validation: object = None
#     y_validation: object = None
#     x_test: object = None
#     y_test: object = None
#     model: object = None

#     def __init__(self):
#         self.reader = FileReader()

#     def hook(self):
#         self.get_data()
#         self.create_model()
#         self.train_model()
#         self.eval_model()
#         # self.debug_model()
#         # self.get_prob()

#     def create_train(self, this):
#         pass

#     def create_label(self, this):
#         pass

#     def get_data(self):
#         self.reader.context = os.path.join(basedir, 'data/')
#         self.reader.fname = 'lgchem.csv'
#         data = self.reader.csv_to_dframe()
#         #print(data)
#         #(890,7)
#         #data = data.to_numpy()
#         # date,close,open,high,low,volume,stock
#         #"date",
#         xdata = data[["open","high","low","volume"]]
#         #print(xdata)
#         ydata = pd.DataFrame(data["close"])
#         #print(ydata)

#         xdata_ss = StandardScaler().fit_transform(xdata)
#         #print(xdata_ss)
#         ydata_ss = StandardScaler(). fit_transform(ydata)
#         #print(ydata_ss)

        

#         x_train, x_test, y_train, y_test = train_test_split(xdata_ss,ydata_ss, test_size=0.4)
#         x_test, x_validation, y_test, y_validation = train_test_split(x_test,y_test, test_size=0.4)

#         self.x_train = x_train; self.x_test = x_test; self.x_validation = x_validation
#         self.y_train = y_train;  self.y_test = y_test; self.y_validation = y_validation
        
#         print(self.x_train.shape)
#         print(self.y_test.shape)

#     def create_model(self):
#         print('create model')
#         model = keras.Sequential()

#         #LSTM으로 바꿔야함
#         model.add(layers.Dense(units=1024, input_dim=4, activation='relu'))
#         model.add(layers.Dense(units=512, activation='relu'))
#         model.add(layers.Dense(units=256, activation='relu'))
#         model.add(layers.Dense(units=128, activation='relu'))
#         model.add(layers.Dense(units=64, activation='relu'))
#         model.add(layers.Dense(units=32, activation='relu'))
#         model.add(layers.Dense(units=1))

#         model.compile(loss='mse', optimizer='adam', metrics=['mae'])

#         self.model = model
    
#     def train_model(self):
#         es = EarlyStopping(patience=10)

#         # seed = 123
#         # np.random.seed(seed)
#         # tf.set_random_seed(seed)
#         hist = self.model.fit(self.x_train, self.y_train,
#         validation_data=(self.x_validation, self.y_validation), epochs=50, batch_size=16, callbacks=[es])
#         print("loss:"+ str(hist.history['loss']))
#         print("MAE:"+ str(hist.history['mae']))


#     def eval_model(self):
#         res = self.model.evaluate(self.x_test, self.y_test, batch_size=32)
#         print('loss',res[0],'mae',res[1])
        
#         xhat = self.x_test
#         yhat = self.model.predict(xhat)

#         plt.figure()
#         plt.plot(yhat, label = "predicted")
#         plt.plot(self.y_test,label = "actual")
#         #xlabel=['date']

#         plt.legend(prop={'size': 20})
#         plt.show()

#         print("Evaluate : {}".format(np.average((yhat - self.y_test)**2)))

 
        
        
# if __name__ =='__main__':
#     training = StockService()
#     training.hook()