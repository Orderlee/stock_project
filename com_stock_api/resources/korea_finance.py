# -*- coding: utf-8 -*- 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
from pandas.io import json
import json
import datetime
from sqlalchemy.dialects.mysql import DATE,DATETIME
import time
import random


# ==============================================================
# =========================                =====================
# =========================  Data Mining   =====================
# =========================                =====================
# ==============================================================

class KoreaStock():
    
    def __init__(self):
        self.stock_code = None

    def new_model(self):
        stock_code = pd.read_html('http://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13',
                       header=0)[0]
        stock_code.종목코드=stock_code.종목코드.map('{:06d}'.format)
        stock_code=stock_code[['회사명','종목코드']]

        stock_code=stock_code.rename(columns={'회사명':'company','종목코드':'code'})
        
        #code_df.head()
        self.stock_code = stock_code
    
    def search_stock(self,company):
        result=[]

        stock_code = self.stock_code
        plusUrl = company.upper()
        plusUrl = stock_code[stock_code.company==plusUrl].code.values[0].strip()
        
        for i in range(1,100):
            try:
                url='https://finance.naver.com/item/sise_day.nhn?code='+str(plusUrl)+'&page={}'.format(i)
            except: 
                pass
            response=requests.get(url)
            text=response.text
            html=BeautifulSoup(text,'html.parser')
            table0=html.find_all("tr",{"onmouseover":"mouseOver(this)"})
            #print(url)
            
            def refine_price(text):
                price=int(text.replace(",",""))
                return price

            
            for tr in table0:
                date= tr.find_all('td')[0].text
                temp=[]  
                
                for idx,td in enumerate(tr.find_all('td')[1:]):
                    if idx==1:
                        try:
                            #print(td.find()['alt'])
                            temp.append(td.find()['alt']) 
                        except: 
                            temp.append('')
                        
                    price=refine_price(td.text)
                    #print(price)
                    temp.append(price)
                
                #print([date]+temp)
                result.append([date]+temp)

                df_result=pd.DataFrame(result,columns=['date','close','up/down','pastday','open','high','low','volume'])
                df_result['ticker']=plusUrl 
                df_result.drop(['up/down', 'pastday'], axis='columns', inplace=True)
                #df_result['date']=pd.to_datetime(df_result['date'].astype(str), format='%Y/%m/%d')
                #df_result.set_index('date', inplace=True)
                #time.sleep( random.uniform(2,4) )
        return df_result
                



# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================
    
class StockDto(db.Model):
    
    __tablename__ = 'korea_finance'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: int = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.String(100))
    open : int = db.Column(db.String(30))
    close : int = db.Column(db.String(30))
    high : int = db.Column(db.String(30))
    low :int = db.Column(db.String(30))
    volume : int = db.Column(db.String(30))
    ticker : str = db.Column(db.String(30))

    def __init__(self,id, date, open, close, high, low, volume, ticker):
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
    date: str= ''
    open: int =0
    close: int =0
    high: int =0
    low: int =0
    volume: int =0
    ticker: str=''


Session = openSession()
session= Session()



class StockDao(StockDto):
    
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")


    def bulk(self): 
        path = self.data
        # krs = KoreaStock()
        # krs.new_model()
        companys = ['lg화학','lg이노텍']
        
        for com in companys:
            print(f'company:{com}')          
            # df = krs.search_stock(com)
            if com =='lg화학':
                com ='lgchem'
            elif com =='lg이노텍':
                com='lginnotek'
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            # df.to_csv(path + '/'+com+'.csv')
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            print(df.head())            
            session.bulk_insert_mappings(StockDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    @staticmethod
    def count():
        return session.query(func.count(StockDto.id)).one()

    @staticmethod
    def save(data):
        db.session.add(data)
        db.sessio.commit()

    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()

    @classmethod
    def delete(cls,id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.sessio.commit()
    
    
    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker']== stock.ticker]
        return json.loads(df.to_json(orient='records'))


        


    @classmethod
    def find_by_date(cls,date):
        return session.query.filter(StockDto.date.like(date)).one()

    @classmethod
    def find_by_ticker(cls, tic):
        return session.query(StockDao).filter((StockDto.ticker.like(tic))).order_by(StockDto.date).all()



# if __name__ =='__main__':
#     r=StockDao()
#     r.bulk()



# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================






parser = reqparse.RequestParser()
parser.add_argument('id',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('date',type=str, required=True,help='This field cannot be left blank')
parser.add_argument('open',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('close',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('high',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('low',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('volume',type=int, required=True,help='This field cannot be left blank')
parser.add_argument('ticker',type=str, required=True,help='This field cannot be left blank')



class Stock(Resource):

    @staticmethod
    def post(self):
        data = self.parset.parse_args()
        
        stock = StockDto(data['date'], data['ticker'],data['open'], data['high'], data['low'], data['close'],  data['adjclose'], data['volume'])  
        try: 
            stock.save(data)
            return {'code' : 0, 'message' : 'SUCCESS'}, 200

        except:
            return {'message': 'An error occured inserting the stock history'}, 500
        return stock.json(), 201
    

    def get(self,ticker):
        stock = StockDao.find_by_ticker(ticker)
        if stock:
            return stock.json()
        return {'message': 'The stock was not found'}, 404


    def put(self,id):
        data = Stock.parser.parse_args()
        stock = StockDao.find_by_id(id)

        stock.date = data['date']
        stock.close = data['close']
        stock.save()
        return stock.json()
    
    @staticmethod
    def delete():
        args = parser.parse_arges()
        print(f'Ticker {args["ticker"]} on date {args["date"]} is deleted')
        StockDao.delete(args['id'])
        return {'code':0, 'message':'SUCCESS'}, 200


class Stocks(Resource):
    def get(self):
        return StockDao.find_all(), 200
    

class lgchem(Resource):
    
    @staticmethod
    def get():
        print("lgchem get")
        stock = StockVo()
        stock.ticker = '051910'
        data = StockDao.find_all_by_ticker(stock)
        return data, 200

    
    @staticmethod
    def post():
        print("lgchem post")
        args = parser.parse_args()
        stock = StockVo()
        stock.ticker = args.ticker
        data = StockDao.find_all_by_ticker(stock)
        #return data.json(), 200
        return data[0], 200

class lginnotek(Resource):

    @staticmethod
    def get():
        print("lginnotek get")
        stock = StockVo()
        stock.ticker = '011070'
        data = StockDao.find_all_by_ticker(stock)
        return data, 200
 

    @staticmethod
    def post():
        print("lginnotek post")
        args = parser.parse_args()
        stock = StockVo()
        stock.ticker = args.ticker
        data = StockDao.find_all_by_ticker(stock)
        #return data.json(), 200
        return data[0], 200



import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
import numpy as np 

class StockService:

    def __init__(self):
        self.reader = FileReader()
        self.data = os.path.abspath(__file__+"/.."+"/data/")

    
    def hook(self):
        self.get_data()
        # self.dataprocessing()
        # self.create_model()
        # self.train_model()
        #self.eval_model()
    
    def get_data(self):
        path = self.data
        self.reader.context = os.path.join(path,)
        self.reader.fname = '/lgchem.csv'
        df = self.reader.csv_to_dframe()
        df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d') #.astype(str) 'datetime64[D]' 'datetime64[ns]'
        # print(type(df['date']))
        # df.shape
        # def to_datetime(df):
        #     date = datetime.strptime(df, '%d.%m.%Y')
        #     return date.strftime("%Y-%m-%d")
        
        # df['date'] = df['date'].apply(lambda x: df(x))
        
        # print(df.columns)
        num_shape = 700
        train = df.iloc[:num_shape, 2:3].values
        test = df.iloc[num_shape:, 2:3].values
        # print(df.shape)
        #train = df[["close"]]
        print(train.shape)
        #test = pd.DataFrame(df["close"])
        print('test:',test.shape)

        sc = MinMaxScaler(feature_range = (0,1))
        train_scaled = sc.fit_transform(train)

        X_train = []
        
        #price on next day
        y_train = []

        window = 60

        for i in range(window, num_shape):
            try:
                X_train_ = np.reshape(train_scaled[i-window:i,0],(window,1))
                X_train.append(X_train_)
                y_train.append(train_scaled[i,0])
            except:
                pass
        
        X_train = np.stack(X_train)
        print(X_train.shape)
        y_train = np.stack(y_train)
        print(y_train.shape)
        #print(self.X_train.shape)
        #print(self.y_train.shape)
        
        model = Sequential()

        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units = 1))
        model.summary()

        #self.model = model

        
        model.compile(optimizer = 'adam', loss = 'mean_squared_error')
        hist = model.fit(X_train, y_train, epochs = 1, batch_size = 32)

        #print("loss:"+ str(hist.history['loss']))

        
        df_volume = np.vstack((train, test))
        print(train.shape)
        print(test.shape)
        print(df_volume.shape)
        inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        num_2 = df_volume.shape[0] - num_shape + window
        
        X_test = []
        
        for i in range(window, num_2):
            X_test_ = np.reshape(inputs[i-window:i, 0], (window, 1))
            X_test.append(X_test_)
            
        X_test = np.stack(X_test)
        print(X_test.shape)

        predict = model.predict(X_test)
        predict = sc.inverse_transform(predict)
        print('======================')
        print('[ test ] ',test.shape)
        print('[ predict ] ',predict.shape)
        
        
        print(f'type: {type(predict)}, value: {predict[50]}')
        print(f'type: {type(test)}, value: {test[50]}')
        print('======================')
        '''
        [ test ]  (290, 1)
        [ predict ]  (290, 1)
        type: <class 'numpy.ndarray'>, value: [1.5268588e+18]
        type: <class 'numpy.ndarray'>, value: ['2017-12-28T00:00:00.000000000']
        test 가 날짜이기 때문에 연산이 안되네요...
        '''
        
        
        diff = predict - test

        print("MSE:", np.mean(diff**2))
        print("MAE:", np.mean(abs(diff)))
        print("RMSE:", np.sqrt(np.mean(diff**2)))

        plt.figure(figsize=(20,7))
        plt.plot(df['date'].values[700:], df_volume[700:], color = 'red', label = 'Real lgchem Stock Price')
        plt.plot(df['date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted lgchem Stock Price')
        plt.xticks(np.arange(100,df[700:].shape[0],200))
        plt.title('lgchem Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (₩)')
        plt.legend()
        plt.show()

        # pred_ = predict[-1].copy()
        # prediction_full = []
        # window = 60
        # df_copy = df.iloc[:, 1:2][1:].values

        # for j in range(20):
        #     df_ = np.vstack((df_copy, pred_))
        #     train_ = df_[:num_shape]
        #     test_ = df_[num_shape:]
    
        # df_volume_ = np.vstack((train_, test_))

        # inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
        # inputs_ = inputs_.reshape(-1,1)
        # inputs_ = sc.transform(inputs_)

        # X_test_2 = []

        # for k in range(window, num_2):
        #     X_test_3 = np.reshape(inputs_[k-window:k, 0], (window, 1))
        #     X_test_2.append(X_test_3)

        # X_test_ = np.stack(X_test_2)
        # predict_ = model.predict(X_test_)
        # pred_ = sc.inverse_transform(predict_)
        # prediction_full.append(pred_[-1][0])
        # df_copy = df_[j:]

        # prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))

        # df_date = df[['Date']]
        
        # for h in range(20):
        #     df_date_add = pd.to_datetime(df_date['Date'].iloc[-1]) + pd.DateOffset(days=1)
        #     df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['Date'])
        #     df_date = df_date.append(df_date_add)
        
        # df_date = df_date.reset_index(drop=True)

        # plt.figure(figsize=(20,7))
        # plt.plot(df['Date'].values[1700:], df_volume[1700:], color = 'red', label = 'Real Tesla Stock Price')
        # plt.plot(df_date['Date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = 'Predicted Tesla Stock Price')
        # plt.xticks(np.arange(100,df[1700:].shape[0],200))
        # plt.title('Tesla Stock Price Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Price ($)')
        # plt.legend()
        # plt.show()

if __name__ =='__main__':
    s = StockService()
    s.hook()
    #StockService.hook()