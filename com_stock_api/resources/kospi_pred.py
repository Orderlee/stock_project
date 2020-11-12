import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession, engine
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import and_,or_,func, extract
from pathlib import Path
from flask import jsonify
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime
from tensorflow import keras
import tensorflow as tf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD
from sklearn.metrics import accuracy_score
from keras.layers import Dense, LSTM, Dropout, GRU
from keras.layers import *
tf.compat.v1.disable_eager_execution()


from com_stock_api.resources.korea_covid import KoreaDto, KoreaCovids, KOCases, SEcases
from com_stock_api.resources.korea_finance import StockDto
from com_stock_api.resources.korea_news import NewsDto


"""
 * @ Module Name : kospi_pred.py
 * @ Description : kospi Prediction
 * @ since 2020.10.15
 * @ version 1.0
 * @ Modification Information
 * @ author 이영우
 * @ special reference libraries
 *     sqlalchemy, flask_restful
 * @ 수정일         수정자                      수정내용
 *   ------------------------------------------------------------------------
 *   2020.11.10     이영우      주식정보와 코로나 상관분석 추가
 *   2020.11.10     이영우      주식데이터 학습 추가
 *   2020.11.02     이영우      
"""



# ==============================================================
# =======================                     =======================
# =======================    DataFrame        ======================
# =======================                     =======================
# ==============================================================

class TotalDF():
    path : str
    tickers : str

    def __init__(self):
        self.path = os.path.abspath(__file__+"/.."+"/data")
        self.fileReader = FileReader()
        self.df = None
        #self.ticker = ''
        self.tickers = ['051910','011070']

    def hook(self):
        for tic in self.tickers:
            self.ticker = tic
            self.dataframe()
            self.scatter()
            self.heatmap_graph()
            
        
    '''
    columns=,date,close,open,high,low,volume,ticker
    '''
    def dataframe(self):
        main_df = self.df

        df = pd.read_sql_table('korea_finance', engine.connect())
        df = df.loc[(df['ticker'] == self.ticker) & (df['date']>'2019-12-31') & (df['date'] <'2020-07-01')]
        df['date'] = pd.to_datetime(df['date'])
        df = df.drop(['ticker','id'],axis= 1)
        df = df[df['date'].notnull() == True].set_index('date')
        # print(df)
        

        tic = [t for t in self.tickers if t != self.ticker]
        df2 = pd.read_sql_table('korea_finance',engine.connect())
        df2 = df2.loc[(df2['ticker'] == tic[0]) & (df2['date'] > '2019-12-31') & (df2['date']<'2020-07-01')]
        df2 = df2.rename(columns={'open':tic[0] + '_open', 'close':tic[0] + '_close','high':tic[0]+'_high','low':tic[0]+'_low'})
        df2 = df2.drop(['id','ticker','volume'], axis=1)
        df2 = df2[df2['date'].notnull() == True].set_index('date')



        covid_json = json.dumps(KoreaCovids.get()[0], default = lambda x: x.__dict__)
        df3 = pd.read_json(covid_json)
        df3 = df3.loc[(df3['date'] > '2019-12-31') & (df3['date']<'2020-07-01')]
        df3 = df3.drop(['id','total_cases','total_deaths','seoul_cases','seoul_deaths'],axis=1)
        df3 = df3[df3['date'].notnull() == True].set_index('date')

        
        
        main_df = df.join(df2, how='outer')
        main_df = main_df.join(df3, how='outer')
        main_df[['ko_cases','ko_deaths','se_cases','se_deaths']] = main_df[['ko_cases','ko_deaths','se_cases','se_deaths']].fillna(value=0)
        
        
        main_df = main_df.astype(float).interpolate()
        main_df = main_df.fillna(value=0)
        print(main_df)
        print(main_df.columns)


        output_file = self.ticker + '_dataset.csv'
        result = os.path.join(self.path, output_file)
        main_df.to_csv(result)
        return main_df
    
    def scatter(self):

        path = os.path.abspath(__file__+"/../"+"/image")
        company = self.ticker +"_dataset.csv"
        input_file = os.path.join(self.path, company)

        df = pd.read_csv(input_file)
        df.drop(['date'], axis=1)   

        sns.pairplot(df, height = 2.5)
        plt.tight_layout()
        plt.title("The Scatter Plot of " + self.ticker)
        file_name = self.ticker + "_correlation.png"
        output_file = os.path.join(path, file_name)
        plt.savefig(output_file)


    def heatmap_graph(self):
        
        path = os.path.abspath(__file__+"/../"+"/image")
        company = self.ticker + "_dataset.csv"
        input_file = os.path.join(self.path, company)
        
        df = pd.read_csv(input_file, header=0)
        df.drop(['date'], axis=1, inplace=True)
        #print(df.columns)

        sns.heatmap(df)
        plt.title('Heatmap of ' + self.ticker, fontsize=20)

        
        file_name2 = self.ticker + "_heatmap.png"
        output_file2 = os.path.join(path,file_name2)
        plt.savefig(output_file2)

 
# if __name__ == '__main__':
#     h = TotalDF()
#     h.hook()

# ==================================================================
# =======================                    =======================
# =======================        analsys     =======================
# =======================                    =======================
# ==================================================================


class StockService():

  
    def __init__(self):
        self.reader = FileReader()
        self.data = os.path.abspath(__file__+"/.."+"/data/")
        self.ticker=''
        self.tickers = ['051910','011070']

    
    def hook(self):
        for tic in self.tickers:
            self.ticker = tic
            self.get_data()


    
    def get_data(self):
        path = self.data
        self.reader.context = os.path.join(path)
        self.reader.fname = '/'+self.ticker+'_dataset.csv'
        df = self.reader.csv_to_dframe()
        print(df)
        #print(df.columns)

        """
        date,open,close,high,low,volume,011070_open,011070_close,011070_high,011070_low,ko_cases,ko_deaths,se_cases,se_deaths
        """

        num_shape = 120
        train = df[:num_shape][["open"]]
        test = df[num_shape:][["close"]]
       
        #print(type(df['date']))
        # print(df.columns)
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

        window  = 60

        for i in range(window, num_shape):
            try:
                X_train_ = np.reshape(train_scaled[i-window:i,0],(window,1))
                X_train.append(X_train_)
                y_train.append(train_scaled[i,0])
            except:
                pass
        
        X_train = np.stack(X_train)
        #print(X_train.shape)
        y_train = np.stack(y_train)
        #print(y_train.shape)

        

        model = tf.keras.models.Sequential()

        model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
        model.add(Dropout(0.2))
        
        model.add(LSTM(units = 50, return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50,return_sequences = True))
        model.add(Dropout(0.2))

        model.add(LSTM(units=50))
        model.add(Dropout(0.2))

        model.add(Dense(units = 1))
        model.summary()



        checkpoint_path = os.path.join(path, self.ticker+'_train', self.ticker+'.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=5)
        
        model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics = ['accuracy'])
        #model.load_weights(checkpoint_path)
        model.save_weights(checkpoint_path.format(epoch=0))
        hist = model.fit(X_train, y_train, callbacks=[cp_callback], epochs = 450, batch_size = 32) 
        model.save(os.path.join(path, self.ticker + '_pred.h5'))
        
        

        

        #print("loss:"+ str(hist.history['loss']))

        
        df_volume = np.vstack((train, test))
        # print(train.shape)
        # print(test.shape)
        # print(df_volume.shape)
        inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
        inputs = inputs.reshape(-1,1)
        inputs = sc.transform(inputs)
        num_2 = df_volume.shape[0] - num_shape + window
        
        X_test = []
        
        for i in range(window, num_2):
            X_test_ = np.reshape(inputs[i-window:i, 0], (window, 1))
            X_test.append(X_test_)
            
        X_test = np.stack(X_test)
        #print(X_test.shape)

        predict = model.predict(X_test)
        predict = sc.inverse_transform(predict)
        #df=df.sort_values(by=['date'])
        #print('======================')
        #print('[ test ] ',test.shape)
        #print('[ predict ] ',predict.shape)
        #print(df['date'][:])
        
        #print(f'type: {type(predict)}, value: {predict[:]}')
        #print(f'type: {type(test)}, value: {test[:]}')
        #print('======================')
        
        
        
        diff = predict - test.astype(float)
        
        # y_pred = predict
        # y_test_ = np.argmax(test, axis = 1)
        # print(accuracy_score(y_pred, y_test_))

        print("MSE:", np.mean(diff**2))
        print("MAE:", np.mean(abs(diff)))
        print("RMSE:", np.sqrt(np.mean(diff**2)))

        # plt.figure(figsize=(20,7))
        # plt.plot(df['date'].values[:], df_volume[:], color = 'red', label = 'Real lgchem Stock Price')
        # plt.plot(df['date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted lgchem Stock Price')
        # plt.xticks(np.arange(1000,df[:].shape[0],2000))
        # plt.title('lgchem Stock Price Prediction')
        # plt.xlabel('Date')
        # plt.ylabel('Price (₩)')
        # plt.legend()
        # plt.show()

        pred_ = predict[-1].copy()
        #print(f'type:{type(pred_)}, value:{pred_[:]}')
        prediction_full = []
        window = 60
        df_copy = df.iloc[:, 2:3][1:].values
        #print(f'type:{type(df_copy)}, value:{df_copy[:]}')

        for j in range(20):
            df_ = np.vstack((df_copy, pred_))
            train_ = df_[:num_shape]
            test_ = df_[num_shape:]
    
            df_volume_ = np.vstack((train_, test_))

            inputs_ = df_volume_[df_volume_.shape[0] - test_.shape[0] - window:]
            inputs_ = inputs_.reshape(-1,1)
            inputs_ = sc.transform(inputs_)

            X_test_2 = []

            for k in range(window, num_2):
                X_test_3 = np.reshape(inputs_[k-window:k, 0], (window, 1))
                X_test_2.append(X_test_3)

            X_test_ = np.stack(X_test_2)
            predict_ = model.predict(X_test_)
            pred_ = sc.inverse_transform(predict_)
            prediction_full.append(pred_[-1][0])
            # print(prediction_full)
            df_copy = df_[j:]

        prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))

        df_date = df[['date']]
        
        for h in range(30):
            df_date_add = pd.to_datetime(df_date['date'].iloc[-1]) + pd.DateOffset(days=1)
            df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['date'])
            df_date = df_date.append(df_date_add)
        
        df_date = df_date.reset_index(drop=True)

        plt.figure(figsize=(20,7))
        plt.plot(df['date'].values[:], df_volume[:], color = 'red', label = 'Real ' +self.ticker+' Stock Price')
        plt.plot(df_date['date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = 'Predicted ' +self.ticker +' Stock Price')
        plt.xticks(np.arange(100,df[:].shape[0],20)) #100, 20
        plt.title(self.ticker + ' Stock Price Prediction')
        plt.xlabel('date')
        plt.ylabel('Price (₩)')
        plt.legend()
        
        
        image_path = os.path.abspath(__file__+"/.."+"/image/")
        graph_image = self.ticker + "_graph.png"
        output_image= os.path.join(image_path,graph_image)
        plt.savefig(output_image) 
        #plt.show()
        

if __name__ =='__main__':
    s = StockService()
    s.hook()
    #StockService.hook()







# ==============================================================
# =======================                =======================
# =======================    Modeling    =======================
# =======================                =======================
# ==============================================================


class KospiDto(db.Model):
    __tablename__ = 'kospi_pred'
    __table_args__ = {'mysql_collate':'utf8_general_ci'}
    
    id: str = db.Column(db.Integer, primary_key = True, index = True)
    date : str = db.Column(db.DATE)
    ticker : str = db.Column(db.VARCHAR(30))
    pred_price : int = db.Column(db.VARCHAR(30))

    covid_id: str = db.Column(db.Integer, db.ForeignKey(KoreaDto.id))
    stock_id: str = db.Column(db.Integer, db.ForeignKey(StockDto.id))
    news_id: str = db.Column(db.Integer, db.ForeignKey(NewsDto.id))


    def __init__(self,id,date, covid_id,stock_id,news_id,ticker, pred_price):
        self.id = id
        self.date = date
        self.covid_id = covid_id
        self.stock_id= stock_id
        self.news_id= news_id
        self.ticker= ticker
        self.pred_price = pred_price
    
    def __repr__(self):
        return f'id={self.id},date={self.date},covid_id ={self.covid_id },stock_id={self.stock_id},news_id={self.news_id}, ticker={self.ticker},\
            pred_price={self.pred_price}'
            
    @property
    def json(self):
        return {
            'id':self.id,
            'date': self.date,
            'covid_id': self.covid_id,
            'stock_id': self.stock_id,
            'news_id': self.news_id,
            'ticker' : self.ticker,
            'pred_price' : self.pred_price
        }

class KospiVo:
    id : int = 0
    date : str =''
    ticker: str =''
    pred_price : int = 0
    covid_id : str = 0
    stock_id : str = 0
    news_id : str =  0
    open: float = 0.0
    close: float = 0.0
    high: float = 0.0
    low : float = 0.0


Session = openSession()
session= Session()


class KospiDao(KospiDto):

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")

    def bulk(self):
        path = self.data
        companys = ['lgchem','lginnotek']
        for com in companys:
            print(f'company:{com}')
            file_name = com +'.csv'
            input_file = os.path.join(path,file_name)
            df = pd.read_csv(input_file ,encoding='utf-8',dtype=str)
            df = df.iloc[84:206, : ]           
            session.bulk_insert_mappings(KospiDto, df.to_dict(orient='records'))
            session.commit()
        session.close()
    
    
    @staticmethod
    def count():
        return session.query(func.count(KospiDto.id)).one()
    
    @staticmethod
    def save(data):
        db.session.add(data)
        db.session.commit()

    @staticmethod
    def update(data):
        db.session.add(data)
        db.session.commit()

    @classmethod
    def delete(cls,id):
        data = cls.query.get(id)
        db.session.delete(data)
        db.session.commit()


    @classmethod
    def find_all(cls):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_all_by_ticker(cls, stock):
        sql = cls.query
        df = pd.read_sql(sql.statement, sql.session.bind)
        df = df[df['ticker'] == stock.ticker]
        return json.loads(df.to_json(orient='records'))

    @classmethod
    def find_by_id(cls,id):
        return session.query(KospiDto).filter(KospiDto.id.like(id)).one()
        
    @classmethod
    def find_by_date(cls, date):
        return session.query(KospiDto).filter(KospiDto.date.like(date)).one()

    @classmethod
    def find_by_predprice(cls,pred_price):
        return session.query(KospiDto).filter(KospiDao.pred_price.like(pred_price)).one()

    @classmethod
    def find_by_stockid(cls,stock_id):
        return session.query(KospiDto).filter(KospiDto.stock_id.like(stock_id)).one()

    @classmethod
    def find_by_ticker(cls,ticker):
        return session.query(KospiDto).filter(KospiDto.ticker.like(ticker)).one()

    @classmethod
    def find_by_covidid(cls,covid_id):
        return session.query(KospiDto).filter(KospiDto.covid_id.like(covid_id)).one()

    @classmethod
    def fidn_by_newsid(cls,news_id):
        return session.queryfilter(KospiDto.news_id.like(news_id)).one()


# if __name__ =='__main__':
#     #KospiDao()
#     r=KospiDao()
#     r.bulk()

# ==============================================================
# =====================                  =======================
# =====================    Resourcing    =======================
# =====================                  =======================
# ==============================================================


parser = reqparse.RequestParser()
parser.add_argument('id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('date', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('covid_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('stock_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('news_id', type=int, required=True, help='This field cannot be left blank')
parser.add_argument('ticker', type=str, required=True, help='This field cannot be left blank')
parser.add_argument('pred_price', type=int, required=True, help='This field cannot be left blank')

class Kospi(Resource):

    @staticmethod
    def post():
        data = parser.parse_args()
        kospiprediction = KospiDto(data['date'], data['ticker'],data['pred_price'], data['stock_id'], data['covid_id'], data['news_id'])
        try:
            kospiprediction.save(data)
            return {'code':0, 'message':'SUCCESS'},200
        except:
            return {'message': 'An error occured inserting the pred history'}, 500
        return kospiprediction.json(),201

    def get(self, id):
        kospiprediction = KospiDao.find_by_id(id)
        if kospiprediction:
            return kospiprediction.json()
        return {'message': 'kospiprediction not found'}, 404
    
    def put(self, id):
        data = Kospi.parser.parse_args()
        prediction = KospiDao.find_by_id(id)

        prediction.date = data['date']
        prediction.price = data['pred_price']
        prediction.save()
        return prediction.json()

class Kospis(Resource):
    def get(self):
        return  KospiDao.find_all(), 200

class lgchem_pred(Resource):
    @staticmethod
    def get():
        stock = KospiVo()
        stock.ticker='051910'
        data = KospiDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def post():
        parser.add_argument('open', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('high', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('low', type=str, required=True,
        help='This field should be a number')
        parser.add_argument('close', type=str, required=True,
        help='This field should be a number')


        args = parser.parse_args()
        stock = KospiVo()
        stock.ticker = args.ticker
        data = KospiDao.find_all_by_ticker(stock)
        return data[0], 200

class lginnotek_pred(Resource):

    @staticmethod
    def get():
        stock = KospiVo()
        stock.ticker='011070'
        data = KospiDao.find_all_by_ticker(stock)
        return data, 200

    @staticmethod
    def post():
        args = parser.parse_args()
        stock = KospiVo()
        stock.ticker = args.ticker
        data = KospiDao.find_all_by_ticker(stock)
        return data[0], 200



class LGchempred(Resource):
    open : float = 0.0
    high : float = 0.0
    low : float = 0.0
    close : float = 0.0

    def __init__(self):
        self.path = os.path.abspath(__file__+"/.."+"/data"+"/011070_train")


    def assign(self, param):
        self.open = param.open
        self.high = param.high
        self.low= param.low
        self.close = param.close
        self.ticker = param.ticker



    def predict(self):
        print('##########  Service ############')

        X = tf.compat.v1.placeholder(tf.float32, shape=[None, 4])
        W = tf.Variable(tf.random.normal([4,1]), name='weight')
        b = tf.Variable(tf.random.normal([1]), name='bias')

        saver =  tf.compat.v1.train.Saver()
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            saver.restore(sess, self.path+'/011070.ckpt')
            data = [[self.open, self.high, self.low], ]                
            arr = np.array(data, dtype=np.float32)
            dict = sess.run(tf.matmul(X,W)+b, feed_dict={X:arr[0,3]})
            print ('dict!!' , dict[0])
            print(dict[0])
        return int(dict[0])
            


# if __name__ == '__main__':
#     service = LGchempred()
#     chem = KospiVo()
#     chem.open ='500000'
#     chem.high= '550000'
#     chem.low = '490000'
#     chem.ticker = '051910'
#     service.assign(chem)
#     price = service.predict()
#     print('price:',price)
