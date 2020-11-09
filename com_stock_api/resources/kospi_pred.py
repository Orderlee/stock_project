import os
from typing import List
from flask import request
from flask_restful import Resource, reqparse
from com_stock_api.ext.db import db, openSession
from com_stock_api.utils.file_helper import FileReader
from sqlalchemy import func
from pathlib import Path
from flask import jsonify
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from com_stock_api.resources.korea_covid import KoreaDto
from com_stock_api.resources.korea_finance import StockDto
from com_stock_api.resources.korea_news import NewsDto






# ==============================================================
# =======================                =======================
# =======================    data merge   ======================
# =======================                =======================
# ==============================================================


class Covidedit():
    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")
    
    def kor_csv(self):
        companys = ['lgchem','lginnotek']
        
        path = self.data
        df_covid = pd.read_csv(path +'/kor&seoul.csv')
        
        
        for com in companys:
            print(f'company:{com}')
            path = self.data
            df_stock = pd.read_csv(path +'/'+com+'.csv',dtype=str)
            df_news = pd.read_csv(path +'/'+com+'_label.csv')    

        
            # print(df_covid.columns)
            # print(df_stock.columns)
            # print(df_news.columns)
            df_stock.drop(['Unnamed: 0','high','low'], axis=1 ,inplace = True)
            
            df_stock['date']=pd.to_datetime(df_stock['date']).apply(lambda x:x.strftime('%Y-%m-%d'))
            df_covid['date']=pd.to_datetime(df_covid['date']).apply(lambda x:x.strftime('%Y-%m-%d'))
            df_news['date']=pd.to_datetime(df_news['date']).apply(lambda x:x.strftime('%Y-%m-%d'))
            #print(df_news)
            
            df_1 = pd.merge(df_covid,df_stock, on='date',how='left')
            #print(df_1)
            df_2 = pd.merge(df_1,df_news, on='date',how='left')

            df_kor = df_2.fillna(0)
            del df_kor['id']
            df_kor.set_index('date', inplace=True)
            
            df_kor.to_csv(path + '/'+ com + '_total.csv',encoding='UTF-8')
            
            # print(df_all)

        return com
        
        
    def usa_csv(self):
        america = ['TSLA','AAPL']
        path = self.data
        df_usa_covid = pd.read_csv(path +'/covid.csv')
        for coms in america:
            print(f'company:{coms}')
            df_usa_stock = pd.read_csv(path+'/'+coms+'.csv')
            df_usa_news = pd.read_csv(path+'/'+coms+'_sentiment.csv')

            df_usa_stock.drop(['high','low','adjclose'],axis=1, inplace = True)
            df_usa_news.drop(['link','headline','ticker'], axis=1, inplace=True)

            df_1 = pd.merge(df_usa_covid,df_usa_stock, on='date',how='left')
            df_usa = pd.merge(df_1,df_usa_news, on='date',how='left')
            df_usa.drop(['Unnamed: 0_x','Unnamed: 0_y'],axis=1, inplace=True)

            df_usa = df_usa.fillna(0)
            df_usa.to_csv(path+'/'+coms +'_total.csv',encoding='utf-8')
            
        
        return coms

    def bulk(self):
        path = self.data
        self.kor_csv()
        self.usa_csv()
        df_lgchem = pd.read_csv(path+'/lgchem_total.csv',dtype=str)
        df_lginnotek = pd.read_csv(path+'/lginnotek_total.csv',dtype=str)
        df_kor = df_lgchem.append(df_lginnotek,ignore_index = True)
        df_kor.columns=['date', 'kor_cases', 'kor_deaths', 'seoul_cases', 'seoul_deaths','kor_close', 'kor_open', 'kor_volume', 'kor_ticker', 'label']

        df_kor.to_csv(path+'/kor_total.csv',encoding='utf-8')
        
        df_apple = pd.read_csv(path+'/AAPL_total.csv')
        df_tesla = pd.read_csv(path+'/TSLA_total.csv')
        df_usa = df_apple.append(df_tesla,ignore_index = True)
        df_usa.to_csv(path+'/usa_total.csv',encoding='utf-8')
        df_usa['date']=pd.to_datetime(df_usa['date']).apply(lambda x:x.strftime('%Y-%m-%d'))
        


        
        df_total = pd.merge(df_usa,df_kor, on='date',how='left')
        del df_total['Unnamed: 0']
        df_total = df_total.fillna(0)
        
        df_total.to_csv(path+'/_total.csv',encoding='utf-8')



# if __name__ == '__main__':
#     #Covidedit()
#     c=Covidedit()
#     c.bulk()


# ==================================================================i
# =======================                    =======================
# =======================        analsys     =======================
# =======================                    =======================
# ==================================================================

class Analsys():

    def __init__(self):
        self.data = os.path.abspath(__file__+"/.."+"/data/")

    def bargraph(self):
        path = self.data
        df = pd.read_csv(path +'/_total.csv', encoding='utf-8')
        #print(df.columns)
        del df['Unnamed: 0']
        #print(df.head())
        #corr = df.corr(method = 'pearson')
       
        #df_heatmap = sns.heatmap(corr, cbar = True, annot = True, annot_kws={'size' : 20}, fmt = '.2f', square = True, cmap = 'Blues')
        
        sns.pairplot(data = df)
        plt.show()
        # ax = sns.heatmap(df)
        # plt.title('Heatmap of Flight by seaborn', fontsize=20)
        # plt.show() 



if __name__ == '__main__':
    #Covidedit()
    c=Analsys()
    c.bargraph()


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
        self.reader.context = os.path.join(path)
        self.reader.fname = '/lgchem.csv'
        df = self.reader.csv_to_dframe()
        df['date'] = pd.to_datetime(df['date'].astype(str), format='%Y-%m-%d') # 'datetime64[D]' 'datetime64[ns]'
        # print(type(df['date']))
  
        
        # print(df.columns)
        num_shape = 900
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
        hist = model.fit(X_train, y_train, epochs = 100, batch_size = 32)

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
        #df=df.sort_values(by=['date'])
        print('======================')
        print('[ test ] ',test.shape)
        print('[ predict ] ',predict.shape)
        print(df['date'][:])
        
        # print(f'type: {type(predict)}, value: {predict[:]}')
        # print(f'type: {type(test)}, value: {test[:]}')
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
        plt.plot(df['date'].values[:], df_volume[:], color = 'red', label = 'Real lgchem Stock Price')
        plt.plot(df['date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted lgchem Stock Price')
        plt.xticks(np.arange(1000,df[:].shape[0],2000))
        plt.title('lgchem Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (₩)')
        plt.legend()
        plt.show()

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
            df_copy = df_[j:]

        prediction_full_new = np.vstack((predict, np.array(prediction_full).reshape(-1,1)))

        df_date = df[['date']]
        
        for h in range(30):
            df_date_add = pd.to_datetime(df_date['date'].iloc[-1]) + pd.DateOffset(days=1)
            df_date_add = pd.DataFrame([df_date_add.strftime("%Y-%m-%d")], columns=['date'])
            df_date = df_date.append(df_date_add)
        
        df_date = df_date.reset_index(drop=True)

        # plt.figure(figsize=(20,7))
        # plt.plot(df['date'].values[:], df_volume[:], color = 'red', label = 'Real lgchem Stock Price')
        # plt.plot(df_date['date'][-prediction_full_new.shape[0]:].values, prediction_full_new, color = 'blue', label = 'Predicted lgchem Stock Price')
        # plt.xticks(np.arange(1000,df[:].shape[0],1500))
        # plt.title('Tesla Stock Price Prediction')
        # plt.xlabel('date')
        # plt.ylabel('Price (₩)')
        # plt.legend() 
        # plt.show()

# if __name__ =='__main__':
#     s = StockService()
#     s.hook()
    #StockService.hook()



# import tensorflow.compat.v1 as tf
# from config import basedir

# class CalculatorModel:

#     def __init__(self):
#         self.model = os.path.join(basedir,'model')
#         self.data = os.path.join(self.model, 'data')
    
#     def create_add_model(self):
#         w1 = tf.placeholder(tf.float32, name='w1')
#         w2 = tf.placeholder(tf.float32, name='w2')
#         feed_dict = {'w1':8.0, 'w2':2.0}
#         r = tf.add(w1,w2, name='op_add')
#         sess = tf.Session()
#         _ = tf.Variable(initial_value = 'fake_variable')
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         print(f"feed_dict['w1'] : {feed_dict['w1']}" )
#         print(f"feed_dict['w2'] : {feed_dict['w2']}" )
#         result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
#         print(f'TF 덧셈 결과 {result}')
#         saver.save(sess, os.path.join(self.model, 'calculator_add','model'), global_step=1000)

#     def create_sub_model(self):
#         w1 = tf.placeholder(tf.float32, name='w1')
#         w2 = tf.placeholder(tf.float32, name='w2')
#         feed_dict = {'w1':8.0, 'w2':2.0}
#         r = tf.subtract(w1,w2, name='op_sub')
#         sess = tf.Session()
#         _ = tf.Variable(initial_value = 'fake_variable')
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         print(f"feed_dict['w1'] : {feed_dict['w1']}" )
#         print(f"feed_dict['w2'] : {feed_dict['w2']}" )
#         result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
#         print(f'TF 뺄셈 결과 {result}')
#         saver.save(sess, os.path.join(self.model, 'calculator_sub','model'), global_step=1000)

#     def create_div_model(self):
#         w1 = tf.placeholder(tf.float32, name='w1')
#         w2 = tf.placeholder(tf.float32, name='w2')
#         feed_dict = {'w1':8.0, 'w2':2.0}
#         r = tf.math.floordiv(w1,w2, name='op_div')
#         sess = tf.Session()
#         _ = tf.Variable(initial_value = 'fake_variable')
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         print(f"feed_dict['w1'] : {feed_dict['w1']}" )
#         print(f"feed_dict['w2'] : {feed_dict['w2']}" )
#         result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
#         print(f'TF 나눗셈 결과 {result}')
#         saver.save(sess, os.path.join(self.model, 'calculator_div','model'), global_step=1000)
    
#     def create_mul_model(self):
#         w1 = tf.placeholder(tf.float32, name='w1')
#         w2 = tf.placeholder(tf.float32, name='w2')
#         feed_dict = {'w1':8.0, 'w2':2.0}
#         r = tf.multiply(w1,w2, name='op_mul')
#         sess = tf.Session()
#         _ = tf.Variable(initial_value = 'fake_variable')
#         sess.run(tf.global_variables_initializer())
#         saver = tf.train.Saver()
#         print(f"feed_dict['w1'] : {feed_dict['w1']}" )
#         print(f"feed_dict['w2'] : {feed_dict['w2']}" )
#         result = sess.run(r, {w1: feed_dict['w1'], w2: feed_dict['w2']})
#         print(f'TF 곱셈 결과 {result}')
#         saver.save(sess, os.path.join(self.model, 'calculator_mul','model'), global_step=1000)


# if __name__=='__main__':
#     print(f'{basedir}')
#     c = CalculatorModel()









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



        
