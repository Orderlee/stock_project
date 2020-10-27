import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
basedir = os.path.dirname(os.path.abspath(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from com_stock_api.util.file_reader import FileReader
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential 
from keras.layers import Dense, Activation
from tensorflow.keras import layers
from tensorflow.keras.layers import LSTM

import tensorflow_datasets as tfds
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
#from sklearn.linear_model import LogisticRegression

class StockService:

    x_train: object = None
    y_train: object = None
    x_validation: object = None
    y_validation: object = None
    x_test: object = None
    y_test: object = None
    model: object = None

    def __init__(self):
        self.reader = FileReader()

    def hook(self):
        self.get_data()
        self.create_model()
        self.train_model()
        self.eval_model()
        # self.debug_model()
        # self.get_prob()

    def create_train(self, this):
        pass

    def create_label(self, this):
        pass

    def get_data(self):
        self.reader.context = os.path.join(basedir, 'data/')
        self.reader.fname = 'lgchem.csv'
        data = self.reader.csv_to_dframe()
        #print(data)
        #(890,7)
        #data = data.to_numpy()
        # date,close,open,high,low,volume,stock
        #"date",
        xdata = data[["open","high","low","volume"]]
        #print(xdata)
        ydata = pd.DataFrame(data["close"])
        #print(ydata)

        xdata_ss = StandardScaler().fit_transform(xdata)
        #print(xdata_ss)
        ydata_ss = StandardScaler(). fit_transform(ydata)
        #print(ydata_ss)

        

        x_train, x_test, y_train, y_test = train_test_split(xdata_ss,ydata_ss, test_size=0.4)
        x_test, x_validation, y_test, y_validation = train_test_split(x_test,y_test, test_size=0.4)

        self.x_train = x_train; self.x_test = x_test; self.x_validation = x_validation
        self.y_train = y_train;  self.y_test = y_test; self.y_validation = y_validation
        
        print(self.x_train.shape)
        print(self.y_test.shape)

    def create_model(self):
        print('create model')
        model = keras.Sequential()

        #LSTM으로 바꿔야함
        model.add(layers.Dense(units=1024, input_dim=4, activation='relu'))
        model.add(layers.Dense(units=512, activation='relu'))
        model.add(layers.Dense(units=256, activation='relu'))
        model.add(layers.Dense(units=128, activation='relu'))
        model.add(layers.Dense(units=64, activation='relu'))
        model.add(layers.Dense(units=32, activation='relu'))
        model.add(layers.Dense(units=1))

        model.compile(loss='mse', optimizer='adam', metrics=['mae'])

        self.model = model
    
    def train_model(self):
        es = EarlyStopping(patience=10)

        # seed = 123
        # np.random.seed(seed)
        # tf.set_random_seed(seed)
        hist = self.model.fit(self.x_train, self.y_train,
        validation_data=(self.x_validation, self.y_validation), epochs=50, batch_size=16, callbacks=[es])
        print("loss:"+ str(hist.history['loss']))
        print("MAE:"+ str(hist.history['mae']))


    def eval_model(self):
        res = self.model.evaluate(self.x_test, self.y_test, batch_size=32)
        print('loss',res[0],'mae',res[1])
        
        xhat = self.x_test
        yhat = self.model.predict(xhat)

        plt.figure()
        plt.plot(yhat, label = "predicted")
        plt.plot(self.y_test,label = "actual")
        #xlabel=['date']

        plt.legend(prop={'size': 20})
        plt.show()

        print("Evaluate : {}".format(np.average((yhat - self.y_test)**2)))

 
        
        
if __name__ =='__main__':
    training = StockService()
    training.hook()

