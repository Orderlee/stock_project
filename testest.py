# import os
# import numpy as np 
# import pandas as pd 
# import matplotlib.pyplot as plt
# from datetime import datetime
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout, GRU
# from keras.layers import *
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# from sklearn.model_selection import train_test_split
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam, SGD

# class StockService:

#     def __init__(self):
#         self.reader = FileReader()
#         self.data = os.path.abspath(__file__+"/.."+"/data/")

    
#     def hook(self):
#         self.get_data()
#         # self.dataprocessing()
#         # self.create_model()
#         # self.train_model()
#         #self.eval_model()
    
#     def get_data(self):
#         path = self.data
#         self.reader.context = os.path.join(path,)
#         self.reader.fname = '/lgchem.csv'
#         df = self.reader.csv_to_dframe()
#         df['date']=pd.to_datetime(df['date'].astype(str), format='%Y/%m/%d')
#         df.shape
#         # def to_datetime(df):
#         #     date = datetime.strptime(df, '%d.%m.%Y')
#         #     return date.strftime("%Y-%m-%d")
        
#         # df['date'] = df['date'].apply(lambda x: df(x))
        
#         num_shape = 990
#         # train = df.iloc[:num_shape, 1:2].values
#         # test = df.iloc[num_shape:, 1:2].values
#         print(df.shape)
#         train = df[["close"]]
#         print(train.shape)
#         test = pd.DataFrame(df["close"])
#         print(test.shape)

#         sc = MinMaxScaler(feature_range = (0,1))
#         train_scaled = sc.fit_transform(train)

#         X_train = []
        
#         #price on next day
#         y_train = []

#         window = 60

#         for i in range(window, num_shape):
#             try:
#                 X_train_ = np.reshape(train_scaled[i-window:i,0],(window,1))
#                 #print( X_train_.shape)
#                 X_train.append(X_train_)
#                 y_train.append(train_scaled[i,0])
#             except:
#                 pass
        
#         self.X_train = X_train = np.stack(X_train)
#         print(X_train.shape)
#         self.y_train = y_train = np.stack(y_train)
#         print(y_train.shape)
#         #print(self.X_train.shape)
#         #print(self.y_train.shape)
        
#         model = Sequential()

#         model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
#         model.add(Dropout(0.2))
        
#         model.add(LSTM(units = 50, return_sequences = True))
#         model.add(Dropout(0.2))

#         model.add(LSTM(units=50, return_sequences = True))
#         model.add(Dropout(0.2))

#         model.add(LSTM(units=50))
#         model.add(Dropout(0.2))

#         model.add(Dense(units = 1))
#         model.summary()

#         self.model = model

        
#         model.compile(optimizer = 'adam', loss = 'mean_squared_error')
#         hist = self.model.fit(self.X_train, self.y_train, epochs = 1, batch_size = 32)

#         #print("loss:"+ str(hist.history['loss']))

#         print(train.shape)
#         print(test.shape)
#         df_volume = np.vstack((train, test))
#         print(df_volume.shape)
#         inputs = df_volume[df_volume.shape[0] - test.shape[0] - window:]
#         inputs = inputs.reshape(-1,1)
#         inputs = sc.transform(inputs)
#         num_2 = df_volume.shape[0] - num_shape + window
        
#         X_test = []
        
#         for i in range(window, num_2):
#             X_test_ = np.reshape(inputs[i-window:i, 0], (window, 1))
#             X_test.append(X_test_)
            
#         X_test = np.stack(X_test)

#         predict = model.predict(X_test)
#         predict = sc.inverse_transform(predict)
#         print(predict.shape)

#         diff = predict - test

#         print("MSE:", np.mean(diff**2))
#         print("MAE:", np.mean(abs(diff)))
#         print("RMSE:", np.sqrt(np.mean(diff**2)))

#         plt.figure(figsize=(20,7))
#         plt.plot(df['date'].values[800:], df_volume[800:], color = 'red', label = 'Real lgchem Stock Price')
#         plt.plot(df['date'][-predict.shape[0]:].values, predict, color = 'blue', label = 'Predicted lgchem Stock Price')
#         plt.xticks(np.arange(100,df[800:].shape[0],200))
#         plt.title('lgchem Stock Price Prediction')
#         plt.xlabel('Date')
#         plt.ylabel('Price (₩)')
#         plt.legend()
#         plt.show()

# if __name__ =='__main__':
#     s = StockService()
#     s.hook()
#     #StockService.hook()