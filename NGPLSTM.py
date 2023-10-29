# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import LSTM,Dense,Dropout
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, r2_score



data = pd.read_csv('daily.csv')
data = data.dropna()

data['Date'] = pd.to_datetime(data['Date'], format='%Y-%m-%d')

#First data visualization
plt.figure(figsize=(12, 6))  
plt.plot(data['Date'],data['Price'], label='Data', color='blue')
plt.title('Natural Gas Prices')
plt.xlabel('Year')
plt.ylabel('Prices')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)

plt.show()

#Data preprocessing
data.set_index('Date',inplace=True)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

training_data = math.ceil(0.8 * len(data))
train_data = scaled_data[0:training_data, :]

x_train = []
y_train = []

for i in range(50, len(train_data)):
    x_train.append(train_data[i-50:i, 0])
    y_train.append(train_data[i, 0])
    
x_train, y_train = np.array(x_train) , np.array(y_train)
x_train = np.reshape(x_train,(x_train.shape[0], x_train.shape[1],1))

#Model
tf.keras.callbacks.EarlyStopping(
    'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

es_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

model =Sequential() 
model.add(LSTM(128, return_sequences=True, input_shape =(x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics =["accuracy"])
model.fit(x_train,y_train,epochs=50,validation_split=0.2,callbacks=[es_callback])

#Test data preprocessing
test_data = scaled_data[training_data - 50 :,:]
x_test = []

y_test = data.iloc[training_data:,:]

for y in range(50,len(test_data)):
    x_test.append(test_data[y-50:y,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)
print(r2_score(y_test,predictions))



#Train datas, test datas and predictions datas visualization
train = data[:training_data]
chart = data[training_data:]
chart['Predictions'] = predictions


plt.figure(figsize=(16,8))
plt.title('Natural Gas Prices')
plt.xlabel('Dates', fontsize=18)
plt.ylabel('Prices', fontsize=18)
plt.plot(data['Price'],color="purple")
plt.plot(chart[['Price','Predictions']])
plt.legend(['Train','Values','Predictions'],loc='lower right')
plt.show()
