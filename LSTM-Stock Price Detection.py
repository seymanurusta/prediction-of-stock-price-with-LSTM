import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import math
from sklearn.metrics import mean_squared_error

dftrain = pd.read_csv("Google_Stock_Price_Train.csv")
trainset = dftrain.iloc[:,1:2].values

scaler = MinMaxScaler(feature_range = (0,1))
trainset_scaled = scaler.fit_transform(trainset)

xtrain = []
ytrain = []

for i in range(60,1258):
    xtrain.append(trainset_scaled[i-60:i,0])
    ytrain.append(trainset_scaled[i,0])
    
xtrain,ytrain = np.array(xtrain), np.array(ytrain)

xtrain =  np.reshape(xtrain,(xtrain.shape[0],xtrain.shape[1],1))


model = Sequential()

model.add(LSTM(units = 50, return_sequences = True, input_shape = (xtrain.shape[1],1)))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))

model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))

model.compile(optimizer = "adam", loss = "mean_squared_error")

model.fit(xtrain, ytrain, epochs = 130, batch_size = 32)

dftest = pd.read_csv("Google_Stock_Price_Test.csv")
testset = dftest.iloc[:,1:2].values

dataset_total = pd.concat((dftrain["Open"], dftest["Open"]),axis = 0) 
inputs = dataset_total[len(dataset_total)-len(dftest)-60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs)

xtest = []

for i in range(60,80):
    xtest.append(inputs[i-60:i,0])

xtest = np.array(xtest)
xtest = np.reshape(xtest,(xtest.shape[0],xtest.shape[1],1))

ypred = model.predict(xtest)
ypred = scaler.inverse_transform(ypred)

mse = math.sqrt(mean_squared_error(testset, ypred))
print("Mean Squared Error:", mse)

plt.plot(testset, color = "red", label = "Real Numbers")
plt.plot(ypred, color = "blue", label = "Predicted Numbers")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")
plt.legend()
plt.show()
