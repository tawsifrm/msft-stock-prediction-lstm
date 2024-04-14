from datetime import datetime 
import tensorflow as tf 
from tensorflow import keras 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler 
import numpy as np 
from keras.metrics import RootMeanSquaredError

# Load the dataset
microsoft = pd.read_csv('MicrosoftStock.csv') 

# Data Preprocessing
# Handle missing values
microsoft.dropna(inplace=True)

# Normalize the data
ss = StandardScaler()
msft_close = microsoft.filter(['close', 'open', 'high', 'low', 'volume'])
msft_close_scaled = ss.fit_transform(msft_close)

# Prepare the training set samples 
dataset = msft_close_scaled
training = int(np.ceil(len(dataset) * 0.8))  # 80% of data for training

x_train = []
y_train = []

# Create the X_train and y_train
for i in range(60, len(dataset)):
    x_train.append(dataset[i-60:i, :])
    y_train.append(dataset[i, 0])  # Close price is the target

x_train, y_train = np.array(x_train), np.array(y_train)
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2])) 

# Build the Model
model = keras.models.Sequential() 
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))) 
model.add(keras.layers.LSTM(units=64)) 
model.add(keras.layers.Dense(128, activation='relu'))  # Add activation function
model.add(keras.layers.Dropout(0.5)) 
model.add(keras.layers.Dense(1)) 

print(model.summary()) 

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=[RootMeanSquaredError()])  # Use MSE loss

# Fit the model to the training data
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)  # Add validation split

# Testing the model and plotting predictions
testing = dataset[training - 60:, :]
x_test = []
y_test = dataset[training:, 0]  # Close price is the target

for i in range(60, len(testing)):
    x_test.append(testing[i-60:i, :])

x_test = np.array(x_test)
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))

# Make predictions
pred = model.predict(X_test)

# Plotting the results
plt.figure(figsize=(10, 8))
plt.plot(y_test, c="b")  # Actual close prices
plt.plot(pred, c="r")   # Predicted close prices
plt.title('Microsoft Stock Close Price')
plt.ylabel("Close")
plt.legend(['Actual', 'Predicted'])
plt.show()
