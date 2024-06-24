import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import load_model
from keras.layers import InputLayer, LSTM, Dropout, Dense
from keras.initializers import Orthogonal, GlorotUniform, Zeros

custom_objects = {
    'InputLayer': InputLayer,
    'LSTM': LSTM,
    'Dropout': Dropout,
    'Dense': Dense,
    'Orthogonal': Orthogonal,
    'GlorotUniform': GlorotUniform,
    'Zeros': Zeros
}

model_path = r'C:\Users\amith\OneDrive\Desktop\Stock Predictions Model.keras'

try:
    model = load_model(model_path, custom_objects=custom_objects)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")


# Streamlit header
st.header('Stock Market Predictor')

# Stock symbol input
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Prepare training and testing data
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the test data
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA50
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(data.Close, 'g')
st.pyplot(fig1)

# Plot Price vs MA50 vs MA100
st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r')
plt.plot(ma_100_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig2)

# Plot Price vs MA100 vs MA200
st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
st.pyplot(fig3)

# Prepare data for model prediction
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict using the model
predict = model.predict(x)

# Rescale predictions
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
