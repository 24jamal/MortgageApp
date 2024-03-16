###LSTM Algorithm


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Fetch historical stock price data for Infosys Limited (INFY.NS) using yfinance
infy_data = yf.download('INFY.NS', start='2010-01-01', end='2024-01-01')

# Data preprocessing
data = infy_data['Close'].values.reshape(-1, 1)  # Extract 'Close' prices as numpy array
scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize data to range [0, 1]
scaled_data = scaler.fit_transform(data)

# Split data into training and testing sets
training_data_len = int(len(scaled_data) * 0.8)  # 80% of data for training, 20% for testing

train_data = scaled_data[:training_data_len]
test_data = scaled_data[training_data_len:]

# Function to create sequences and labels
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        X.append(seq)
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

# Define sequence length (number of time steps to look back)
seq_length = 60  # You can adjust this parameter based on your preference

# Create sequences and labels for training and testing sets
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

# Build LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32)

# Make predictions
predictions = model.predict(X_test)

# Inverse scaling to get actual prices
predictions = scaler.inverse_transform(predictions)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(y_test, label='Actual Stock Price')
plt.plot(predictions, label='Predicted Stock Price')
plt.title('Infosys Limited (INFY.NS) Stock Price Prediction using LSTM:')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
