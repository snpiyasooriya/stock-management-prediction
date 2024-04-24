import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from pathlib import Path


# Function to create dataset for LSTM model
def create_dataset(dataset, step):
    X_train, y_train = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        X_train.append(a)
        y_train.append(dataset[i + step, 0])
    return np.array(X_train), np.array(y_train)


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Stock Prediction')
parser.add_argument('stock_symbol', type=str, help='Stock symbol (e.g., TSLA)')
parser.add_argument('value_type', type=str, help='Stock value type (e.g., Close)')
args = parser.parse_args()
stock_symbol = args.stock_symbol
value_type = args.value_type
# Load data from CSV
data = pd.read_csv(
    '/home/sayuru/Projects/stock-management/stock-management-backend/assets/uploads/historical_data/' + stock_symbol + '.csv')
opn = data[[value_type]]
ds = opn.values

# Normalize data
normalizer = MinMaxScaler(feature_range=(0, 1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))

# Define train and test data sizes
train_size = int(len(ds_scaled) * 0.70)
test_size = len(ds_scaled) - train_size
ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]

# Create datasets for training and testing
time_steps = 100
X_train, y_train = create_dataset(ds_train, time_steps)
X_test, y_test = create_dataset(ds_test, time_steps)

# Reshape data for LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='linear'))

# Compile and train the model
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

# Save the model
Path("/home/sayuru/Projects/stock-management/stock-management-prediction/prediction_models/" + stock_symbol).mkdir(parents=True, exist_ok=True)
model.save('/home/sayuru/Projects/stock-management/stock-management-prediction/prediction_models/' + stock_symbol + '/' + value_type + '.h5')

# Predict next 30 days
fut_inp = ds_test[-time_steps:].reshape(1, -1)
tmp_inp = list(fut_inp[0])
lst_output = []
for _ in range(30):
    if len(tmp_inp) > time_steps:
        fut_inp = np.array(tmp_inp[1:])
        fut_inp = fut_inp.reshape(1, -1)
        fut_inp = fut_inp.reshape((1, time_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
    else:
        fut_inp = fut_inp.reshape((1, time_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())

# Combine predicted data with original data
ds_new = ds_scaled.tolist()
ds_new.extend(lst_output)


# Plot the results
# final_graph = normalizer.inverse_transform(ds_new).flatten()
# start_index = len(ds_scaled) - len(ds_test) + time_steps + 1
# end_index = start_index + len(lst_output)
#
# plt.plot(final_graph[start_index:])  # Plot actual values
# plt.plot(np.arange(end_index, end_index + len(lst_output)),
#          normalizer.inverse_transform(lst_output))  # Plot predicted values
# plt.ylabel("Price")
# plt.xlabel("Time")
# plt.title("{0} prediction of next month open".format(stock_symbol))
# plt.axhline(y=final_graph[-1], color='red', linestyle=':',
#             label='NEXT 30D: {0}'.format(round(float(final_graph[-1]), 2)))
# plt.legend()
# plt.show()
