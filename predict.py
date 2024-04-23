import sys

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

model=load_model('prediction_models/TSLA/Open.h5')

# Function to create dataset for LSTM model
def create_dataset(dataset, step):
    x_test, y_test = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        x_test.append(a)
        y_test.append(dataset[i + step, 0])
    return np.array(x_test), np.array(y_test)

def process_input_data(test_data, time_steps):
    # Normalize data
    normalizer = MinMaxScaler(feature_range=(0, 1))
    ds_test_scaled = normalizer.fit_transform(np.array(test_data).reshape(-1, 1))

    #create dataset for testing
    x_test,y_test=create_dataset(ds_test_scaled,time_steps)

    #reshape for the LSTM model
    x_test=x_test.reshape(x_test.shape[0], x_test.shape[1],1)
    return ds_test_scaled

def predict(test_data, time_steps):
    fut_input=process_input_data(test_data, time_steps)
    tmp_inp=list(fut_input[0])
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
    return lst_output


if __name__ == "__main__":
    stock_symbol = sys.argv[1]
    input_data = list(map(float, sys.argv[2:]))
    print(predict(input_data, stock_symbol))
