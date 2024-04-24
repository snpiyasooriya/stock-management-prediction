import sys

import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler


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

    # Save the scaler to use for inverse transformation later
    scaler = normalizer

    # Create dataset for testing
    x_test, y_test = create_dataset(ds_test_scaled, time_steps)

    # Reshape for the LSTM model
    x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

    # Return both the scaled and unscaled data
    return ds_test_scaled, x_test, scaler

def predict(test_data, time_steps):
    fut_input_scaled, x_test, scaler = process_input_data(test_data, time_steps)
    tmp_inp = list(fut_input_scaled[0])
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
        elif len(tmp_inp) < time_steps:
            # Handle cases where tmp_inp has fewer elements than time_steps
            # Pad tmp_inp with zeros to make its length equal to time_steps
            tmp_inp.extend([0] * (time_steps - len(tmp_inp)))
            fut_inp = np.array(tmp_inp).reshape(1, -1)
            fut_inp = fut_inp.reshape((1, time_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
        else:
            fut_inp = np.array(tmp_inp).reshape(1, -1)
            fut_inp = fut_inp.reshape((1, time_steps, 1))
            yhat = model.predict(fut_inp, verbose=0)
            tmp_inp.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
    # Inverse transform the predicted data to get the actual values
    lst_output_actual = scaler.inverse_transform(lst_output)
    return lst_output_actual 


if __name__ == "__main__":
    stock_symbol = sys.argv[1]
    # Convert space-separated string to list of floats
    input_data = list(map(float, sys.argv[2].split()))
    model_path = '/home/sayuru/Projects/stock-management/stock-management-prediction/prediction_models/' + stock_symbol + '/Open.h5'
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    try:
        prediction = predict(input_data, 100)
        # Print predictions without brackets
        for pred in prediction:
            print(pred[0], end=' ')  # Print each prediction value followed by a space
        print()  # Print a newline character after printing all predictions
    except Exception as e:
        print(f"Error predicting: {e}")
        sys.exit(1)
