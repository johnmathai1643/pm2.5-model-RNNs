import os
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Input, TimeDistributed, Flatten, Bidirectional
from keras.layers.normalization import BatchNormalization
from keras import regularizers
from keras.models import Sequential, Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def load_data(filename, seq_len, normalise_window):
    f = open(filename, 'rb').read()
    data = f.decode().split('\n')

    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    if normalise_window:
        result = normalise_windows(result)

    result = np.array(result)

    row = round(0.9 * result.shape[0])
    train = result[:int(row), :]
    np.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1]

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return [x_train, y_train, x_test, y_test]

def normalise_windows(window_data):
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data

def build_model(en_layers, layers, dropouts, pre_train=None):
    #layer[1] = seq_len
    #layer[0] = dimensions
    #layer[2], layer[3] = state_neurons
    #layer[4] = output

    #pm25
    inputs_pm25 = Input(shape=(layers[1], 1))
    inputs_op_pm25 = TimeDistributed(Dense(1))(inputs_pm25)
    inputs_op_pm25 = BatchNormalization()(inputs_op_pm25)

    lstm_1_pm25 = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_pm25)
    lstm_1_pm25 = Dropout(dropouts[0])(lstm_1_pm25)
    lstm_1_pm25 = BatchNormalization()(lstm_1_pm25)

    lstm_2_pm25 = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_pm25)
    lstm_2_pm25 = Dropout(dropouts[1])(lstm_2_pm25)
    lstm_2_pm25 = BatchNormalization()(lstm_2_pm25)

    #ws
    inputs_ws = Input(shape=(layers[1], 1))
    inputs_op_ws = TimeDistributed(Dense(1))(inputs_ws)
    inputs_op_ws = BatchNormalization()(inputs_op_ws)

    lstm_1_ws = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_ws)
    lstm_1_ws = Dropout(dropouts[0])(lstm_1_ws)
    lstm_1_ws = BatchNormalization()(lstm_1_ws)

    lstm_2_ws = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_ws)
    lstm_2_ws = Dropout(dropouts[1])(lstm_2_ws)
    lstm_2_ws = BatchNormalization()(lstm_2_ws)

    #rh
    inputs_rh = Input(shape=(layers[1], 1))
    inputs_op_rh = TimeDistributed(Dense(1))(inputs_rh)
    inputs_op_rh = BatchNormalization()(inputs_op_rh)

    lstm_1_rh = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_rh)
    lstm_1_rh = Dropout(dropouts[0])(lstm_1_rh)
    lstm_1_rh = BatchNormalization()(lstm_1_rh)

    lstm_2_rh = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_rh)
    lstm_2_rh = Dropout(dropouts[1])(lstm_2_rh)
    lstm_2_rh = BatchNormalization()(lstm_2_rh)

    #bp
    inputs_bp = Input(shape=(layers[1], 1))
    inputs_op_bp = TimeDistributed(Dense(1))(inputs_bp)
    inputs_op_bp = BatchNormalization()(inputs_op_bp)

    lstm_1_bp = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_bp)
    lstm_1_bp = Dropout(dropouts[0])(lstm_1_bp)
    lstm_1_bp = BatchNormalization()(lstm_1_bp)

    lstm_2_bp = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_bp)
    lstm_2_bp = Dropout(dropouts[1])(lstm_2_bp)
    lstm_2_bp = BatchNormalization()(lstm_2_bp)

    #vws
    inputs_vws = Input(shape=(layers[1], 1))
    inputs_op_vws = TimeDistributed(Dense(1))(inputs_vws)
    inputs_op_vws = BatchNormalization()(inputs_op_vws)

    lstm_1_vws = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_vws)
    lstm_1_vws = Dropout(dropouts[0])(lstm_1_vws)
    lstm_1_vws = BatchNormalization()(lstm_1_vws)

    lstm_2_vws = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_vws)
    lstm_2_vws = Dropout(dropouts[1])(lstm_2_vws)
    lstm_2_vws = BatchNormalization()(lstm_2_vws)

    #sr
    inputs_sr = Input(shape=(layers[1], 1))
    inputs_op_sr = TimeDistributed(Dense(1))(inputs_sr)
    inputs_op_sr = BatchNormalization()(inputs_op_sr)

    lstm_1_sr = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_sr)
    lstm_1_sr = Dropout(dropouts[0])(lstm_1_sr)
    lstm_1_sr = BatchNormalization()(lstm_1_sr)

    lstm_2_sr = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_sr)
    lstm_2_sr = Dropout(dropouts[1])(lstm_2_sr)
    lstm_2_sr = BatchNormalization()(lstm_2_sr)

    #wd
    inputs_wd = Input(shape=(layers[1], 1))
    inputs_op_wd = TimeDistributed(Dense(1))(inputs_wd)
    inputs_op_wd = BatchNormalization()(inputs_op_wd)

    lstm_1_wd = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_wd)
    lstm_1_wd = Dropout(dropouts[0])(lstm_1_wd)
    lstm_1_wd = BatchNormalization()(lstm_1_wd)

    lstm_2_wd = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_wd)
    lstm_2_wd = Dropout(dropouts[1])(lstm_2_wd)
    lstm_2_wd = BatchNormalization()(lstm_2_wd)

    #temp
    inputs_temp = Input(shape=(layers[1], 1))
    inputs_op_temp = TimeDistributed(Dense(1))(inputs_temp)
    inputs_op_temp = BatchNormalization()(inputs_op_temp)

    lstm_1_temp = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(inputs_op_temp)
    lstm_1_temp = Dropout(dropouts[0])(lstm_1_temp)
    lstm_1_temp = BatchNormalization()(lstm_1_temp)

    lstm_2_temp = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros', unit_forget_bias=True))(lstm_1_temp)
    lstm_2_temp = Dropout(dropouts[1])(lstm_2_temp)
    lstm_2_temp = BatchNormalization()(lstm_2_temp)

    #concatenate
    output = keras.layers.concatenate([lstm_2_pm25, lstm_2_ws, lstm_2_rh, lstm_2_bp, lstm_2_vws, lstm_2_sr, lstm_2_wd, lstm_2_temp])
    output = BatchNormalization()(output)

    output = TimeDistributed(Dense(512, activation='tanh'))(output)
    output = BatchNormalization()(output)
    output = TimeDistributed(Dense(1, activation='tanh'))(output)
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = (Dense(1, activation='linear'))(output)
    model = Model(inputs=[inputs_pm25, inputs_ws, inputs_rh, inputs_bp, inputs_vws, inputs_sr, inputs_wd, inputs_temp], outputs=[output])

    if not (pre_train is None):
        model.load_weights(pre_train)

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print(model.summary())
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point_aux(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

def predict_sequence_full(model, data, window_size):
    #Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
        curr_frame = curr_frame[1:]
        print(curr_frame)
        curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        print(curr_frame)
    return predicted

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/prediction_len)):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs
