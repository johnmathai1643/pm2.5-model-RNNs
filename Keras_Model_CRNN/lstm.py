import os
import time
import warnings
import numpy as np
import keras
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import LSTM, GRU, CuDNNGRU, CuDNNLSTM, Input, TimeDistributed, Flatten, Bidirectional, Conv1D, MaxPooling1D
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

    #batch normalize before elu layer at axis =-1
    #check wat BN for differernt axis in keras


    #pm25
    inputs_pm25 = Input(shape=(layers[1], 1))
    conv_1_pm25 = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_pm25)
    conv_1_pm25 = BatchNormalization()(conv_1_pm25)
    # conv_1_pm25 = MaxPooling1D(4)(conv_1_pm25)

    #ws
    inputs_ws = Input(shape=(layers[1], 1))
    conv_1_ws = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_ws)
    conv_1_ws = BatchNormalization()(conv_1_ws)
    # conv_1_ws = MaxPooling1D(4)(conv_1_ws)

    #rh
    inputs_rh = Input(shape=(layers[1], 1))
    conv_1_rh = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_rh)
    conv_1_rh = BatchNormalization()(conv_1_rh)
    # conv_1_rh = MaxPooling1D(4)(conv_1_rh)

    #bp
    inputs_bp = Input(shape=(layers[1], 1))
    conv_1_bp = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_bp)
    conv_1_bp = BatchNormalization()(conv_1_bp)
    # conv_1_bp = MaxPooling1D(4)(conv_1_bp)

    #vws
    inputs_vws = Input(shape=(layers[1], 1))
    conv_1_vws = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_vws)
    conv_1_vws = BatchNormalization()(conv_1_vws)
    # conv_1_vws = MaxPooling1D(4)(conv_1_vws)

    #sr
    inputs_sr = Input(shape=(layers[1], 1))
    conv_1_sr = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_sr)
    conv_1_sr = BatchNormalization()(conv_1_sr)
    # conv_1_sr = MaxPooling1D(4)(conv_1_sr)

    #wd
    inputs_wd = Input(shape=(layers[1], 1))
    conv_1_wd = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_wd)
    conv_1_wd = BatchNormalization()(conv_1_wd)
    # conv_1_wd = MaxPooling1D(4)(conv_1_wd)

    #temp
    inputs_temp = Input(shape=(layers[1], 1))
    conv_1_temp = (Conv1D(128, 8, kernel_initializer='glorot_uniform', activation='elu'))(inputs_temp)
    conv_1_temp = BatchNormalization()(conv_1_temp)
    # conv_1_temp = MaxPooling1D(4)(conv_1_temp)

    #concatenate
    output = keras.layers.concatenate([conv_1_pm25, conv_1_ws, conv_1_rh, conv_1_bp, conv_1_vws, conv_1_sr, conv_1_wd, conv_1_temp])
    # output = BatchNormalization()(output)

    lstm_1 = (CuDNNGRU(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros'))(output)
    lstm_1 = Dropout(dropouts[0])(lstm_1)

    lstm_2 = (CuDNNGRU(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros'))(lstm_1)
    lstm_2 = Dropout(dropouts[1])(lstm_2)

    output = TimeDistributed(Dense(layers[2], activation='linear'))(lstm_2)
    output = BatchNormalization()(output)
    output = TimeDistributed(Dense(1, activation='linear'))(output)
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
