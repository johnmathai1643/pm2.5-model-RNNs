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

    #main_input
    inputs = Input(shape=(layers[1], 1), name='main_input')
    # inputs_hd = TimeDistributed(Dense(en_layers[0]))(inputs)
    inputs_op = TimeDistributed(Dense(1))(inputs)
    inputs_op = BatchNormalization()(inputs_op)

    lstm_1 = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros'))(inputs_op)
    lstm_1 = Dropout(dropouts[0])(lstm_1)
    lstm_1 = BatchNormalization()(lstm_1)
    # lstm_1_sum = keras.layers.Add()([lstm_1, inputs_op])

    lstm_2 = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros'))(lstm_1)
    lstm_2 = Dropout(dropouts[1])(lstm_2)
    lstm_2 = BatchNormalization()(lstm_2)

    lstm_2 = keras.layers.concatenate([lstm_2, lstm_1])
    # lstm_2_sum = keras.layers.Add()([lstm_2, lstm_1])
    lstm_3 = (CuDNNLSTM(layers[2],return_sequences=True, kernel_initializer='glorot_uniform',
     recurrent_initializer='orthogonal', bias_initializer='zeros'))(lstm_2)
    lstm_3 = Dropout(dropouts[2])(lstm_3)
    lstm_3 = BatchNormalization()(lstm_3)

    # unit_forget_bias=True,
    #aux_input

    auxiliary_inputs = Input(shape=(layers[1], layers[0]), name='aux_input')
    auxiliary_outputs = BatchNormalization()(auxiliary_inputs)
    auxiliary_outputs = TimeDistributed(Dense(en_layers[0], activation='tanh'))(auxiliary_outputs)
    auxiliary_outputs = BatchNormalization()(auxiliary_outputs)
    auxiliary_outputs = TimeDistributed(Dense(en_layers[1], activation='tanh'))(auxiliary_outputs)
    auxiliary_outputs = BatchNormalization()(auxiliary_outputs)

    #concatenate
    output = keras.layers.concatenate([lstm_3, auxiliary_outputs])
    output = BatchNormalization()(output)

    output = TimeDistributed(Dense(512, activation='tanh'))(output)
    output = BatchNormalization()(output)
    output = TimeDistributed(Dense(1, activation='tanh'))(output)
    output = BatchNormalization()(output)
    output = Flatten()(output)
    output = (Dense(1, activation='linear'))(output)
    model = Model(inputs=[inputs, auxiliary_inputs], outputs=[output])

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
