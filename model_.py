from keras.models import Sequential, Model
from keras.layers import LSTM, Input, Activation, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from statsmodels.tools.eval_measures import rmse, mse, meanabs
import numpy as np
import pandas as pd

from gathering import pipeline_gathering

n_feature = 1
n_input = 12 

def get_y_from_generator(gen):
    '''
    Get all targets y from a TimeseriesGenerator instance.
    '''
    y = None
    for i in range(len(gen)):
        batch_y = gen[i][1]
        if y is None:
            y = batch_y
        else:
            y = np.append(y, batch_y)
    y = y.reshape((-1,1))
    print(y.shape)
    return y


# input_shape=x_train.shape[-2:]

def create_model_lstm(input_shape=(12,1)):
    
    input = Input(input_shape)

    x = LSTM(512, return_sequences=True)(input)
    x = Activation('relu')(x)
    
    x = LSTM(256, return_sequences=False)(x)
    x = Activation('relu')(x)
    
    x = Dropout(0.2)(x)
    x = Dense(50)(x)
    x = Activation('relu')(x)
    
    x = Dropout(0.1)(x)
    x = Dense(1, name='output')(x)

    model = Model(inputs = input, outputs = x, name='TimeSerieModel')
    return model



def prepare_data(data_ts):
    ds = pd.DataFrame(data=data_ts)[['revenue']]

    len_train = int(ds.shape[0] * 0.8) 
    train = ds.iloc[:len_train]
    test = ds.iloc[len_train:]

    scaler = MinMaxScaler()
    scaler.fit(train)
    train = scaler.transform(train)
    test = scaler.transform(test)

    return train, test, scaler 


def train_model(data_ts):
    logging.info('Creating model LSTM...')
    model = create_model_lstm((n_input, n_feature))

    logging.info('Preparing Data.')
    train, test, scaler = prepare_data(data_ts)

    train_generator = TimeseriesGenerator(train, train, length=n_input, batch_size=6)
    model.compile(optimizer='adam', loss='mse')

    logging.info('Starting model training')
    model.fit_generator(train_generator, epochs=70, verbose=1)

    logging.info('Finished model training')
    return model, train, test, scaler


def predict_model(model, scaler, train, days_counts=30):
    logging.info('Predicting data')
    pred_list = []

    batch = train[-n_input:].reshape((1, n_input, 1))

    for i in range(days_counts):   
        pred_list.append(model.predict(batch)[0]) 
        batch = np.append(batch[:,1:,:],[[pred_list[i]]],axis=1)
        
    pred_list = np.asarray(pred_list)

    return pred_list