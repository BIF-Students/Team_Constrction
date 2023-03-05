from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras import regularizers

from keras.layers import LeakyReLU
from keras.layers import BatchNormalization
from keras.utils import plot_model
from matplotlib import pyplot
import pandas as pd
from keras import regularizers
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing

import helpers

from sklearn.model_selection import train_test_split


event_data_cleaned = pd.read_csv ('C:/ITU/ITU_Research_Project/events_cleaned_temp.csv')

event_data_cleaned.drop(event_data_cleaned.iloc[:, 0:2], inplace=True, axis=1)

#Pre-preocessing
#Replace nan values with 0
event_data_cleaned_no_zones = event_data_cleaned.drop(['Simple pass_zone', 'Cross_zone', 'Launch_zone', 'Ground attacking duel_zone', 'Head pass_zone', 'Ground loose ball duel_zone', 'Ball out of the field_zone', 'Touch_zone', 'Ground defending duel_zone', 'Free Kick_zone', 'High pass_zone', 'Hand foul_zone', 'Throw in_zone', 'Smart pass_zone', 'Free kick cross_zone', 'Foul_zone', 'Save attempt_zone', 'Goal kick_zone', 'Acceleration_zone', 'Corner_zone', 'Goalkeeper leaving line_zone', 'Shot_zone', 'Reflexes_zone', 'Hand pass_zone', 'Protest_zone', 'Late card foul_zone', 'Late card foul_zone', 'Penalty_zone', 'Time lost foul_zone', 'Out of game foul_zone', 'Violent Foul_zone', 'Simulation_zone'], axis=1)

prob_missing = 0.1
event_data_cleaned_incomplete = event_data_cleaned.copy()
event_data_cleaned_na_removed = event_data_cleaned.dropna()


x_train,x_test =train_test_split(event_data_cleaned_no_zones,test_size=0.2)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0





'''input_size = 111
hidden_size = 50
code_size = 32
'''
#input_dat = keras.Input(shape=(input_size,))
#encoded = Dense(input_size, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_dat)
#decoded = Dense(784, activation='sigmoid')(encoded)
#autoencoder = keras.Model(input_dat, decoded)
'''
input = Input(shape=(input_size,))
hidden_1 = Dense(hidden_size, activation='relu')(input)
code = Dense(code_size, activation='relu')(hidden_1)
hidden_2 = Dense(hidden_size, activation='relu')(code)
output = Dense(input_size, activation='sigmoid')(hidden_2)



autoencoder = Model(input, output)
autoencoder.compile(optimizer='adam', loss='MSE')
autoencoder.fit(x_train, x_train, epochs=1000)
'''
'''
encoding_dim = 111

input_layer = Input(shape=(encoding_dim,))

encoder = Dense(111, activation='tanh')(input_layer)
encoder = Dense(80, activation='tanh')(encoder)
encoder = Dense(50, activation='tanh', name='bottleneck_layer')(encoder)

decoder = Dense(50, activation='tanh')(encoder)
decoder = Dense(80, activation='tanh')(decoder)
decoder = Dense(111, activation='sigmoid')(decoder)


# full model
model_full = Model(input_layer, decoder)

model_full.compile(optimizer='Adam', loss='mean_squared_error')

model_full.fit(x_train, x_train,
            epochs=10,
            batch_size=2,
            shuffle=True,
            validation_data=(x_test, x_test))

# bottleneck model
bottleneck_output = model_full.get_layer('bottleneck_layer').output
print(bottleneck_output)


model_bottleneck = Model(inputs = model_full.input, outputs = bottleneck_output)

bottleneck_predictions = model_bottleneck.predict(X_inference)'''

def statMissingValue(X):
    lstSummary = []
    for col in X.columns:
        liTotal = len(X.index)
        liMissing = X[col].isna().sum()
        lfMissingRate = round(liMissing * 100/liTotal,2)
        liZero = 0
        liNUnique = X[col].nunique()
        if(X[col].dtype!='object'):
            liZero = X[col].isin([0]).sum()
        lfZeroRate = round(liZero*100/liTotal,2)
        lstSummary.append([col,str(X[col].dtype),liTotal, liNUnique, liMissing, lfMissingRate,liZero,lfZeroRate])
    return pd.DataFrame(lstSummary,columns=['feature','col_type','total', 'unique', 'na','na_rate','zero','zero_rate'])

df_stat = statMissingValue(event_data_cleaned)



df_filtered = df_stat[df_stat['na_rate'] < 50]
plt.figure(figsize=(20,4))
plt.barh(df_filtered.feature, df_filtered.na_rate, label='na rate (%)')
plt.legend()
plt.show()


def fill_na_with_random(df_ref, df_na):
    df_ret = df_na.copy()
    for col in df_ret.columns:
        ret_nan = df_ret[col][df_ret[col].isna()]
        ref_n_nan = df_ref[~df_ref[col].isna()][col]
        df_ret[col].loc[df_ret[col].isna()] = np.random.choice(ref_n_nan, size=len(ret_nan))
    return df_ret

data_upd = event_data_cleaned[df_filtered.feature]


x_train_v2,x_test_v2 =train_test_split(data_upd,test_size=0.2)
x_train_v2 = x_train_v2.astype('float32') / 255.0
x_test_v2 = x_test_v2.astype('float32') / 255.0
x_ref = x_train_v2.copy()
x_train_v2 = fill_na_with_random(x_ref, x_train_v2)
x_test_v2 = fill_na_with_random(x_ref, x_test_v2)

size = x_train_v2.shape[1]

input_layer = Input(shape=(size,))

encoder = Dense(100, activation='relu')(input_layer)
encoder = Dense(80, activation='relu')(encoder)
encoder = Dense(60, activation='relu')(encoder)
encoder = Dense(40, activation='relu')(encoder)
encoder = Dense(20, activation='relu')(encoder)
encoder = Dense(10, activation='relu', name='bottleneck_layer')(encoder)

decoder = Dense(10, activation='relu')(encoder)
decoder = Dense(20, activation='relu')(decoder)
decoder = Dense(40, activation='relu')(decoder)
decoder = Dense(60, activation='relu')(decoder)
decoder = Dense(80, activation='relu')(decoder)
decoder = Dense(119, activation='sigmoid')(decoder)


# full model
model_full = Model(input_layer, decoder)

model_full.compile(optimizer='Adam', loss='mean_squared_error')

model_full.fit(x_train_v2, x_train_v2,
            epochs=100,
            batch_size=10,
            shuffle=True,
            validation_data=(x_test_v2, x_test_v2))

# bottleneck model
bottleneck_output = model_full.get_layer('bottleneck_layer').output
model_bottleneck = Model(inputs = model_full.input, outputs = bottleneck_output)

bottleneck_predictions = model_bottleneck.predict(x_test_v2)


