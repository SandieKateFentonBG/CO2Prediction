from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras import initializers
from datetime import datetime as dt
from utils import *
from sklearn.ensemble import RandomForestRegressor
import tensorflow

"""
QUESTIONS:

How choose my initializer - variance scaling? scaling = choose good vector size,
not linked to normalization scaling?

"""



def buildLinearRegressionModel():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    return

def buildRandomForestModel():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    return model

def buildSVMREGRESSIONModel():
    model=SVR()
    return model

def buildXGBOOSTRegModel():
    model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 300)
    return model


def buildNormalModel():
    # look for regularization with keras
    initializer = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    model = Sequential()

    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])
    model.summary()
    return model