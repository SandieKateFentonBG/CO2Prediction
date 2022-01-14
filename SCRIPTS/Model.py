
#todo : why doesn't his allow for any regularization?
# lasso/ridge - regularization integrated


def buildLinearRegressionModel():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    return model

def buildRandomForestModel():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    return model

def buildSVMRegressionModel():
    from sklearn.svm import SVR
    model=SVR()
    return model

# def buildXGBOOSTRegModel():
#     import xgboost as xgb #todo : import xgboost
#     model = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,max_depth = 5, alpha = 10, n_estimators = 300)
#     return model


def buildNormalModel():
    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense
    from tensorflow.python.keras import initializers, optimizers
    """
    QUESTIONS:

    How choose my initializer - variance scaling?
    allows to initiate weights with tensor of values within a variance - all slightly different

    """
    # look for regularization with keras
    initializer = initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])
    model.summary()
    return model
