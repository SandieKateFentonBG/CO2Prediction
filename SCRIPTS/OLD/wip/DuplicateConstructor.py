
#todo : why doesn't his allow for any regularization?
# lasso/ridge - regularization integrated


def buildLinearRegressionModel():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    return model

def buildRandomForestModel():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor()
    # model = RandomForestRegressor(max_depth=2, random_state=0,n_estimators=100)
    return model

def buildXGBOOSTRegModel(regul='reg:squarederror'):  # 'reg:linear'
    import xgboost as xgb
    model = xgb.XGBRegressor(objective=regul, colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10,
                             n_estimators=300)
    # TODO : question : where do these come from?
    # objective = 'reg:linear', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 300)

    return model


def buildSVMRegressionModel():
    from sklearn.svm import SVR
    model=SVR() #(kernel="rbf", gamma=0.1)
    return model

#SVR(*, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200, verbose=False, max_iter=- 1)


def buildKernelRidgeRegressionModel():
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge() #(kernel="rbf", gamma=0.1)

def buildLassoRegressionModel():



def buildNormalModel():
    # from tensorflow.python.keras.models import Sequential
    # from tensorflow.python.keras.layers import Dense
    # from tensorflow.python.keras import initializers, optimizers

    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense
    from keras import initializers, optimizers

    """
    QUESTIONS:

    How choose my initializer - variance scaling?
    allows to initiate weights with tensor of values within a variance - all slightly different

    """
    # look for regularization with keras
    initializer = keras.initializers.VarianceScaling(scale=1.0, mode='fan_in', distribution='normal', seed=None)
    model = Sequential()
    model.add(Dense(4, input_dim=4, kernel_initializer='normal', activation='relu'))
    model.add(Dense(2, activation='relu'))
    model.add(Dense(1, activation='relu'))
    optimizer = keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse', 'mae'])
    model.summary()
    return model
