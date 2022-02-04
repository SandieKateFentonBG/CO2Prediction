
#todo : why doesn't his allow for any regularization?
# lasso/ridge - regularization integrated


def buildLinearRegressionModel():
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)
    return model

def buildSVMRegressionModel():
    from sklearn.svm import SVR
    model=SVR() #(kernel="rbf", gamma=0.1)
    return model

def buildKernelRidgeRegressionModel():
    from sklearn.kernel_ridge import KernelRidge
    model = KernelRidge() #(kernel="rbf", gamma=0.1)

def buildLassoRegressionModel():
    pass


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
