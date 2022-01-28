from ModelConstructor import *
from PrepData import *

def save_model(dataFrame, model_name, min_value, max_value): #todo : not used or checked yet
    import pickle
    file_name_string = './data/model_data_prediction/' + model_name + "_predictions"
    if min_value != None or max_value != None:
        file_name_string += "_from_" + str(min_value) + "_to_" + str(max_value)
    file_name_string += ".bin"
    print("this is the filename string")
    print("Saving model on " + file_name_string)

    dataFrame[min_value:max_value].to_pickle(file_name_string)

def computeAccuracy(model, xTest, yTest, tolerance): #thos could be done unscaled
    yPred = model.predict(xTest).tolist()
    rTrue = yTest.values.tolist()
    yTrue = [item for sublist in rTrue for item in sublist]
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

def computeMSE(model, xTest, yTest, scaler):
    xScaler, yScaler = scaler
    yPred = model.predict(xTest).reshape(-1, 1)
    yPredScaled = yScaler.inverse_transform(yPred)
    yTestScaled = yScaler.inverse_transform(yTest)
    from sklearn.metrics import mean_squared_error

    return mean_squared_error(yPredScaled, yTestScaled)

def learn(method, xTrain, yTrain, xTest, yTest, scaler, epochs, display, tolerance=0.05): #modelingParams
    """Build / Train / Evaluate Model """
    if method =='Nmodel':
        model = buildNormalModel()
        model.fit(xTrain, yTrain, epochs=epochs)
        evalu = model.evaluate(xTest, yTest)
    else:
        if method =='LRmodel':
            model = buildLinearRegressionModel()
        if method == 'RFmodel':
            model = buildRandomForestModel()
        if method == 'SVMmodel':
            model = buildSVMRegressionModel()
        if method == 'XGBmodel':
            model = buildXGBOOSTRegModel()
        model.fit(xTrain, yTrain.values.ravel())
        #.values.ravel() converts 1D vector to array (len, 1) to (len, ) #todo : check utility cfr line 59
        evalu = model.score(xTest, yTest)
    acc = computeAccuracy(model, xTest, yTest, tolerance=tolerance)
    mse = computeMSE(model, xTest, yTest, scaler)

    if display:
        yPred = model.predict(xTest).reshape(-1, 1)
        #.reshape(-1, 1) converts (len, ) to (len, 1) #todo : check utility cfr line 50
        xScaler, yScaler = scaler
        yPredScaled = yScaler.inverse_transform(yPred)
        yTestScaled = yScaler.inverse_transform(yTest)
        plot(yTestScaled, yPredScaled)

    return model, evalu, acc, mse

def execute(filterDf,yLabels, method, epochs=None, singleTest = 1, display = False):
    run = dict()
    run['method'] = method
    xs, ys, scaler = TrainTestSets(filterDf, yLabels)
    run['xs'], run['ys'], run['scaler'] = xs, ys, scaler
    if singleTest:
        run['singleTest'] = True
        (xTrain, yTrain), (xTest, yTest) = TrainTestSplit(xs, ys, testSetIndex=singleTest)
        model, evalu, acc, mse = learn(method, xTrain, yTrain, xTest, yTest, scaler, epochs=epochs, display = display)

        run['model'] = model #todo : def archive()
        run['evalu'] = evalu
        run['acc'] = acc
        run['mse'] = mse
        run['xTrain'] = xTrain
        run['yTrain'] = yTrain
        run['xTest'] = xTest
        run['yTest'] = yTest


    else:
        run['singleTest'] = False
        models, evalus, accs, mses = [], [], [], []
        for i in range(5):
            (xTrain, yTrain), (xTest, yTest) = TrainTestSplit(xs, ys, testSetIndex=i)
            model, evalu, acc, mse = learn(method, xTrain, yTrain, xTest, yTest, scaler, epochs=epochs, display=display)
            models.append(model),  evalus.append(evalu), accs.append(acc), mses.append(mse) #fits.append(fit),

        run['model'] = models #todo : def archive()
        run['evalu'] = evalus
        run['acc'] = accs
        run['mse'] = mses
        run['avgEvalu'] = sum(evalus) / len(evalus)
        run['avgAccs'] = sum(accs) / len(accs)
        run['avgMSEs'] = sum(mses) / len(mses)

    return run

def plot(yTest, yPred):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [18, 18]
    l1, = plt.plot(yTest, 'g')
    l2, = plt.plot(yPred, 'r', alpha=0.7)
    plt.legend(['Ground truth', 'Predicted'])
    plt.show()


def archive():
    pass