
from PrepData import *
from ModelConstructor import *
from sklearn.model_selection import GridSearchCV, cross_val_score

def searchCV(filterDf, yLabels, model, parameters, testSetIndex=1):

    # data
    (xTrain, yTrain),(xTest, yTest) = TrainTestArray(filterDf, yLabels, testSetIndex=testSetIndex)
    # grid search
    grid = GridSearchCV(model, param_grid=parameters)
    grid.fit(xTrain, yTrain.ravel())

    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))

    scores = grid.cv_results_['mean_test_score']
    print("Scores:")
    print(scores)

    print("c_range:", parameters['C'])

    # clf = model(grid.best_params_) #verbose : Controls the verbosity:computation time/messages

    return grid


def learn(method, xTrain, yTrain, xTest, yTest, scaler, epochs, display, tolerance=0.05): #modelingParams
    """Build / Train / Evaluate Model """
    if method =='Nmodel':
        model = buildNormalModel()
        model.fit(xTrain, yTrain.values.ravel(), epochs=epochs)
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
        plot(yTest, yPred)

    return model, evalu, acc, mse

#todo : rplace learn with this function - it runs fit evalluate and returrns many infos
# sklearn.model_selection.GridSearchCV(estimator, param_grid, *, scoring=None, n_jobs=None, refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=nan, return_train_score=False)[source]


def execute(filterDf,yLabels, method, epochs=None, singleTest = 1, display = False):
    run = dict()
    run['method'] = method
    xs, ys = TrainTestSets(filterDf, yLabels)
    run['xs'], run['ys'], run['scaler'] = xs, ys
    if singleTest:
        run['singleTest'] = True
        (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xs, ys, testSetIndex=singleTest)
        model, evalu, acc, mse = learn(method, xTrain, yTrain, xTest, yTest, epochs=epochs, display = display)

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
            (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xs, ys, testSetIndex=i)
            model, evalu, acc, mse = learn(method, xTrain, yTrain, xTest, yTest,  epochs=epochs, display=display)
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