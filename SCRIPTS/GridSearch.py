from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from Archiver import *
from PrepData import TrainTestDf
"""
Docum

Linear Models : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net

examples : https://www.programcreek.com/python/example/91151/sklearn.model_selection.GridSearchCV

gridsearch doc : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

kernel vs svr : https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
"""


def computeAccuracy(yTrue, yPred):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    tolerance = 0.05
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

def plotPredTruth(yTest, yPred, displayParams, modeldict):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [18, 18]
    # plt.grid()
    print(type(yPred[0]))
    l1, = plt.plot(yTest, 'g')
    l2, = plt.plot(yPred, 'r', alpha=0.7)
    plt.legend(['Ground truth', 'Predicted'], fontsize=18)
    title = str(modeldict['model'])+ '- BEST PARAM (%s) ' % modeldict['bestParam'] \
            + '- SCORE : ACC(%s) ' % modeldict['accuracy'] + 'MSE(%s) ' % modeldict['mse'] + 'R2(%s)' % modeldict['r2']
    plt.title(title, fontdict = {'fontsize' : 20})
    plt.xticks(fontsize=14)
    plt.xlabel('Test Building', fontsize=18)
    plt.ylim(ymin=displayParams['TargetMinMaxVal'][0], ymax=displayParams['TargetMinMaxVal'][1])
    plt.yticks(fontsize=14)
    plt.ylabel(displayParams['Target'], fontsize=18)
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Pred_Truth'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + str(modeldict['model']) + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

# def saveStudy(displayParams, Results):
#
#     import os
#     if not os.path.isdir(displayParams["outputPath"]):
#         os.makedirs(displayParams["outputPath"])
#
#     with open(displayParams["outputPath"] + displayParams["reference"] + ".txt", 'a') as f:
#         print('', file=f)
#         if type(Results) == dict:
#             for k,v in Results.items():
#                 print(k, ":", v, file=f)
#         else:
#             for r in Results:
#                 print(r, file=f)
#
#     f.close()
def paramSearch(model, paramkey, paramValues, cv, xTrain, yTrain, custom = False):

    parameters = dict()
    if paramkey:
        parameters[paramkey] = paramValues
    if custom:
        score = make_scorer(computeAccuracy(), greater_is_better=True)
        grid = GridSearchCV(model, scoring=score, param_grid=parameters, cv = cv)
    else:
        grid = GridSearchCV(model, param_grid=parameters, cv = cv)
    grid.fit(xTrain, yTrain.ravel())

    return grid

def paramEval(modelWithParam, xTrain, yTrain, xTest, yTest, displayParams, bestParam = None):

    clf = modelWithParam
    clf.fit(xTrain, yTrain.ravel())
    trainScore = clf.score(xTest, yTest.ravel())
    testScore = clf.score(xTest, yTest.ravel())
    yPred = clf.predict(xTest)
    accuracy = computeAccuracy(yTest, clf.predict(xTest))
    mse = mean_squared_error(yTest, clf.predict(xTest))
    r2 = r2_score(yTest, clf.predict(xTest))
    modeldict = {'model': clf,'trainScore':round(trainScore, displayParams['roundNumber']) , 'testScore': round(testScore, displayParams['roundNumber']),
                 'bestParam': bestParam,'accuracy': round(accuracy, displayParams['roundNumber']),
                 'mse': round(mse, displayParams['roundNumber']),'r2':round(r2, displayParams['roundNumber'])}
    if displayParams['showPlot']:
        plotPredTruth(yTest, yPred, displayParams, modeldict)

    return modeldict

def searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest):

    for m in models:
        bestModel = paramSearch(m['model'], m['param'], modelingParams['RegulVal'], modelingParams['CVFold'], xTrain, yTrain)
        m['bestParam'] = bestModel.best_params_
        modelDict = paramEval(m['model'], xTrain, yTrain, xTest, yTest,
                              displayParams, m['bestParam'])
        m.update(modelDict)
        resDict = paramResiduals(m['model'], xTrain, yTrain, xTest, yTest, displayParams, m['bestParam'])
        m.update(resDict)

        if displayParams["showResults"]:
            print(m)
    if displayParams["archive"] or displayParams["showPlot"]:
        saveStudy(displayParams, models)

    return models


def paramResiduals(modelWithParam, xTrain, yTrain, xTest, yTest, displayParams, bestParam = None):
    import matplotlib.pyplot as plt
    from yellowbrick.regressor import ResidualsPlot
    title = 'Residuals for ' + str(modelWithParam)
    if bestParam:
        title += '- BEST PARAM (%s) ' % bestParam

    visualizer = ResidualsPlot(modelWithParam, title = title, fig=plt.figure(figsize=(18,18)), fontsize = 18)
    #todo : ymin = -0.5, ymax = 0.5
    print(xTrain.shape, yTrain.shape)
    visualizer.fit(xTrain, yTrain.ravel())  # Fit the training data to the visualizer
    visualizer.score(xTest, yTest.ravel())  # Evaluate the model on the test data

    resDict = {'visualizerTrainScore': round(visualizer.train_score_, displayParams['roundNumber']),
               'visualizerTestScore': round(visualizer.test_score_, displayParams['roundNumber'])}

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Residuals'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        visualizer.show(outpath=outputFigPath + '/' + str(modelWithParam) + '.png')

    if displayParams['showPlot']:
        visualizer.show()

    visualizer.finalize()

    return resDict




def rotateSearchEval(xSets, ySets, modelingParams, displayParams, models):
    for i in range(5):

        (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xSets, ySets, i)
        (xTrainArr, yTrainArr), (xTestArr, yTestArr) = (xTrain.values, yTrain.values.reshape(-1, 1)), (
        xTest.values, yTest.values.reshape(-1, 1))
        print('Rotation', i)
        searchEval(modelingParams, displayParams, models, xTrainArr, yTrainArr, xTestArr, yTestArr)
