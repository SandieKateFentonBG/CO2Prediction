from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from Archiver import *
import numpy as np
from PlotSearch import *
from PlotPredTruth import *

"""
Docum

Linear Models : https://scikit-learn.org/stable/modules/linear_model.html#elastic-net

examples : https://www.programcreek.com/python/example/91151/sklearn.model_selection.GridSearchCV

gridsearch doc : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV

kernel vs svr : https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
"""

def computeAccuracy(yTrue, yPred, tolerance):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

def paramEval(model, paramkey, paramValues, cv, xTrain, yTrain, displayParams, custom = False):

    parameters = dict()
    if paramkey:
        parameters[paramkey] = paramValues
    if custom:
        score = make_scorer(computeAccuracy(), greater_is_better=True)
        grid = GridSearchCV(model, scoring=score, param_grid=parameters, cv = cv)
    else:
        grid = GridSearchCV(model, param_grid=parameters, cv = cv, return_train_score=True)
    grid.fit(xTrain, yTrain.ravel())

    paramDict = {'paramMeanScore': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['mean_test_score'])],
                 'paramStdScore': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['std_test_score'])],
                 'paramRankScore': list(grid.cv_results_['rank_test_score']),
                 'paramValues': paramValues}

    return grid, paramDict

def modelEval(modelWithParam, Linear, xTrain, yTrain, xTest, yTest, displayParams, modelingParams, bestParam = None):

    clf = modelWithParam
    clf.fit(xTrain, yTrain.ravel())
    trainScore = clf.score(xTrain, yTrain.ravel())
    testScore = clf.score(xTest, yTest.ravel())
    yPred = clf.predict(xTest)
    accuracy = computeAccuracy(yTest, clf.predict(xTest),modelingParams['accuracyTol'])
    mse = mean_squared_error(yTest, clf.predict(xTest))
    r2 = r2_score(yTest, clf.predict(xTest))
    if bestParam:
        vals = [bestParam[k] for k in bestParam.keys()]
        val = vals[0]
    else :
        val = 'default'
    modeldict = {'bModel': clf,'bModelTrScore':round(trainScore, displayParams['roundNumber']) , 'bModelTeScore': round(testScore, displayParams['roundNumber']),
                 'bModelParam': val,'bModelAcc': round(accuracy, displayParams['roundNumber']),
                 'bModelMSE': round(mse, displayParams['roundNumber']),'bModelr2': round(r2, displayParams['roundNumber'])}
    # format weight to list
    if Linear:
        content = clf.coef_
        if type(content[0]) == np.ndarray:
            content = content[0]
    else:
        content = clf.dual_coef_
        if type(content[0]) == np.ndarray:
            content = content[0]

    modeldict['bModelWeights'] = [round(num, displayParams['roundNumber']) for num in list(content)]

    if displayParams['showPlot']:
        plotPredTruth(yTest, yPred, displayParams, modeldict)

    return modeldict

def searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features):

    for m in models:
        m['features'] = features
        bestModel, paramDict = paramEval(m['model'], m['param'], modelingParams['RegulVal'], modelingParams['CVFold'], xTrain, yTrain, displayParams)
        m.update(paramDict)
        m['bModelParam'] = bestModel.best_params_
        bModelDict = modelEval(m['model'], m['Linear'], xTrain, yTrain, xTest, yTest,
                              displayParams, modelingParams, m['bModelParam'])
        m.update(bModelDict)
        resDict = paramResiduals(m['model'], xTrain, yTrain, xTest, yTest, displayParams, m['bModelParam'],
                                 yLim = displayParams['residualsYLim'], xLim = displayParams['residualsXLim'], fontsize = displayParams['fontsize'])
        m.update(resDict)

        if displayParams["showResults"]:
            print(m)

    if displayParams["archive"] or displayParams["showPlot"]:
        saveStudy(displayParams, models)

    if displayParams["showResults"]:
        printStudy(displayParams, models)

    MetricsSummaryPlot(models, displayParams)
    predTruthCombined(displayParams, models, xTrain, yTrain, Train=True)

    return models

def paramResiduals(modelWithParam, xTrain, yTrain, xTest, yTest, displayParams, bestParam = None, yLim = None , xLim = None, fontsize = None):
    import matplotlib.pyplot as plt
    from yellowbrick.regressor import ResidualsPlot
    title = 'Residuals for ' + str(modelWithParam)
    if bestParam:
        title += '- BEST PARAM (%s) ' % bestParam
    fig = plt.figure()#figsize=(18,10)
    if fontsize:
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel('Predicted Value ', fontsize=14)
        plt.ylabel('Residuals', fontsize=14)
    ax = plt.gca()
    if yLim:
        plt.ylim(yLim[0], yLim[1])
    if xLim:
        plt.xlim(xLim[0], xLim[1])
    visualizer = ResidualsPlot(modelWithParam, title = title, fig=fig, hist =True)#"frequency" qqplot = True
    visualizer.fit(xTrain, yTrain.ravel())  # Fit the training data to the visualizer
    visualizer.score(xTest, yTest.ravel())  # Evaluate the model on the test data
    # visualizer.hax.grid(False)

    resDict = {'bModelResTrScore': round(visualizer.train_score_, displayParams['roundNumber']),
               'bModelResTeScore': round(visualizer.test_score_, displayParams['roundNumber'])}

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




