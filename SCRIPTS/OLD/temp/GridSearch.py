from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from SCRIPTS.FeatureReport.Archiver import *
from temp.PlotMetrics import *
from SCRIPTS.ModelPredTruthPt import *
from temp.PlotResiduals import *

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

def paramEval(model, paramkey, paramValues, cv, xTrain, yTrain, displayParams, custom = False, refit = 'r2' ):

    """
    Evaluate the
    :param model:
    :param paramkey:
    :param paramValues:
    :param cv:
    :param xTrain:
    :param yTrain:
    :param displayParams:
    :param custom:
    :param refit:
    :return:
    """

    #https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    #https://scikit-learn.org/stable/auto_examples/model_selection/plot_multi_metric_evaluation.html#sphx-glr-auto-examples-model-selection-plot-multi-metric-evaluation-py
    parameters = dict()
    if paramkey:
        parameters[paramkey] = paramValues
    if custom:
        score = make_scorer(computeAccuracy(), greater_is_better=True)
        grid = GridSearchCV(model, scoring=score, param_grid=parameters, cv = cv)
    else:
        # grid = GridSearchCV(model, param_grid=parameters,  cv = cv, return_train_score=True)
        scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        grid = GridSearchCV(model, param_grid=parameters, scoring =scoring, refit = refit,  cv = cv, return_train_score=True)
    grid.fit(xTrain, yTrain.ravel())
    results =grid.cv_results_

    paramDict = {'paramMeanMSETest': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['mean_test_neg_mean_squared_error'])],
                 'paramStdSMSETest': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['std_test_neg_mean_squared_error'])],
                 'paramRankMSETest': list(grid.cv_results_['rank_test_neg_mean_squared_error']),
                 'paramMeanR2Test': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['mean_test_r2'])],
                 'paramStdR2Test': [round(num, displayParams['roundNumber']) for num in list(grid.cv_results_['std_test_r2'])],
                 'paramRankR2Test': list(grid.cv_results_['rank_test_r2']),
                 'paramValues': paramValues}

    return grid, paramDict



def modelEval(modelWithParam, Linear, xTrain, yTrain, xTest, yTest, displayParams, modelingParams, bestParam = None):

    clf = modelWithParam
    clf.fit(xTrain, yTrain.ravel())
    trainScore = clf.score(xTrain, yTrain.ravel())
    testScore = clf.score(xTest, yTest.ravel())
    yPred = clf.predict(xTest).reshape(-1,1)
    residual = yTest - yPred

    accuracy = computeAccuracy(yTest, clf.predict(xTest),modelingParams['accuracyTol'])
    mse = mean_squared_error(yTest, clf.predict(xTest))
    r2 = r2_score(yTest, clf.predict(xTest))

    if bestParam:
        vals = [bestParam[k] for k in bestParam.keys()]
        val = vals[0]
    else :
        val = 'default'
    modeldict = {'bModel': clf,'bModelTrR2':round(trainScore, displayParams['roundNumber']) , 'bModelTeR2': round(testScore, displayParams['roundNumber']),
                 'bModelParam': val,'bModelAcc': round(accuracy, displayParams['roundNumber']), 'bModelResid': residual,
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
    weights = [round(num, displayParams['roundNumber']) for num in list(content)]
    modeldict['bModelWeights'] = weights
    modeldict['bModelWeightsScaled'] = scaledList(weights)


    print('test', modeldict['bModelResid'])
    if displayParams['showPlot'] or displayParams['archive']:
        plotPredTruth(yTest, yPred, displayParams, modeldict)

    return modeldict

def searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features, resPlot = False, restDist = True):

    for m in models:
        m['features'] = features
        bestModel, paramDict = paramEval(m['model'], m['param'], modelingParams['RegulVal'], modelingParams['CVFold'],
                                         xTrain, yTrain, displayParams, refit = modelingParams['rankGridSearchModelsAccordingto'])
        m.update(paramDict)
        m['bModelParam'] = bestModel.best_params_
        bModelDict = modelEval(m['model'], m['Linear'], xTrain, yTrain, xTest, yTest,
                              displayParams, modelingParams, m['bModelParam'])
        m.update(bModelDict)
        if resPlot:
            resDict = paramResiduals(m['model'], xTrain, yTrain, xTest, yTest, displayParams, m['bModelParam'],
                                     yLim = displayParams['residualsYLim'], xLim = displayParams['residualsXLim'], fontsize = displayParams['fontsize'])
            m.update(resDict)
        if restDist:
            plotResiduals(m, displayParams, bestParam=m['bModelParam'])

        if displayParams["showResults"]:
            print(m)

    saveStudy(displayParams, models)

    if displayParams["showResults"]:
        printStudy(displayParams, models)

    if displayParams["archive"] or displayParams["showPlot"]:
        sortedMod = sortGridResults(models, metric = 'bModelAcc', highest = True)
        MetricsSummaryPlot(sortedMod, displayParams, metricLabels = ['bModelTrR2','bModelTeR2','bModelAcc','bModelMSE'])
        MetricsSummaryPlot(sortedMod, displayParams, metricLabels = ['bModelTrR2','bModelTeR2'])
        MetricsSummaryPlot(sortedMod, displayParams, metricLabels = ['bModelAcc'])
        MetricsSummaryPlot(sortedMod, displayParams, metricLabels = ['bModelMSE'])

        predTruthCombined(displayParams, sortedMod, xTest, yTest, Train=False)

    return models



def sortGridResults(models, metric = 'bModelAcc', highest = True):
    return sorted(models, key=lambda x: x[metric], reverse=highest)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# def scaledList(means, type='StandardScaler'):#'MinMaxScaler'
#
#     from sklearn import preprocessing
#
#     if type == 'MinMaxScaler':
#         vScaler = preprocessing.MinMaxScaler()
#         v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
#     if type == 'StandardScaler':
#         vScaler = preprocessing.StandardScaler()
#         v_normalized = vScaler.fit_transform(np.array(means).reshape(-1, 1)).reshape(1, -1)
#
#     return v_normalized.tolist()[0]
