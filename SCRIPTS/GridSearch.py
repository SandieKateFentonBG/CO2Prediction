from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from Helpers import *
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

def plotPredTruth(yTest, yPred, displayParams, modelWithParam):
    import matplotlib.pyplot as plt
    plt.rcParams['figure.figsize'] = [18, 18]
    l1, = plt.plot(yTest, 'g')
    l2, = plt.plot(yPred, 'r', alpha=0.7)
    plt.legend(['Ground truth', 'Predicted'])
    plt.title(str(modelWithParam))

    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/Pred_Truth'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + str(modelWithParam) + '.png')
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

class GridSearch:

    """
    Goal - for each model
    - build it
    - run it with different parameters
    - store the best parameter and score
    """

    def __init__(self):
        self.linearReg = LinearRegression()
        self.lassoReg = Lasso() #for overfitting
        self.ridgeReg = Ridge()
        self.elasticNetReg = ElasticNet()
        self.supportVector = SVR()
        self.kernelRidgeReg = KernelRidge() #for underfitting
        #self.normalModel = buildNormalModel()

    def update(self, modelName, model, bestParameters):
        update = model(bestParameters)
        self.modelName = update
        # if modelName == 'linearReg':
        #     self.linearReg = update
        # if modelName == 'lassoReg':
        #     self.lassoReg = update
        # if modelName == 'ridgeReg':
        #     self.ridgeReg = update
        # if modelName == 'elasticNetReg':
        #     self.elasticNetReg = update
        # if modelName == 'supportVector':
        #     self.supportVector = update
        # if modelName == 'kernelRidgeReg':
        #     self.kernelRidgeReg = update
          #todo : how to do this in a generic way?
        pass

def paramSearch(model, paramkey, paramValues, xTrain, yTrain, custom = False):

    parameters = dict()
    parameters[paramkey] = paramValues
    if custom:
        score = make_scorer(computeAccuracy(), greater_is_better=True)
        grid = GridSearchCV(model, scoring=score, param_grid=parameters)

    else:
        grid = GridSearchCV(model, param_grid=parameters)
    grid.fit(xTrain, yTrain.ravel())
    return grid

def paramEval(modelWithParam, xTrain, yTrain, xTest, yTest, displayParams):

    clf = modelWithParam
    clf.fit(xTrain, yTrain.ravel())
    scores = clf.score(xTest, yTest.ravel())
    yPred = clf.predict(xTest)
    accuracy = computeAccuracy(yTest, clf.predict(xTest))
    mse = mean_squared_error(yTest, clf.predict(xTest))
    r2 = r2_score(yTest, clf.predict(xTest))
    if displayParams['showPlot']:
        plotPredTruth(yTest, yPred, displayParams, modelWithParam)

    return clf, accuracy, mse, r2

def searchEval(modelingParams, displayParams, models, xTrainArr, yTrainArr, xTestArr, yTestArr):

    for m in models:
        bestModel = paramSearch(m['model'], m['param'], modelingParams['RegulVal'], xTrainArr, yTrainArr)
        m['bestParam'] = bestModel.best_params_
        model, accuracy, mse, r2 = paramEval(m['model'], xTrainArr, yTrainArr, xTestArr, yTestArr,
                                                  displayParams)
        m['accuracy'] = round(accuracy, 3)
        m['mse'] = round(mse, 3)
        m['r2'] = round(r2, 3)
        if displayParams["showResults"]:
            print(m)
    if displayParams["archive"] or displayParams["showPlot"]:
        saveStudy(displayParams, models)

    return models