from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LinearRegression
import pandas as pd

rs = 42
RFEEstimators = {'LinearRegression': LinearRegression(),
                 'DecisionTreeRegressor': DecisionTreeRegressor(random_state = rs),
                  'RandomForestRegressor' : RandomForestRegressor(random_state = rs),
                  'GradientBoostingRegressor' : GradientBoostingRegressor(random_state = rs),
                  'DecisionTreeClassifier' : DecisionTreeClassifier(random_state = rs)}

step = 1 #features removed at every itteration
scoring="r2"
n_features_to_select = 10
cv = KFold(n_splits=5, shuffle=True, random_state=rs) #cross-validation splitting - ensures same split for every function run
loo = LeaveOneOut() #alternative to cv TODO : check this
featureCount = [5, 10, 15]

"""
Questions
1. Trainscore is too igh - What should i evaluate my rfe on? validation set? > what does the RFECV use as a score? 
2. Scaling, When should it be done? Look into pipeline
3. I inserted a random-state to have stable results - is this ok?
4. leave one out?
"""

def RFECVGridsearch(RFEEstimators, xTrain, yTrain, step, cv, scoring , display = False, testTuple = None):
    """


    Look optimal number of features
    (or, the best combination of n features, given their importance for the wrapped estimator)

    :param RFEEstimators:
    :param xTrain: Input samples - array-like of shape (n_samples, n_features) (xTrain.to_numpy())
    :param yTrain: Target values - array-like of shape (n_samples,) (yTrain.to_numpy()) or (n_samples, n_outputs) (yTrain.to_numpy().ravel())
    :param step:
    :param cv:
    :param scoring:
    :param display:
    :param testTuple: (xTest, yTest) - insert these if you want to test rfe on testing, else None
    :return:
    """

    rfecvDict = dict()

    for k,estimator in RFEEstimators.items():

        rfecvDict[k] = dict()

        # TODO : pipeline - insert scaling here?
        rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=scoring)
        rfecv = rfecv.fit(xTrain.to_numpy(), yTrain.to_numpy().ravel())

        rfecvDict[k]['model'] = rfecv

        if testTuple:
            xTest, yTest = testTuple
            score = rfecv.score(xTest.to_numpy(), yTest.to_numpy().ravel())
            rfecvDict[k]['Test Score'] = score

        if display :
            print("RFECV Gridsearch:")
            print("Estimator:", k)
            print("Number of features:", rfecv.n_features_)
            # score function reduces x to selected features and returns score
            print("CV Score on training", rfecv.score(xTrain.to_numpy(), yTrain.to_numpy().ravel()))
            #print('Selected feature labels', xTrain.columns[rfecv.support_])
            if testTuple:
                print("Score on testing", score)
            print("")

    return rfecvDict

def RFEGridsearch(RFEEstimators,n_features_to_select, xTrain, yTrain, display = False, testTuple = None) :

    """
    Look for n most important features
    (or, the best combination of n features, given their importance for the wrapped estimator)
    """

    rfeDict = dict()

    for k,estimator in RFEEstimators.items():

        rfeDict[k] = dict()
        rfe = RFE(estimator, n_features_to_select=n_features_to_select)
        rfe = rfe.fit(xTrain.to_numpy(), yTrain.to_numpy().ravel())

        rfeDict[k]['model'] = rfe

        if testTuple:
            xTest, yTest = testTuple
            score = rfe.score(xTest.to_numpy(), yTest.to_numpy().ravel())
            # rfeDict_Test[k] = score
            rfeDict[k]['Test Score'] = score

        if display :
            print("RFE Gridsearch:")
            print("Estimator:", k)
            print("Number of features:", n_features_to_select)
            print("Score on training", rfe.score(xTrain.to_numpy(), yTrain.to_numpy().ravel()))
            print('Selected feature labels', xTrain.columns[rfe.support_])
            if testTuple:
                print("Score on testing", score)
            print("")

    # return rfeDict, rfeDict_Test
    return rfeDict

def RFEHyperparameterSearch(RFEEstimators,featureCount, xTrain, yTrain, display = False, testTuple = None):

    """
    Look for number of features - by selecting the n most important features
    (or, the best combination of n features, given their importance for the wrapped estimator)
    :param RFEEstimators:
    :param featureCount:
    :param xTrain:
    :param yTrain:
    :param display:
    :param testTuple:
    :return:
    """
    paramDict = dict()

    for k in RFEEstimators.keys():

        paramDict[k] = dict()
        paramDict[k]['featureCount'] = featureCount  # [featureList, trainScore, testScore]
        paramDict[k]['Train Score'] = []
        paramDict[k]['Test Score'] = []

    for f in featureCount:
        rfeDict = RFEGridsearch(RFEEstimators, f, xTrain, yTrain, testTuple=testTuple)

        for k in RFEEstimators.keys():
            paramDict[k]['Train Score'].append(rfeDict[k]['model'].score(xTrain.to_numpy(), yTrain.to_numpy().ravel()))
            if testTuple:
                paramDict[k]['Test Score'].append(rfeDict[k]['Test Score'])
    if display:
        for k, v in paramDict.items():
            print(k)
            print(v)

    return paramDict

def WrapperLabels(rfeDict):
    RFELabelsDict = dict()
    for k in rfeDict.keys():
        RFELabelsDict[k] = rfeDict[k]['model'].support_
    return RFELabelsDict

def EliminateDf(xtrainDf, xvalidDf, xtestDf, ytrainDf, yvalidDf, ytestDf, rfeDict):

    RFEDf = dict()
    for k in rfeDict.keys():

        RFETrainDf = xtrainDf.columns[rfeDict[k]['model'].support_]
        RFEValidDf = xvalidDf.columns[rfeDict[k]['model'].support_]
        RFETestDf = xtestDf.columns[rfeDict[k]['model'].support_]

        #todo : fix train df concatenated from xtrain and y train

        # RFETrainDf = pd.concat([RFETrainDf, ytrainDf], axis=1)
        # RFEValidDf = pd.concat([RFEValidDf, yvalidDf], axis=1)
        # RFETestDf = pd.concat([RFETestDf, ytestDf], axis=1)


        RFEDf[k] = [RFETrainDf, RFEValidDf, RFETestDf]

    return RFEDf


# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html
# https://blog.datadive.net/selecting-good-features-part-iv-stability-selection-rfe-and-everything-side-by-side/
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html
#https://machinelearningmastery.com/rfe-feature-selection-in-python/
#https://www.section.io/engineering-education/recursive-feature-elimination/
#https://stackoverflow.com/questions/46062679/right-order-of-doing-feature-selection-pca-and-normalization