from sklearn.feature_selection import RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

CoreEstimators = [DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), DecisionTreeClassifier()]
step = 1 #features removed at every itteration
cv = 5 #cross-validation splitting strategy
scoring="r2"
n_features_to_select = 25

#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html

def WrapDataCV(estimator, xTrain, yTrain, step, cv, scoring):

    rfecv = RFECV(estimator= estimator, step = step, cv = cv, scoring=scoring)
    rfecv = rfecv.fit(xTrain.to_numpy(), yTrain.to_numpy().ravel())
    trainScore = rfecv.score(xTrain.to_numpy(), yTrain.to_numpy().ravel())

    print("The optimal number of features:", rfecv.n_features_)
    print("Best features:", xTrain.columns[rfecv.support_])
    print("trainScore", trainScore)

    return rfecv


from sklearn.feature_selection import RFE

# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html

def WrapData(estimator, xTrain, yTrain, n_features_to_select):

    # Init the transformer
    rfe = RFE(estimator, n_features_to_select=n_features_to_select)
    # Fit to the training data
    rfe = rfe.fit(xTrain.to_numpy(), yTrain.to_numpy().ravel())
    trainScore = rfe.score(xTrain.to_numpy(), yTrain.to_numpy().ravel())

    print("The optimal number of features:", rfe.n_features_)
    print("Best features:", xTrain.columns[rfe.support_])
    print("trainScore", trainScore)

    return rfe