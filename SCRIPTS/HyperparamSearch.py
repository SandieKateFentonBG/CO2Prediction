# X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
#
# est = SVR(kernel="linear")
#
# std_scaler = preprocessing.StandardScaler()
# selector = feature_selection.RFE(est)
# pipe_params = [('feat_selection',selector),('std_scaler', std_scaler), ('clf', est)]
# pipe = pipeline.Pipeline(pipe_params)
#
# param_grid = dict(clf__C=[0.1, 1, 10])
# clf = GridSearchCV(pipe, param_grid=param_grid, cv=2)
# clf.fit(X, y)
# print clf.grid_scores_

"""
# >>> estimator = SVR(kernel="linear")
# >>> selector = RFE(estimator, n_features_to_select=0.5, step=1)
# >>> selector = selector.fit(X, y)
"""

from ModelsDoc import *
from sklearn import feature_selection
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

# estimator=predictors[4]['model']
# selector = feature_selection.RFE(estimator, n_features_to_select=0.5, step=1)

def computeAccuracy(yTrue, yPred, tolerance):
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

def paramEval(predictors, xTrain, yTrain, custom = False, cv = None, refit ='r2', rounding = 3):


    parameters = dict()
    if predictors['param']:
        for k,v in predictors['param'].items():
            parameters[k] = v
    print('here', parameters)
    if custom:
        score = make_scorer(computeAccuracy(), greater_is_better=True)
        grid = GridSearchCV(predictors['model'], scoring=score, param_grid=parameters, cv = cv)
    else:
        scoring = {'neg_mean_squared_error': 'neg_mean_squared_error', 'r2': 'r2'}
        grid = GridSearchCV(predictors['model'], param_grid=parameters, scoring =scoring, refit = refit, cv = cv, return_train_score=True)

    grid.fit(xTrain, yTrain)
    # grid.fit(xTrain, yTrain.ravel()) #TODO - understand this RAVEL - Is it needed?
    results =grid.cv_results_

    paramDict = {'paramMeanMSETest': [round(num, rounding) for num in list(grid.cv_results_['mean_test_neg_mean_squared_error'])],
                 'paramStdSMSETest': [round(num, rounding) for num in list(grid.cv_results_['std_test_neg_mean_squared_error'])],
                 'paramRankMSETest': list(grid.cv_results_['rank_test_neg_mean_squared_error']),
                 'paramMeanR2Test': [round(num, rounding) for num in list(grid.cv_results_['mean_test_r2'])],
                 'paramStdR2Test': [round(num, rounding) for num in list(grid.cv_results_['std_test_r2'])],
                 'paramRankR2Test': list(grid.cv_results_['rank_test_r2']),
                 'paramValues': predictors['param']}

    return grid, paramDict