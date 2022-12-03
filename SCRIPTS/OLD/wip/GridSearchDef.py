
from Main import *

x, y, xlabels = dat.asArray()

"""
With default scorer
"""
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

parameters = {"C": [1e0, 1e1, 1e2, 1e3]}
estimator = SVR()
clf = GridSearchCV(estimator, param_grid=parameters) #scoring = score
clf.fit(x, y.ravel())

print(sorted(clf.cv_results_.keys()))
print(clf.cv_results_)
print(clf.score(x, y))

"""
With custom scorer
"""
from sklearn.metrics import make_scorer
def computeAccuracy(yTrue, yPred, tolerance = 0.05): #thos could be done unscaled
    #https: // scikit - learn.org / stable / modules / model_evaluation.html  # scoring
    #todo : sklearn.metrics.accuracy_score - but this requires exacte prediction : ypred = ytrue

    validated = [1 if abs(yPred[i] - yTrue[i]) < abs(yTrue[i]) * tolerance else 0 for i in range(len(yTrue))]
    return sum(validated) / len(validated)

score = make_scorer(computeAccuracy, greater_is_better=True)
# scoring = {'accuracy': make_scorer(computeAccuracy, greater_is_better=True), 'prec': 'precision'}
parameters = {"C": [1e0, 1e1, 1e2, 1e3]}
estimator = SVR()
clf2 = GridSearchCV(estimator, scoring=score, param_grid=parameters) #scoring = score
clf2.fit(x, y.ravel())

computeAccuracy(y, clf2.predict(x))
score(clf2, x, y)
print(sorted(clf2.cv_results_.keys()))
print(clf2.cv_results_)
print(score(clf2, x, y))

"""
With Train-test split
"""
TrainSize = 60
# clf2.fit(x[TrainSize:], y.ravel()[TrainSize:])
# print(clf2.best_params_)
# print(clf2.score(x[TrainSize:], y[TrainSize:]))
# def runGridSearch(model, param, scoring=myacc):
#     mods = GridSearchCV(model, param)
#     mods.fit()

# def my_custom_loss_func(y_true, y_pred):
#     diff = np.abs(y_true - y_pred).max()
#     return np.log1p(diff)


#
#
#