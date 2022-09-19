#https://stackoverflow.com/questions/61142862/recursive-feature-elimination-and-grid-search-for-svr-using-scikit-learn
#https://scikit-learn.org/stable/modules/grid_search.html
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV


from Main import *
from sklearn.svm import SVR
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV

estimator=SVR(kernel='linear')
selector = RFECV(estimator, step = 1, cv = 5)

gsc = GridSearchCV(selector, param_grid={
        'estimator__C': [0.1, 1, 100, 1000],
        'estimator__epsilon': [0.0001, 0.0005],
        'estimator__gamma': [0.0001, 0.001]},

    cv=5, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)

grid_result = gsc.fit(xTrain, yTrain.ravel())

trainScore = gsc.score(xTrain, yTrain.ravel())
testScore = gsc.score(xTest, yTest.ravel())
yPred = gsc.predict(xTest).reshape(-1, 1)


residual = yTest - yPred

accuracy = computeAccuracy(yTest, gsc.predict(xTest), modelingParams['accuracyTol'])
mse = mean_squared_error(yTest, gsc.predict(xTest))
r2 = r2_score(yTest, gsc.predict(xTest))

print()
print("residual", residual)
print("accuracy", accuracy)
print("mse", mse)
print("r2", r2)

print(gsc.cv_results_)
print(gsc.n_features_in_)
print(gsc.feature_names_in_)
