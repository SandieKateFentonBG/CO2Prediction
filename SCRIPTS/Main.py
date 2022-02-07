from RawData import RawData
from Data import *
from FilteredData import *
from PrepData import *
from Dashboard import *
from GridSearch import *

"""
------------------------------------------------------------------------------------------------------------------------
1.DATA
------------------------------------------------------------------------------------------------------------------------
"""

"""Import libraries & Load data"""
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data & One hot encoding"""
dat = Data(rdat)
df = dat.asDataframe(powers)

baseLabels = xQuantLabels # if higher orders : ['GIFA (m2)_exp1', 'Storeys_exp1', 'Typical Span (m)_exp1','Typ Qk (kN_per_m2)_exp1']

""" Remove outliers"""
ValidDf = removeOutliers(df, labels = xQuantLabels, cutOffThreshhold=3)

# """Correlation of variables & Feature selection"""
HighCorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = 0.1)
#
# """Remove Multi-correlated Features """
CorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = 0.1,
                     removeLabels=['Basement_None', 'Foundations_Raft'])

trackDataProcessing(df, ValidDf, CorDf)

"""Scale Data"""#todo : check this..

(ScaledDf, Scaler) = normalize(CorDf)
UnscaledDf = unscale(ScaledDf, Scaler)

"""Split"""

# xSets, ySets = TrainTestSets(CorDf, yLabels)
xSets, ySets = TrainTestSets(ScaledDf, yLabels)
(xTrain, yTrain), (xTest, yTest) = TrainTestDf(xSets, ySets, testSetIndex=1)
(xTrainArr, yTrainArr), (xTestArr, yTestArr) = (xTrain.values, yTrain.values.reshape(-1, 1)), (xTest.values, yTest.values.reshape(-1, 1))

"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""

# linearReg = {'model' : LinearRegression(), 'param' : 'alpha'}
lassoReg = {'model' : Lasso() , 'param': 'alpha'} # for overfitting
ridgeReg = {'model' : Ridge(), 'param': 'alpha'}
elasticNetReg = {'model' : ElasticNet(), 'param': 'alpha'}
supportVector = {'model' : SVR(), 'param': 'C'}
kernelRidgeReg = {'model' : KernelRidge(), 'param': 'alpha'}
kernelRidgeLinReg = {'model' : KernelRidge(kernel='linear'), 'param': 'alpha'}
kernelRidgeRbfReg = {'model' : KernelRidge(kernel='rbf'), 'param': 'alpha'}
kernelRidgePolReg = {'model' : KernelRidge(kernel='polynomial'), 'param': 'alpha'}
models = [lassoReg, ridgeReg, elasticNetReg, supportVector, kernelRidgeReg, kernelRidgeLinReg, kernelRidgeRbfReg, kernelRidgePolReg] #linearReg,

"""
------------------------------------------------------------------------------------------------------------------------
3. HYPERPARAM GRID SEARCH
------------------------------------------------------------------------------------------------------------------------
"""
# kernelVal = ['linear', 'rbf', 'polynomial']
# paramVal = [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]
# grid = GridSearch()

# m = {'model' : SVR(), 'param' : 'C'}
# bestModel = grid.searchCV(m['model'], m['param'], paramVal, xTrainArr, yTrainArr, xTestArr, yTestArr)

# bestModel = grid.searchCV(SVR(), parameters, xTrainArr, yTrainArr, xTestArr, yTestArr)

# def tuneModels(modelingParams, displayParams, models):
#     # store = dict()
#     for m in models:
#         grid = GridSearch()
#         bestModel = grid.paramSearch(m['model'], m['param'], modelingParams['RegulVal'], xTrainArr, yTrainArr)
#         # store[m['model']] = bestModel
#         # m['bestModel'] = bestModel(bestModel)
#     #     m['bestModel'] = bestModel.best_estimator_
#     # for m in models:
#         m['bestParam'] = bestModel.best_params_
#         model, accuracy, mse, r2 = grid.paramEval(m['model'], xTrainArr, yTrainArr, xTestArr, yTestArr, displayParams)
#         m['accuracy'] = accuracy
#         m['mse'] = mse
#         m['r2'] = r2
#         print(m)
#     if displayParams["archive"]:
#         saveStudy(displayParams, models)

searchEval(modelingParams, displayParams, models, xTrainArr, yTrainArr, xTestArr, yTestArr)

# tuneModels(modelingParams, displayParams, models)
# print('bestModel', bestModel)

#3.2.5.1. Model specific cross-validation : https://scikit-learn.org/stable/modules/grid_search.html#alternative-cv

"""
------------------------------------------------------------------------------------------------------------------------
3. Plot
------------------------------------------------------------------------------------------------------------------------
"""

# bestModel = grid.run(SVR(C = 1), xTrainArr, yTrainArr, xTestArr, yTestArr, display = True, tolerance=0.05)

# for m in models:
#     model, accuracy, mse = grid.run(m['model'], xTrainArr, yTrainArr, xTestArr, yTestArr, displayParams)
#     m['accuracy'] = accuracy
#     m['mse'] = mse
#
# for m in models:
#     print(m)


#Accuracy : 'accuracy': 0.07142857142857142 means 1 good out of 14
#todo : for a wider variety of params : sklearn.model_selection.ParameterGrid
#todo : about tuning hyperparams :https://scikit-learn.org/stable/modules/grid_search.html -
#todo : read abput kernels/rbf : sklearn.kernel_ridge.KernelRidge¶
#todo : about comparing results Statistical comparison of models using grid search - displaying order
#todo : saving function

#todo: ASK feature selection - how much? forward or backward stepwise search?
#todo: ASK normalize/unnormalize - what? when?
#todo: ASK what regression models - what parameters - SOA
#todo: ASK understand scoring metrics - what are negative values ? why do they not coincide? (test vs train maybe? but very different vallues..)
#todo: ASK tolerance - accuracy / custom score - can only have 2 input params?




# from sklearn.svm import SVR
#
# parameters = {'C': [1, 2, 4]}
# model=SVR()
#
# a = searchCV(filterDf, yLabels, model=model, parameters=parameters)

# #single run
# method = 'Nmodel' #'LRmodel', 'RFmodel','SVMmodel', 'LRmodel'
# run = execute(filterDf, yLabels, method, epochs=5, singleTest = 1, display = True)
# #
# # #multiple run
# # methods = ['LRmodel', 'SVMmodel', 'RFmodel', 'XGBmodel']

# # for m in methods:
# #     run = execute(filterDf2, yLabels, m, epochs=None, singleTest=1, display = False)
# #     print('Method:', run['method'], 'Evaluation:', run['evalu'], 'Accuracy:', run['acc'],'MSE:', run['mse'] )
# #     # plot(run['yTest'],run['model'].predict(run['xTest']))
# #todo : normal model not working
#
# # #todo: look into lasso/ ridge - these also allow for feature selection -
# # # modularize feature selection either earlier with pearson or later with lasso
# # #todo: look into constrained optim
# # #todo: understand model saving
# #todo : add higher powers/power up
# #todo: support vector regression, kernel regression, lasso regression and fully connected neural networks
# #
# # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# # https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html

