from RawData import RawData
from Data import *
from FilteredData import *
from PrepData import *
from Dashboard import *
from GridSearch import *
from Helpers import *

"""
------------------------------------------------------------------------------------------------------------------------
1.DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Save input"""
saveInput(csvPath, outputPath, displayParams, xQualLabels, xQuantLabels, yLabels, processingParams, modelingParams,
          powers, mixVariables)

"""Import libraries & Load data"""
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data & One hot encoding"""
dat = Data(rdat)
df = dat.asDataframe(powers)

baseLabels = xQuantLabels # if higher orders : ['GIFA (m2)_exp1', 'Storeys_exp1', 'Typical Span (m)_exp1','Typ Qk (kN_per_m2)_exp1']

""" Remove outliers"""
ValidDf = removeOutliers(df, labels = xQuantLabels, cutOffThreshhold=processingParams['cutOffThreshhold'])

"""Correlation of variables & Feature selection"""
HighCorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = processingParams['lowThreshold'])
#
"""Remove Multi-correlated Features """
CorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = processingParams['lowThreshold'],
                     removeLabels=processingParams['removeLabels'])

"""Scale"""
xSets, ySets, xScaler = TrainTestSets(CorDf, yLabels, processingParams['scaler'])

"""Split"""
(xTrain, yTrain), (xTest, yTest) = TrainTestDf(xSets, ySets, testIdParam=1)
(xTrainArr, yTrainArr), (xTestArr, yTestArr) = (xTrain.values, yTrain.values.reshape(-1, 1)), (xTest.values, yTest.values.reshape(-1, 1))

"""Save Data Processing"""
trackDataProcessing(displayParams = displayParams, df = df, noOutlierdf = ValidDf, filterdf=HighCorDf , removeLabelsdf = CorDf)

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
searchEval(modelingParams, displayParams, models, xTrainArr, yTrainArr, xTestArr, yTestArr)

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

