from RawData import RawData
from Data import *
from FilteredData import *
from PrepData import *
from Dashboard import *
# from Dashboard_V2 import *
from GridSearch import *
from Archiver import *
from Model import *


"""
------------------------------------------------------------------------------------------------------------------------
0.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Save input"""
content = saveInput(csvPath, outputPath, displayParams, xQualLabels, xQuantLabels, yLabels, processingParams, modelingParams,
          powers, mixVariables)

"""Import libraries & Load data"""
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data & One hot encoding"""
dat = Data(rdat)
df = dat.asDataframe(powers)

""" Remove outliers - only exist/removed on Quantitative features"""
ValidDf = removeOutliers(df, labels = xQuantLabels, cutOffThreshhold=processingParams['cutOffThreshhold'])

"""
------------------------------------------------------------------------------------------------------------------------
1.DATA
------------------------------------------------------------------------------------------------------------------------
"""
baseLabels = xQuantLabels
# if higher orders :
# baseLabels = ['GIFA (m2)_exp1', 'Storeys_exp1', 'Typical Span (m)_exp1','Typ Qk (kN_per_m2)_exp1']
#  #

"""Correlation of variables & Feature selection"""
HighCorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = processingParams['lowThreshold'])
#
"""Remove Multi-correlated Features """
CorDf = filteredData(ValidDf, baseLabels, yLabels, plot = False, lt = processingParams['lowThreshold'],
                     removeLabels=processingParams['removeLabels'])
"""Scale"""
xdf, ydf, xScaler = XScaleYSplit(CorDf, yLabels, processingParams['scaler'])

#import statsmodels.api as sm
#xdf1 = sm.add_constant(xdf) #todo : add constant?

"""Train Test Split"""
xTrain, xTest, yTrain, yTest = TrainTest(xdf, ydf, test_size=0.2, random_state=8)

"""Save Data Processing"""
trackDataProcessing(displayParams = displayParams, df = df, noOutlierdf = ValidDf, filterdf=HighCorDf , removeLabelsdf = CorDf)

"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""




"""
------------------------------------------------------------------------------------------------------------------------
3. HYPERPARAM GRID SEARCH - LOOK FOR BEST REGUL
------------------------------------------------------------------------------------------------------------------------
"""

"""
------------------------------------------------------------------------------------------------------------------------
3. HYPERPARAM GRID SEARCH - LOOK AT WEIGHTS
------------------------------------------------------------------------------------------------------------------------
"""

searchedModels = searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features = list(xdf.keys()))
for m in searchedModels:
    for k, v in m.items():
        print (k, ':', v)

#
#
# for m in searchedModels :
#     features, weights = m['features'], m['bModelWeights']
#     print(modelWeightsDict(df, displayParams['Target'], features, weights, 0))
#     print(modelWeightsList(df, displayParams['Target'], features, weights, 0))

# for m in searchedModels:
#     for k, v in zip(m['features'], m['weightVals']):
#         print (k, ':', v)

exportStudy(displayParams, searchedModels, content)

# paramResiduals(Ridge(), xTrain, yTrain, xTest, yTest, displayParams, bestParam = None)
#
# import statsmodels.api as sm
#
# xdf1 = sm.add_constant(xdf)
# """Train Test Split"""
# xTrain1, xTest1, yTrain1, yTest1 = TrainTest(xdf1, ydf, test_size=0.2, random_state=8)
#
# """Save Data Processing"""
# searchedModels = searchEval(modelingParams, displayParams, models, xTrain1, yTrain1, xTest1, yTest1)
# print(searchedModels)
# exportStudy(displayParams, searchedModels)
# """
# ------------------------------------------------------------------------------------------------------------------------
# 3. Plot
# ------------------------------------------------------------------------------------------------------------------------
# """
#              # Finalize and render the figure

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

# todo : good display https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv