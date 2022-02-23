from RawData import RawData
from Data import *
from PrepData import *
from Dashboard import *
# from Dashboard_V2 import *
from GridSearch import *
from Archiver import *
from PlotWeightsl import *


"""
------------------------------------------------------------------------------------------------------------------------
1.RAW DATA
------------------------------------------------------------------------------------------------------------------------
"""
"""Import libraries & Load data"""
inputData = saveInput(csvPath, outputPath, displayParams, xQualLabels, xQuantLabels, yLabels, processingParams, modelingParams,
          powers, mixVariables)
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data & One hot encoding"""
dat = Data(rdat)
df = dat.asDataframe(powers)

""" Remove outliers - only exist/removed on Quantitative features"""
ValidDf = removeOutliers(df, labels = xQuantLabels, cutOffThreshhold=processingParams['cutOffThreshhold'])

"""
------------------------------------------------------------------------------------------------------------------------
2.DATA
------------------------------------------------------------------------------------------------------------------------
"""

"""Correlation of variables & Feature selection"""
HighCorDf, _ = filteredData(ValidDf, baseLabels, yLabels, displayParams, lt=processingParams['lowThreshold'])
#
"""Remove Multi-correlated Features """
CorDf, prepData = filteredData(ValidDf, baseLabels, yLabels, displayParams, lt=processingParams['lowThreshold'],
                     removeLabels=processingParams['removeLabels'])
"""Scale"""
xdf, ydf, xScaler = XScaleYSplit(CorDf, yLabels, processingParams['scaler'])

"""Train Test Split"""
xTrain, xTest, yTrain, yTest = TrainTest(xdf, ydf, test_size=modelingParams['test_size'], random_state=modelingParams['random_state'])

"""Save Data Processing"""
trackDataProcessing(displayParams=displayParams, df=df, noOutlierdf=ValidDf, filterdf=HighCorDf, removeLabelsdf=CorDf)

"""
------------------------------------------------------------------------------------------------------------------------
3. MODEL 
------------------------------------------------------------------------------------------------------------------------
"""
print(models)
"""Search"""
searchedModels = searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features=list(xdf.keys()))

"""Save"""
exportStudy(displayParams, inputData, prepData, searchedModels)

"""
------------------------------------------------------------------------------------------------------------------------
3. RESULTS
------------------------------------------------------------------------------------------------------------------------
"""


#sklearn.model_selection.ParameterGrid
#https://scikit-learn.org/stable/modules/grid_search.html -
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html

# todo : good display https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv