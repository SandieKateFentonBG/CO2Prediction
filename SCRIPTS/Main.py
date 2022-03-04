from RawData import RawData
from Data import *
from PrepData import *
from Dashboard_PMv1 import *
# from Dashboard_PMv2 import *
from GridSearch import *
from Archiver import *
from PlotRegul import *
from PlotWeights import *
from PlotSearch import *
from Visualizers import *

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
ValidDf = removeOutliers(df, labels = xQuantLabels+yLabels, cutOffThreshhold=processingParams['cutOffThreshhold'])

"""
------------------------------------------------------------------------------------------------------------------------
2.DATA
------------------------------------------------------------------------------------------------------------------------
"""

"""Correlation of variables & Feature selection"""
NoFilterDf, _ = filteredData(ValidDf, processingParams['baseLabels'], yLabels, displayParams, lt=0)

HighCorDf, _ = filteredData(NoFilterDf, processingParams['baseLabels'], yLabels, displayParams, lt=processingParams['lowThreshold'])
checkDf, _ = filteredData(HighCorDf, processingParams['baseLabels'], yLabels, displayParams, lt=processingParams['lowThreshold'], checkup = True)

"""Remove Multi-correlated Features """
CorDf, prepData = filteredData(ValidDf, processingParams['baseLabels'], yLabels, displayParams, lt=processingParams['lowThreshold'],
                     removeLabels=processingParams['removeLabels'])
"""Scale"""
# xdf, ydf, xScaler = XScaleYSplit(CorDf, yLabels, processingParams['scaler'])

xdf, xScaler, ydf, yScaler = XScaleYScaleSplit(CorDf, yLabels, processingParams['scaler'], yScale = True)

print(np.array(xdf))
print(np.array(ydf))
print('x', xScaler)
print('y', yScaler)

unscaley = unscale(ydf, yScaler, processingParams['scaler'])

print(np.array(unscaley))
"""Train Test Split"""
xTrain, xTest, yTrain, yTest = TrainTest(xdf, ydf, test_size=modelingParams['test_size'], random_state=modelingParams['random_state'])
#
"""Save Data Processing"""
trackDataProcessing(displayParams=displayParams, df=df, noOutlierdf=ValidDf, filterdf=HighCorDf, removeLabelsdf=CorDf)

"""
------------------------------------------------------------------------------------------------------------------------
3. MODEL
------------------------------------------------------------------------------------------------------------------------
"""
#
# """Search"""
# searchedModels = searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features=list(xdf.keys()))
# sortedDc = sortGridResults(searchedModels, metric = 'bModelAcc', highest = True)
# """Save & Dump"""
# exportStudy(displayParams, inputData, prepData, searchedModels, sortedDc)
# pickleDumpMe(displayParams, searchedModels)
#
# dc = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"], name = '/Records', show = False)
# sortedDc = sortGridResults(dc, metric = 'bModelAcc', highest = True)

"""
------------------------------------------------------------------------------------------------------------------------
4. RESULTS
------------------------------------------------------------------------------------------------------------------------
"""
"""Regularization Influence"""
# WeightsBarplotAll(dc, displayParams)
# WeightsSummaryPlot(dc, displayParams, sorted=True, yLim=None)
#
# plotRegul3D(dc, displayParams, modelingParams, lims = True, ticks = True)
# plotRegul2D(dc, displayParams, modelingParams,)
# plotRegul3D(dc, displayParams, modelingParams, lims = True, log = True)
# plotRegul2D(dc, displayParams, modelingParams, log = True)
#


# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html

# todo : understand advantage of ridge regression - istructe talk
#
#  good display https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv

#One way to combat heteroscedasticity is through Weighted Least Squares


# todo : use other database?
# todo : understand residual plot
# todo : understand LASSO

