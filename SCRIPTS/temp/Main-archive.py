from RawData import RawData
from SCRIPTS.Features import *
from SCRIPTS.temp.PrepData import *
from Dashboard_PMv1 import *
# from Dashboard_PMv2 import *
from SCRIPTS.temp.Archiver import *

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
dat = Features(rdat)
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

xdf, xScaler, ydf, yScaler = XScaleYScaleSplit(CorDf, yLabels, processingParams['scaler'],
                                               processingParams['yScale'], processingParams['yUnit'])


"""Train Test Split"""
xTrain, xTest, yTrain, yTest = TrainTest(xdf, ydf, test_size=modelingParams['test_size'], random_state=modelingParams['random_state'])

"""Save Data Processing"""
trackDataProcessing(displayParams=displayParams, df=df, noOutlierdf=ValidDf, filterdf=HighCorDf, removeLabelsdf=CorDf)

# """
# ------------------------------------------------------------------------------------------------------------------------
# 3. MODEL
# ------------------------------------------------------------------------------------------------------------------------
# """
#
# """Search"""
# searchedModels = searchEval(modelingParams, displayParams, models, xTrain, yTrain, xTest, yTest, features=list(xdf.keys()),
#                             resPlot=True, restDist=True)
# sortedDc = sortGridResults(searchedModels, metric = 'bModelAcc', highest = True)
# """Save & Dump"""
# exportStudy(displayParams, inputData, prepData, searchedModels, sortedDc)
# pickleDumpMe(displayParams, searchedModels)
#
# dc = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']), name = '/Records', show = False)
# sortedDc = sortGridResults(dc, metric = 'bModelAcc', highest = True)
#
# """
# ------------------------------------------------------------------------------------------------------------------------
# 4. RESULTS
# ------------------------------------------------------------------------------------------------------------------------
# """
#
# """Regularization Influence"""
#
# WeightsBarplotAll(dc, displayParams, yLim=None)
# WeightsSummaryPlot(dc, displayParams, sorted=True, yLim=None)
#
# """Regularization Influence"""
#
# plotRegul3D(dc, displayParams, modelingParams, lims = True, log = True)
# plotRegul2D(dc, displayParams, modelingParams, log = True)
#
# dc_ = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) , name = '/Records', show = False)

"""Residuals """
# ? where is random used > for train / test split > when we assemble residuals, are we re-testing on xamples used for training?
# dc_a = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"] + '4' , name = '/Records', show = False)
# dc_b = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"] + '5' , name = '/Records', show = False)
# dc_c = pickleLoadMe(displayParams["outputPath"] + displayParams["reference"] + '3' , name = '/Records', show = False)
#
# studies = [dc_a, dc_b, dc_c]
#
# # WeightsSummaryPlot(dc_a, displayParams, sorted=True, yLim=None, fontsize =14)
#
# sortedMod = sortGridResults(dc_a, metric='bModelAcc', highest=True) #crap
# slice = sortedMod[0:5]
#
# print(len(dc_a), type(dc_a))
# print([elem for elem in dc_a])
# # MetricsSummaryPlot(sortedMod, displayParams, metricLabels=['bModelAcc'])
# predTruthCombined(displayParams, slice, xTest, yTest, Train=False, scatter = True)

# plotScaleResDistribution(studies, displayParams)
#
# residualsMeanVar = plotResHistGauss(studies, displayParams, binwidth = 10, setxLim =(-300, 300))# (-150, 150)


# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
#
#  good display https://stackoverflow.com/questions/37161563/how-to-graph-grid-scores-from-gridsearchcv




#
#
# C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\sklearn\linear_model\_coordinate_descent.py:529: ConvergenceWarning: Objective did not converge. You might want to increase the number of iterations. Duality gap: 15323.542520435309, tolerance: 17.256706976744187
#   model = cd_fast.enet_coordinate_descent(