
import pandas as pd
from Archiver import *

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def filteredData(noOutlierDf, baseLabels, yLabels, displayParams, lt, removeLabels = None, checkup = False):

    """Discard features with close to 0 correlation coefficient to CO2"""

    correlationMatrix = computeCorrelation(noOutlierDf, round = 2)

    highMat, lowMat = filterCorrelation(correlationMatrix, lowThreshhold = lt, yLabel = yLabels[0])
    keep, drop = filteredLabels(highMat.index, lowMat.index, baseLabels, yLabels)

    filteredData = noOutlierDf.drop(columns = drop)
    filteringName = 'dropuncorr'
    if removeLabels:
        filteredData = filteredData.drop(columns = [elem for elem in removeLabels if elem in filteredData.keys()])#[removeLabels[i] for i range(len(removeLabels) if removeLabels[i] in )

        # filteredData = filteredData.drop(columns = removeLabels)
        filteringName = 'dropcolinear'
    if lt == 0:
        filteringName = 'nofilter'
    if checkup:
        filteringName = 'checkup'
    if displayParams['showCorr']or displayParams['archive']:
        plotCorrelation(computeCorrelation(filteredData), displayParams, filteringName, lt)
    Labels = {"baseLabels": baseLabels,"HighCorr": keep, "LowCorr": drop, "MultiCorr/Removed": removeLabels}
    return filteredData, Labels

def computeCorrelation(df, round = 2):

    return df.corr().round(round) #Method :pearson standard correlation coefficient

def filterCorrelation(correlationMatrix, lowThreshhold, yLabel):

    """
    :param correlationMatrix: correlation matrix identifies linear relation between pairs of variables
    :param threshhold:features with a PCC > 0.1 are depicted
    :return: labels with high correlation to output
    """
    #
    highCorMatrix = correlationMatrix.loc[abs((correlationMatrix[yLabel])) >= lowThreshhold]
    lowCorMatrix = correlationMatrix.loc[(abs((correlationMatrix[yLabel])) < lowThreshhold)] + correlationMatrix.loc[correlationMatrix[yLabel].isna()]

    return highCorMatrix, lowCorMatrix

def filteredLabels(hihCorLabels, lowCorLabels, xQuantLabels, yLabel):

    keep = xQuantLabels + [l for l in hihCorLabels if l not in xQuantLabels]
    drop = [l for l in lowCorLabels if l not in xQuantLabels]

    return keep, drop

def plotCorrelation(correlationMatrix, displayParams, filteringName, lt = 0.3, ht = 0.6 ):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    mask = np.zeros_like(correlationMatrix)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(18,15))
    xticklabels = list(range(len(correlationMatrix)))
    if filteringName == 'checkup':
        xticklabels = "auto"
        sub = '- Check FEATURES - '
        cbar= True
        annot = True

    if filteringName == 'nofilter':
        title = 'Pearson correlation coefficient heatmap'
        sub = '- UNFILTERED FEATURES - '
        cbar= False
        annot = False
    if filteringName == 'dropuncorr' :
        title ='Pearson correlation coefficient heatmap'
        sub = '- UNCORRELATED FEATURES REMOVED - (r2 > %s)' % lt
        cbar= False
        annot = False
    if filteringName == 'dropcolinear':
        title ='Pearson correlation coefficient heatmap'
        sub = '- MULTI-COLLINEAR FEATURES REMOVED (r2 < %s) - ' % ht
        cbar = True
        annot = True
    plt.title(label = sub, fontsize = 18, loc='left', va='bottom' )
    # plt.suptitle(t = sub, fontsize = 14, horizontalalignment='left', verticalalignment = 'bottom')
    sns.heatmap(correlationMatrix, annot=annot, mask = mask, cbar = cbar, cbar_kws={"shrink": .80},
                xticklabels = xticklabels, fmt=".001f",ax=ax, cmap="bwr", center = 0, vmin=-1, vmax=1, square = True)

    # sns.set(font_scale=0.5)
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + str(displayParams['random_state']) +'/correlation'
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + filteringName + '.png')
    if displayParams['showCorr']:
        plt.show()
    plt.close()

def trackDataProcessing(displayParams, df, noOutlierdf, filterdf, removeLabelsdf = pd.Series([]) ):

    Content = dict()
    Content["DATAFRAME DIMENSION (X-y) "] = df.shape #including y
    Content["df initial size"] = df.shape
    Content["df without outlier samples"] = noOutlierdf.shape
    Content["df without uncorrelated features"] = filterdf.shape
    if not removeLabelsdf.empty:
        Content["df without multicorrelated features"] = removeLabelsdf.shape

    if not removeLabelsdf.empty:
        Content["Remaining features"] = [k for k in removeLabelsdf.keys()]
    else:
        Content["Remaining features"] = [k for k in filterdf.keys()]
    Content["outlier samples"] = [df.shape[0]-noOutlierdf.shape[0]]
    Content["uncorrelated features"] = [k for k in noOutlierdf.keys() if k not in filterdf.keys()]
    if not removeLabelsdf.empty:
        Content["multicorrelated"] = [k for k in filterdf.keys() if k not in removeLabelsdf.keys()]

    if displayParams["showResults"]:
        for k, v in Content.items():
            print(k, ":", v)
    if displayParams["archive"]:
        saveStudy(displayParams, Content)

def computeYLabelCor(correlationMatrix, yLabel = 'Calculated tCO2e_per_m2'):

    """
    To visualize correlation values
    """

    return correlationMatrix.loc[yLabel]




"""
------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------
"""

#
#
# def XScaleYSplit(df, yLabels, scalerParam):
#     ydf = df[yLabels]
#     xdf = df.drop(columns = yLabels)
#     xScaler = None
#     if scalerParam:
#         if scalerParam == 'MinMaxScaler':
#             xScaler = preprocessing.MinMaxScaler()
#             x_normalized = xScaler.fit_transform(xdf)
#             xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
#         if scalerParam == 'StandardScaler':
#             xScaler = preprocessing.StandardScaler()
#             x_normalized = xScaler.fit_transform(xdf)
#             xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
#         xdf = xScaled
#     return xdf, ydf, xScaler
#
# def XScaleYScaleSplit(df, yLabels, scalerParam, yScale = False, yUnit = None):
#     ydf = df[yLabels]
#     xdf = df.drop(columns = yLabels)
#     xScaler = None
#     yScaler = None
#     if scalerParam:
#         xdf, xScaler = Vscale(scalerParam, xdf)
#         if yScale:
#             ydf, yScaler = Vscale(scalerParam, ydf)
#         else:
#             yScaler = None
#     if yUnit:
#         ydf = np.multiply(ydf,yUnit)
#     return xdf, xScaler, ydf, yScaler
#
# def Vscale(scalerParam, vdf):
#
#     if scalerParam == 'MinMaxScaler':
#         vScaler = preprocessing.MinMaxScaler()
#         v_normalized = vScaler.fit_transform(vdf)
#         vScaled = pd.DataFrame(v_normalized, columns = vdf.keys())
#
#     if scalerParam == 'StandardScaler':
#         vScaler = preprocessing.StandardScaler()
#         v_normalized = vScaler.fit_transform(vdf)
#         vScaled = pd.DataFrame(v_normalized, columns = vdf.keys())
#     vdf = vScaled
#
#     return vdf, vScaler
#
# def unscale(elem, scaler, unitChange = None):
#
#     if unitChange:
#         elem = np.multiply(elem, 1/unitChange)
#     if scaler:
#         return pd.DataFrame(scaler.inverse_transform(elem), columns = elem.keys())
#     else:
#         return elem
#
# def TrainTest(xdf, ydf, test_size, random_state):
#
#     XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size, random_state=random_state)
#
#     return XTrain, XTest, yTrain, yTest
#

# def TrainTestSplit(df, yLabels, test_size, random_state, yUnit = None):
#
#     ydf = df[yLabels]
#     xdf = df.drop(columns = yLabels)
#     if yUnit:
#         ydf = np.multiply(ydf, yUnit)
#
#     #todo : here replace with something exporting panda df
#     XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size, random_state=random_state)
#
#
#     return XTrain, XTest, yTrain, yTest




