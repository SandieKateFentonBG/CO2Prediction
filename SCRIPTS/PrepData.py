
import pandas as pd
from Archiver import *

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd



def removeOutliers(dataframe, labels, cutOffThreshhold=1.5):
    """

    :param labels: labels to remove outliers from - here we take xQuanti
    :param cutOffThreshhold: default 1.5
    :return: Dataframe without outliers

    """

    for l in labels:
        noOutlierDf = removeOutlier(dataframe, l, cutOffThreshhold=cutOffThreshhold)
        dataframe = noOutlierDf

    return noOutlierDf

def removeOutlier(df, colName, cutOffThreshhold = 1.5):

    """Removes all outliers on a specific column from a given dataframe.

    Args:
        df (pandas.DataFrame): Iput pandas dataframe containing outliers
        colName (str): Column name on which to search outliers
        CutOfftreshhold : default =1.5 ; extreme = 3

    Returns:
        pandas.DataFrame: DataFrame without outliers

    Comments : Interquartile range Method for removing outliers is specific to non Gaussian distribution of data
    - could consider other methods


    """

    q1 = df[colName].quantile(0.25)
    q3 = df[colName].quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - cutOffThreshhold * iqr
    fence_high = q3 + cutOffThreshhold * iqr
    return df.loc[(df[colName] > fence_low) & (df[colName] < fence_high)]

def filteredData(noOutlierDf, baseLabels, yLabels, displayParams, lt, removeLabels = None):

    """Discard features with close to 0 correlation coefficient to CO2"""

    correlationMatrix = computeCorrelation(noOutlierDf, round = 2)

    highMat, lowMat = filterCorrelation(correlationMatrix, lowThreshhold = lt, yLabel = yLabels[0])
    keep, drop = filteredLabels(highMat.index, lowMat.index, baseLabels, yLabels)

    filteredData = noOutlierDf.drop(columns = drop)
    filteringName = 'keepcorr'
    if removeLabels:
        filteredData = filteredData.drop(columns = [elem for elem in removeLabels if elem in filteredData.keys()])#[removeLabels[i] for i range(len(removeLabels) if removeLabels[i] in )

        # filteredData = filteredData.drop(columns = removeLabels)
        filteringName = 'dropcolin'

    if displayParams['showCorr']or displayParams['archive']:
        plotCorrelation(computeCorrelation(filteredData), displayParams, filteringName)
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

def plotCorrelation(correlationMatrix, displayParams, filteringName):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    mask = np.zeros_like(correlationMatrix)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(20,20))

    sns.heatmap(correlationMatrix, annot=True, mask = mask, fmt=".001f",ax=ax, cmap="bwr", center = 0, vmin=-1, vmax=1, square = True)
    # sns.set(font_scale=0.5)
    if displayParams['archive']:
        import os
        outputFigPath = displayParams["outputPath"] + displayParams["reference"] + '/correlation'
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

def XScaleYSplit(df, yLabels, scalerParam):
    ydf = df[yLabels]
    xdf = df.drop(columns = yLabels)
    xScaler = None
    if scalerParam:
        if scalerParam == 'MinMaxScaler':
            xScaler = preprocessing.MinMaxScaler()
            x_normalized = xScaler.fit_transform(xdf)
            xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
        if scalerParam == 'StandardScaler':
            xScaler = preprocessing.StandardScaler()
            x_normalized = xScaler.fit_transform(xdf)
            xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
        xdf = xScaled
    return xdf, ydf, xScaler

def TrainTest(xdf, ydf, test_size, random_state):

    XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size, random_state=random_state)
    return XTrain, XTest, yTrain, yTest

