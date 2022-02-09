#todo : convert to a class? or simplify to one single function?
import pandas as pd
from Helpers import *

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
    # print(fence_low, fence_high)
    return df.loc[(df[colName] > fence_low) & (df[colName] < fence_high)]

def filteredData(noOutlierDf, baseLabels, yLabels, plot = False, lt = 0.1, ht = 0.5, yLabel ='Calculated tCO2e_per_m2',
                 removeLabels = None):

    """Discard features with close to 0 correlation coefficient to CO2"""

    correlationMatrix = computeCorrelation(noOutlierDf, round = 2)

    highMat, lowMat = filterCorrelation(correlationMatrix, lowThreshhold = lt, yLabel = yLabel)
    keep, drop = filteredLabels(highMat.index, lowMat.index, baseLabels, yLabels)

    filteredData = noOutlierDf.drop(columns = drop)

    if removeLabels:
        filteredData = filteredData.drop(columns = removeLabels)

    if plot:
        plotCorrelation(computeCorrelation(filteredData))
    Labels = {"baseLabels": baseLabels,"HighCorr": keep, "LowCorr": drop, "MultiCorr/Removed": removeLabels}
    return filteredData

def computeCorrelation(df, round = 2):

    return df.corr().round(round) #Method :pearson standard correlation coefficient

def filterCorrelation(correlationMatrix, lowThreshhold = 0.1, yLabel ='Calculated tCO2e_per_m2'):

    """
    :param correlationMatrix: correlation matrix identifies linear relation between pairs of variables
    :param threshhold:features with a PCC > 0.1 are depicted
    :return: labels with high correlation to output
    """
    #
    highCorMatrix = correlationMatrix.loc[abs((correlationMatrix[yLabel])) >= lowThreshhold]
    lowCorMatrix = correlationMatrix.loc[(abs((correlationMatrix[yLabel])) < lowThreshhold)] + correlationMatrix.loc[correlationMatrix['Calculated tCO2e_per_m2'].isna()]

    #todo : remove muticol not with y !!
    #todo : this filters outmy GIFA !! I should scale everything before starting?
    #todo : use this only for filtering out qualitative features?

    return highCorMatrix, lowCorMatrix

def filteredLabels(hihCorLabels, lowCorLabels, xQuantLabels, yLabel):

    keep = xQuantLabels + [l for l in hihCorLabels if l not in xQuantLabels]
    drop = [l for l in lowCorLabels if l not in xQuantLabels]

    return keep, drop

def plotCorrelation(correlationMatrix):

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    mask = np.zeros_like(correlationMatrix)
    mask[np.triu_indices_from(mask)] = True
    fig, ax = plt.subplots(figsize=(20,20))
    sns.heatmap(correlationMatrix, annot=True, mask = mask, fmt=".001f",ax=ax, cmap="bwr", center = 0, vmin=-1, vmax=1, square = True)
    plt.show()
    # plt.clf()

def trackDataProcessing(displayParams, df, noOutlierdf, filterdf, removeLabelsdf = pd.Series([]) ):
    # print("")
    # print("DATAFRAME DIMENSION", df.shape)
    # print("")
    # print("initial size", df.shape)
    # print("without outliers", noOutlierdf.shape)
    # print("without uncorrelated features", filterdf.shape)
    # if not removeLabelsdf.empty:
    #     print("without multicorrelated features", removeLabelsdf.shape)
    # print("")

    Content = dict()
    Content["DATAFRAME DIMENSION"] = df.shape
    Content["df initial size"] = df.shape
    Content["initial keys"] = df.keys()
    Content["df without outliers"] = noOutlierdf.shape
    Content["keys without outliers"] = noOutlierdf.keys()
    Content["df without uncorrelated features"] = filterdf.shape
    Content["keys without uncorrelated features"] = filterdf.keys()
    if not removeLabelsdf.empty:
        Content["df without multicorrelated features"] = removeLabelsdf.shape
        Content["keys without multicorrelated features"] = removeLabelsdf.keys()

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