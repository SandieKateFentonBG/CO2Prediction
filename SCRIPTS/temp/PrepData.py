
from SCRIPTS.temp.Archiver import *
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


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


def TrainTestSplitAsDf(df, yLabels, test_size, random_state, yUnit=None):
    ydf = df[yLabels]
    xdf = df.drop(columns=yLabels)
    if yUnit:
        ydf = np.multiply(ydf, yUnit)

    XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size,
                                                    random_state=random_state)
    columnsNamesArr = xdf.columns.values
    XTrainDf = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
    XTestDf = pd.DataFrame(data=XTest, columns=columnsNamesArr)
    yTrainDf = pd.DataFrame(data=yTrain, columns=yLabels)
    yTestDf = pd.DataFrame(data=yTest, columns=yLabels)

    return XTrainDf, XTestDf, yTrainDf, yTestDf

def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd

def scaleXDf(XTrain, XTest, xQuantLabels):

    mydict = dict()
    for l in xQuantLabels:
        colMean, colStd = dfColMeanStd(XTrain, l)

        XTrain[l] = (XTrain[l] - colMean) / colStd
        XTest[l] = (XTest[l] - colMean) / colStd
        mydict[l] = [colMean, colStd]

    MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])

    return XTrain, XTest, MeanStdDf





