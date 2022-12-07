from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#data = lines

def removeOutliers(dataframe, labels, cutOffThreshhold):
    """

    :param labels: labels to remove outliers from - here we take xQuanti
    :param cutOffThreshhold: default 1.5
    :return: Dataframe without outliers

    """
    for l in labels:
        noOutlierDf = removeOutlier(dataframe, l, cutOffThreshhold)
        dataframe = noOutlierDf

    return noOutlierDf

def removeOutlier(df, colName, cutOffThreshhold):

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




