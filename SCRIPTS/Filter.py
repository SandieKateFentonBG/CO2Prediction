import pandas as pd
import numpy as np

#DEFAULT VALUES
method = "spearman"
round = 2
lowThreshhold = 0.1
highThreshhold = 0.65

#todo : convert this to a class - find better naming

def computeCorrelation(df, method = method, round = round):

    "correlationMatrix: correlation matrix identifies relation between pairs of variables"

    return df.corr(method = method).round(round) #Method :pearson standard correlation coefficient

def filterUncorrelated(noOutlierDf, baseLabels, yLabel):

    """
    :param correlationMatrix: correlation matrix identifies relation between pairs of variables
    :param threshhold:features with a PCC > 0.1 are depicted #todo : minimum threshold for Pearson/ Spearman?
    :return: labels with high correlation to output
    """
    #

    #correlation
    unfilteredCorrelation = noOutlierDf.corr(method=method).round(round)
    unfilteredCorrelationMatrixAbs = unfilteredCorrelation.abs()
    #labels
    highCorMatrix = unfilteredCorrelationMatrixAbs.loc[abs((unfilteredCorrelationMatrixAbs[yLabel])) >= lowThreshhold]
    lowCorMatrix = unfilteredCorrelationMatrixAbs.loc[(abs((unfilteredCorrelationMatrixAbs[yLabel])) < lowThreshhold)]
    nanCorMatrix = unfilteredCorrelationMatrixAbs.loc[unfilteredCorrelationMatrixAbs[yLabel].isna()]
    dropCorMatrix = pd.concat([lowCorMatrix, nanCorMatrix], axis=0)
    dropLabels = [l for l in dropCorMatrix.index if l not in baseLabels]
    #filtered df
    filteredDf = noOutlierDf.drop(columns=dropLabels)
    #dictionary
    filterUncorrelated = {"correlationMatrix": filteredDf.corr(method=method).round(round), "dropLabels": dropLabels,
                          "filteredDf": filteredDf}

    #todo :  understand NaN = 0
    # spearman : cor(i,j) = cov(i,j)/[stdev(i)*stdev(j)]
    # If the values of the ith or jth variable do not vary,
    # then the respective standard deviation will be zero
    # and so will the denominator of the fraction.

    #return highCorMatrix, dropCorMatrix
    return filterUncorrelated #todo: check abs value > output content

def filterRedundant(noOutlierDf, baseLabels, yLabel):

    #correlation
    unfilteredCorrelation = noOutlierDf.corr(method=method).round(round)
    unfilteredCorrelationMatrixAbs = unfilteredCorrelation.abs()
    #labels
    upper_tri = unfilteredCorrelationMatrixAbs.where(np.triu(np.ones(unfilteredCorrelationMatrixAbs.shape), k=1).astype(np.bool))
    redundantLabels = [column for column in upper_tri.columns if any(upper_tri[column] >= highThreshhold)]
    #this drops one of the two features that are collinear
    dropLabels = [l for l in redundantLabels if l not in baseLabels]
    #filterdropLabelsed df
    filteredDf = noOutlierDf.drop(columns=dropLabels)
    #dictionary
    filterRedundant = {"correlationMatrix": filteredDf.corr(method=method).round(round), "dropLabels": dropLabels,
                          "filteredDf": filteredDf}

    return filterRedundant

def filteringData(noOutlierDf, baseLabels, yLabel):
    filterUncorrelatedDict = filterUncorrelated(noOutlierDf, baseLabels, yLabel)
    filterRedundantDict = filterRedundant(filterUncorrelatedDict['filteredDf'], baseLabels, yLabel)
    return filterUncorrelatedDict, filterRedundantDict

def filterDf(trainDf, validDf, testDf, filterUncorrelatedDict, filterRedundantDict) :
    # filteredTrainDf = filterRedundantDict['filteredDf']
    dropLabels = filterUncorrelatedDict["dropLabels"] + filterRedundantDict["dropLabels"]
    filteredTestDf = testDf.drop(columns=dropLabels)
    filteredValidDf = validDf.drop(columns=dropLabels)
    filteredTrainDf = trainDf.drop(columns=dropLabels)
    return filteredTrainDf, filteredValidDf, filteredTestDf