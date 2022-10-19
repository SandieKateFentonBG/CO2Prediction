import pandas as pd
import numpy as np

#DEFAULT VALUES
# method = "spearman"
# round = 2
# lowThreshhold = 0.1
# highThreshhold = 0.65

#todo : find better naming

def computeCorrelation(df, method, round):

    "correlationMatrix: correlation matrix identifies relation between pairs of variables"

    return df.corr(method = method).round(round) #Method :pearson standard correlation coefficient

class FilterFeatures:

    def __init__(self, trainDf, valDf, testDf, baseLabels, yLabel, method ="spearman", corrRounding = 2,
                 lowThreshhold = 0.1, highThreshhold = 0.65):
        self.method = method
        self.lowThreshhold = lowThreshhold
        self.highThreshhold = highThreshhold
        self.corrRounding = corrRounding
        self.filterUncorrelated(trainDf, baseLabels, yLabel, method, lowThreshhold)
        self.yLabel = yLabel
        # Generates :
        # self.correlationMatrix_All = correlationMatrix_All
        # self.uncorrelatedLabels = uncorrelatedFeatures
        # self.correlationMatrix_NoUncorrelated = correlationMatrix_NoUncorrelated
        # self.DfNoUncorrelated = DfNoUncorrelated

        self.filterRedundant(self.DfNoUncorrelated, baseLabels, method, highThreshhold)

        # Generates :
        # self.redundantLabels = redundantFeatures
        # self.correlationMatrix_NoRedundant = correlationMatrix_NoRedundant
        # self.DfNoRedundant = DfNoRedundant

        self.droppedLabels = self.uncorrelatedLabels + self.redundantLabels

        self.trainDf = trainDf.drop(columns=self.droppedLabels)
        self.valDf = valDf.drop(columns=self.droppedLabels)
        self.testDf = testDf.drop(columns=self.droppedLabels)

        self.XTrain = self.trainDf.drop(columns=yLabel)
        self.XVal = self.valDf.drop(columns=yLabel)
        self.XTest = self.testDf.drop(columns=yLabel)
        self.yTrain = self.trainDf[yLabel]
        self.yVal = self.valDf[yLabel]
        self.yTest = self.testDf[yLabel]

        self.selectedLabels = list(self.trainDf.columns.values)

    def filterUncorrelated(self, df, baseLabels, yLabel, method, lowThreshhold):
        """
        :param correlationMatrix: correlation matrix identifies relation between pairs of variables
        :param threshhold:features with a PCC > 0.1 are depicted #todo : minimum threshold for Pearson/ Spearman?
        :return: labels with high correlation to output
        """

        # correlation
        self.correlationMatrix_All = df.corr(method=method).round(self.corrRounding)
        unfilteredCorrelationMatrixAbs = self.correlationMatrix_All.abs()
        # labels
        highCorMatrix = unfilteredCorrelationMatrixAbs.loc[
            abs((unfilteredCorrelationMatrixAbs[yLabel])) >= lowThreshhold]
        lowCorMatrix = unfilteredCorrelationMatrixAbs.loc[
            (abs((unfilteredCorrelationMatrixAbs[yLabel])) < lowThreshhold)]
        nanCorMatrix = unfilteredCorrelationMatrixAbs.loc[unfilteredCorrelationMatrixAbs[yLabel].isna()]
        dropCorMatrix = pd.concat([lowCorMatrix, nanCorMatrix], axis=0)
        self.uncorrelatedLabels = [l for l in dropCorMatrix.index if l not in baseLabels]
        # filtered df
        self.DfNoUncorrelated = df.drop(columns=self.uncorrelatedLabels)#todo: check abs value > output content
        self.correlationMatrix_NoUncorrelated = self.DfNoUncorrelated.corr(method=method).round(self.corrRounding)

        #todo :  understand NaN = 0
        # spearman : cor(i,j) = cov(i,j)/[stdev(i)*stdev(j)]
        # If the values of the ith or jth variable do not vary,
        # then the respective standard deviation will be zero
        # and so will the denominator of the fraction.

    def filterRedundant(self, df, baseLabels, method, highThreshhold):

        #correlation
        unfilteredCorrelation = df.corr(method=method).round(self.corrRounding)
        unfilteredCorrelationMatrixAbs = unfilteredCorrelation.abs()
        #labels
        upper_tri = unfilteredCorrelationMatrixAbs.where(np.triu(np.ones(unfilteredCorrelationMatrixAbs.shape), k=1).astype(np.bool))
        redundantLabelsAll = [column for column in upper_tri.columns if any(upper_tri[column] >= highThreshhold)]
        #this drops one of the two features that are collinear
        self.redundantLabels = [l for l in redundantLabelsAll if l not in baseLabels]
        #filter df
        self.DfNoRedundant = df.drop(columns=self.redundantLabels)
        self.correlationMatrix_NoRedundant = self.DfNoRedundant.corr(method=method).round(self.corrRounding)



