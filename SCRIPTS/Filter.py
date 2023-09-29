import pandas as pd
import numpy as np

#DEFAULT VALUES
# method = "spearman"
# round = 2
# lowThreshhold = 0.1
# highThreshhold = 0.65

#todo : find better naming


# changed all the xTrain to xVal = what RFE is trained on
# added XCheck = what RFE is tested on



def computeCorrelation(df, method, round):

    "correlationMatrix: correlation matrix identifies relation between pairs of variables"

    return df.corr(method = method).round(round) #Method :pearson standard correlation coefficient



class FilterFeaturesCV:

    def __init__(self, splitDf, baseLabels, method="spearman", corrRounding=2,
                 lowThreshhold=0.1, highThreshhold=0.65):

        FullDf = pd.concat([splitDf.RDf, splitDf.checkDf, splitDf.valDf], axis=0)
        valDf = splitDf.valDf
        self.yLabel = splitDf.yLabel
        # self.random_state = splitDf.random_state
        self.method = method
        self.lowThreshhold = lowThreshhold
        self.highThreshhold = highThreshhold
        self.corrRounding = corrRounding

        self.filterUncorrelated(FullDf, baseLabels, self.yLabel, method, lowThreshhold)
        # todo :  this was changed to have more data to compute correlation on / avoid having columns of 0
        # self.filterUncorrelated(valDf, baseLabels, self.yLabel, method, lowThreshhold)

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

        self.valDf = valDf.drop(columns=self.droppedLabels)
        self.XVal = self.valDf.drop(columns=self.yLabel)
        self.selectedLabels = list(self.XVal.columns.values)
        self.selector = 'fl_' + self.method

        #for later
        self.random_state = None
        self.trainDf = None
        self.testDf = None
        self.checkDf = None
        self.XTrain = None
        self.XTest = None
        self.XCheck = None
        self.yTrain = None
        self.yVal = None
        self.yTest = None
        self.yCheck = None

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
        self.DfNoUncorrelated = df.drop(columns=self.uncorrelatedLabels)  # todo: check abs value > output content
        self.correlationMatrix_NoUncorrelated = self.DfNoUncorrelated.corr(method=method).round(self.corrRounding)

        # todo :  understand NaN = 0
        # spearman : cor(i,j) = cov(i,j)/[stdev(i)*stdev(j)]
        # If the values of the ith or jth variable do not vary,
        # then the respective standard deviation will be zero
        # and so will the denominator of the fraction.

    def filterRedundant(self, df, baseLabels, method, highThreshhold):
        # correlation
        unfilteredCorrelation = df.corr(method=method).round(self.corrRounding)
        unfilteredCorrelationMatrixAbs = unfilteredCorrelation.abs()

        # sort in ascending correlation to target lines
        sortedMatrix = unfilteredCorrelationMatrixAbs.sort_values(by=self.yLabel, ascending=True).sort_values(
            by=self.yLabel, axis=1, ascending=True)

        # labels
        upper_tri = sortedMatrix.where(np.triu(np.ones(sortedMatrix.shape), k=1).astype(np.bool))
        redundantLabelsAll = [column for column in upper_tri.columns if any(upper_tri[column] >= highThreshhold)]

        # this drops one of the two features that are collinear
        self.redundantLabels = [l for l in redundantLabelsAll if l not in baseLabels + [self.yLabel]]

        # filter df
        self.DfNoRedundant = df.drop(columns=self.redundantLabels)
        self.correlationMatrix_NoRedundant = self.DfNoRedundant.corr(method=method).round(self.corrRounding)

    def updateFilterCV(self, baseFormatedDf):

        trainDf = baseFormatedDf.trainDf
        testDf = baseFormatedDf.testDf
        checkDf = baseFormatedDf.checkDf

        self.random_state = baseFormatedDf.random_state
        self.trainDf = trainDf.drop(columns=self.droppedLabels)
        self.testDf = testDf.drop(columns=self.droppedLabels)
        self.checkDf = checkDf.drop(columns=self.droppedLabels)
        self.XTrain = self.trainDf.drop(columns=self.yLabel)
        self.XTest = self.testDf.drop(columns=self.yLabel)
        self.XCheck = self.checkDf.drop(columns=self.yLabel)
        self.yTrain = self.trainDf[self.yLabel]
        self.yVal = self.valDf[self.yLabel]
        self.yTest = self.testDf[self.yLabel]
        self.yCheck = self.checkDf[self.yLabel]





class FilterFeatures:

    def __init__(self, baseFormatedDf, baseLabels, method ="spearman", corrRounding = 2,
                 lowThreshhold = 0.1, highThreshhold = 0.65):
        trainDf = baseFormatedDf.trainDf

        valDf = baseFormatedDf.valDf
        testDf = baseFormatedDf.testDf
        checkDf = baseFormatedDf.checkDf
        self.yLabel = baseFormatedDf.yLabel
        self.random_state = baseFormatedDf.random_state

        self.method = method
        self.lowThreshhold = lowThreshhold
        self.highThreshhold = highThreshhold
        self.corrRounding = corrRounding

        self.filterUncorrelated(valDf, baseLabels, self.yLabel, method, lowThreshhold)

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
        self.checkDf = checkDf.drop(columns=self.droppedLabels)
        self.XTrain = self.trainDf.drop(columns=self.yLabel)
        self.XVal = self.valDf.drop(columns=self.yLabel)
        self.XTest = self.testDf.drop(columns=self.yLabel)
        self.XCheck = self.checkDf.drop(columns=self.yLabel)
        self.yTrain = self.trainDf[self.yLabel]
        self.yVal = self.valDf[self.yLabel]
        self.yTest = self.testDf[self.yLabel]
        self.yCheck = self.checkDf[self.yLabel]

        self.selectedLabels = list(self.XVal.columns.values)
        self.selector = 'fl_' + self.method






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

        #sort in ascending correlation to target lines
        sortedMatrix = unfilteredCorrelationMatrixAbs.sort_values(by=self.yLabel, ascending=True).sort_values(by=self.yLabel, axis = 1, ascending=True)

        #labels
        upper_tri = sortedMatrix.where(np.triu(np.ones(sortedMatrix.shape), k=1).astype(np.bool))
        redundantLabelsAll = [column for column in upper_tri.columns if any(upper_tri[column] >= highThreshhold)]

        #this drops one of the two features that are collinear
        self.redundantLabels = [l for l in redundantLabelsAll if l not in baseLabels + [self.yLabel]]


        #filter df
        self.DfNoRedundant = df.drop(columns=self.redundantLabels)
        self.correlationMatrix_NoRedundant = self.DfNoRedundant.corr(method=method).round(self.corrRounding)





