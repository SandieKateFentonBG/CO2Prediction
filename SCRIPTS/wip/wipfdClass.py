#todo : convert to a class? does this make sense?


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

def computeCorrelation(df, round = 2):

    return df.corr().round(round)


class FilterData:
    def __init__(self, Data, cutOffThreshhold=3, rawData):

        self.noOutlierDf = self.removeOutliers(Data.asDataframe(), rawData.xQuanti, cutOffThreshhold)
        self.

    def removeOutliers(self, dataframe, labels, cutOffThreshhold=1.5):
        """

        :param labels: labels to remove outliers from - here we take xQuanti
        :param cutOffThreshhold: default 1.5
        :return: Dataframe without outliers

        """

        for l in labels:
            noOutlierDf = removeOutlier(dataframe, l, cutOffThreshhold=cutOffThreshhold)
            dataframe = noOutlierDf

        return noOutlierDf


    def filteredData(self, noOutlierDf, rawData.xQuanti, rawData.yLabel, plot = False, threshhold = 0.1, yLabel = 'Calculated tCO2e_per_m2'):

        """Discard features with close to 0 correlation coefficient to CO2"""

        correlationMatrix = computeCorrelation(self.learningDf, round = 2)

        highMat, lowMat = self.splitUncorrelated(correlationMatrix, threshhold = threshhold, yLabel = yLabel)
        keep, drop = self.filteredLabels(highMat.index, lowMat.index, xQuantLabels, yLabels)

        filteredData = self.learningDf.drop(columns = drop)

        if plot:
            plotCorrelation(computeCorrelation(filteredData))

        return filteredData


    def filterCorrelation(self, threshhold = 0.1, yLabel ='Calculated tCO2e_per_m2'):

        """
        :param correlationMatrix: correlation matrix identifies linear relation between pairs of variables
        :param threshhold:features with a PCC > 0.1 are depicted
        :return: labels with high correlation to output
        """
        correlationMatrix = self.computeCorrelation(self.learningDf, round = 2)
        highCorMatrix = correlationMatrix.loc[abs((correlationMatrix[yLabel])) >= threshhold]
        lowCorMatrix = correlationMatrix.loc[(abs((correlationMatrix[yLabel])) < threshhold)] + correlationMatrix.loc[correlationMatrix['Calculated tCO2e_per_m2'].isna()]

        return highCorMatrix, lowCorMatrix

    def filteredLabels(self, hihCorLabels, lowCorLabels, xQuantLabels, yLabel):

        keep = xQuantLabels + [l for l in hihCorLabels if l not in xQuantLabels]
        drop = [l for l in lowCorLabels if l not in xQuantLabels]

        return keep, drop

    def plotCorrelation(self, correlationMatrix):

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(18,18))

        sns.heatmap(correlationMatrix, annot=True, fmt=".001f",ax=ax)
        plt.show()



    def computeYLabelCor(self, correlationMatrix, RawData.yLabels):

        """
        To visualize correlation values
        """

        return correlationMatrix.loc[yLabel]