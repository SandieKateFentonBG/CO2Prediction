from Data import *
from FilteredData import *
from PrepData import *


# this allows to handle xQantlabels raised to higher powers
baseLabels = ['GIFA (m2)_exp1', 'Storeys_exp1', 'Typical Span (m)_exp1', 'Typ Qk (kN_per_m2)_exp1']


""" Remove outliers"""
noOutlierDf = removeOutliers(df, labels = baseLabels, cutOffThreshhold=1.5)

# """Correlation of variables & Feature selection"""
# filterDf0 = filteredData(noOutlierDf, baseLabels, yLabels, plot = True, lt = 0.1)
#
# """Remove Multi-correlated Features """
filterDf = filteredData(noOutlierDf, baseLabels, yLabels, plot = False, lt = 0.1,
                         removeLabels=['Basement_None', 'Foundations_Raft'])

#trackDataProcessing(df, noOutlierDf, filterDf)


class Filtered:
    def __init__(self, Data):
        xArray, yArray, xlabels = Data.asArray()
        Dataframe = Data.asDataframe()
        self.x = xArray
        self.y = yArray
        self.xlabels = xlabels
    def
        noOutlierDf = removeOutliers(df, labels=baseLabels, cutOffThreshhold=1.5)
        xArray, yArray = Data.DataArray()[0], Data.DataArray()[1]
        xDataframe, yDataframe = Data.Dataframe()[0], Data.Dataframe()[1]
        xArr