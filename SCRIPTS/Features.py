import numpy as np
import pandas as pd


#features = columns

def logitize(xQuali, possibleValues, splitter = '='):

    output = dict()
    for label, column in xQuali.items():
        for sublabel in possibleValues[label]:
            output[splitter.join([label, sublabel])] = [1 if value == possibleValues[label].index(sublabel) else 0 for value in column]
    return output

def countPowers(powers):
    count = 0
    for powerList in powers.values():
        count += len(powerList)
    return count


class Features:
    def __init__(self, rawData):

        self.rawData = rawData
        self.x = dict(rawData.xQuanti)
        self.x.update(logitize(rawData.xQuali, rawData.possibleQualities))
        self.y = rawData.y
        self.removedDict = dict()
        self.droppedLabels = []
        self.remainingLabels = []
        self.allLabels = []


    def asDataframes(self, batchCount=5, powers=None, mixVariables=None):
        x, y, xlabels = self.asArray(powers, mixVariables)
        cutoffIndex = batchCount if x.shape[0] % batchCount == 0\
            else [int(x.shape[0] / batchCount * i) for i in range(1, batchCount)]
        return np.split(x, cutoffIndex), np.split(y, cutoffIndex), xlabels

    def asArray(self, powers={}, mixVariables=[]):
        numValues = len(next(iter(self.x.values())))
        x = np.zeros((numValues, len(self.x)-len(powers)))
        y = np.zeros((numValues, len(self.y)))
        xlabels = [f for f in self.x.keys() if f not in powers.keys()]
        for i in range(numValues):  # 80
            x[i, :] = np.array([self.x[f][i] for f in self.x.keys() if f not in powers.keys()])
            y[i, :] = np.array([self.y[f][i] for f in self.y.keys()])

        return x, y, xlabels

    def asDataframe(self, powers={}, mixVariables=[]):
        x, y, xlabels = self.asArray(powers, mixVariables)
        self.Dataframe = [x, y]

        return pd.DataFrame(np.hstack((x, y)), columns=xlabels + list(self.y.keys()))

    def removeUnderrepresented(self, df, label, value, cutOffThreshhold, splitter = '='):

        """this removes rows of categories with too few features"""

        removed_Value, removed_Label_Value = None, None
        # info = '='.join([label, value]) #changed here
        info = splitter.join([label, value]) #changed here

        newdf = df.groupby(info).filter(lambda x: len(x) > cutOffThreshhold)

        if newdf.shape != df.shape:
            removed_Value = value
            removed_Label_Value = info

        return newdf, removed_Value, removed_Label_Value

    def removeUnderrepresenteds(self, df, cutOffThreshhold, removeUnderrepresentedsFrom, splitter = '='):
        """
        this removes columns of categories with too few features

        :param labels: labels to remove underrepresented values from - here we take xQuali
        :param cutOffThreshhold: default 1.5
        :return: Dataframe without outliers

        """
        # if df == "default":

        # else:
        #     print("true")


        # dataframe = self.asDataframe()
        dataframe = df

        for label in removeUnderrepresentedsFrom:
            self.removedDict[label] = []
            for value in self.rawData.possibleQualities[label] :
                noOutlierDf,  removed_Value, removed_Label_Value = self.removeUnderrepresented(dataframe, label, value, cutOffThreshhold)
                dataframe = noOutlierDf


        newdf = noOutlierDf[[i for i in noOutlierDf if len(set(noOutlierDf[i])) > 1]]
        self.remainingLabels = [elem for elem in newdf.columns]
        self.allLabels = [elem for elem in dataframe.columns]
        self.droppedLabels = [elem for elem in self.allLabels if elem not in self.remainingLabels]
        for label_value in self.droppedLabels:
            lab, val = label_value.split(splitter)
            self.removedDict[lab].append(val)

        return newdf, self.removedDict, self.droppedLabels




