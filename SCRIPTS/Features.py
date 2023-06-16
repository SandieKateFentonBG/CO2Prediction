import numpy as np
import pandas as pd


#features = columns

def logitize(xQuali, possibleValues):

    output = dict()
    for label, column in xQuali.items():
        for sublabel in possibleValues[label]:
            output['_'.join([label, sublabel])] = [1 if value == possibleValues[label].index(sublabel) else 0 for value in column]
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

    def removeUnderrepresented(self, df, label, value, cutOffThreshhold):

        removed = None
        info = '_'.join([label, value])
        newdf = df.groupby(info).filter(lambda x: len(x) > cutOffThreshhold)

        if newdf.shape != df.shape:
            removed = value
        return newdf, removed

    def removeUnderrepresenteds(self, cutOffThreshhold, removeUnderrepresentedsFrom):
        """

        :param labels: labels to remove underrepresented values from - here we take xQuali
        :param cutOffThreshhold: default 1.5
        :return: Dataframe without outliers

        """

        dataframe = self.asDataframe()

        for label in removeUnderrepresentedsFrom:
            self.removedDict[label] = []
            for value in self.rawData.possibleQualities[label] :
                noOutlierDf, removed = self.removeUnderrepresented(dataframe, label, value, cutOffThreshhold)
                dataframe = noOutlierDf
                if removed :
                    self.removedDict[label].append(removed)
        newdf = noOutlierDf[[i for i in noOutlierDf if len(set(noOutlierDf[i])) > 1]]
        #todo this stepp removes columns with all values identical > remaining when underrrepresented rows are removed
        # > maybe they should be left in to avoid issues when plotting

        #todo : maybe these features should be removed from posssible qualities > will generate issues in the plots

        return newdf, self.removedDict




