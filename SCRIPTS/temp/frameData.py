import numpy as np
import pandas as pd


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


class Data:
    def __init__(self, rawData):

        self.x = dict(rawData.xQuanti)
        self.x.update(logitize(rawData.xQuali, rawData.possibleQualities))
        self.y = rawData.y

        #todo : powers and mixvariables - is this worth investigating
        # - no info is provided about this in built-in regressions - only weights ..




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

