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

        #todo : 1 function that runs through and stores info

        # self.DataArray = []
        # self.DataFrame = []
        # self.xLabels = []
        # self.xLabels.update(dataModification)


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
        if powers:
            xPowers, xPowerLabels = self.powerUpVariables(powers, numValues)
            x = np.hstack((x, xPowers))
            xlabels += xPowerLabels
        if mixVariables:
            # todo : not used for now
            xCross, xCrossLabels = self.combineVariables(mixVariables, numValues)
            x = np.hstack((x,xCross))
            xlabels += xCrossLabels
        # self.DataArray = [x, y]
        return x, y, xlabels

    def asDataframe(self, powers=None, mixVariables=None):
        x, y, xlabels = self.asArray(powers, mixVariables)
        self.Dataframe = [x, y]

        return pd.DataFrame(np.hstack((x, y)), columns=xlabels + list(self.y.keys()))

    def powerUpVariables(self, powers, numValues):
        xPowers = np.zeros((numValues, countPowers(powers)))
        colIndex = 0
        xPowerLabels = []
        for label, powerList in powers.items():
            for power in powerList:
                xPowers[:, colIndex] = np.float_power(np.abs(self.x[label]), power)  # todo: check fix : np.abs
                xPowerLabels.append(label + '_exp' + str(power))
                colIndex += 1
        return xPowers, xPowerLabels

    def combineVariables(self, mixVariables, numValues):
        #todo : this doesn't work if the keys aren't in the linear labels:
        # ex : y =  GIFA  + STOREY + GIFA * STOREY ok
        # but y = GIFA * Storey ko
        xCross = np.zeros((numValues, len(mixVariables))) #80, 2
        colIndex = 0
        xCrossLabels = []
        for list in mixVariables:
            arrays = [np.array([self.x[f] for f in list]).T]
            xCrossLabels.append('*'.join(list))
            xCross[:, colIndex] = np.prod(np.hstack(arrays), axis=1)
            colIndex += 1
        return xCross, xCrossLabels

    def dataModification(self, powers={}, mixVariables=[]):

        numValues = len(next(iter(self.x.values())))
        xUnchanged = np.zeros((numValues, len(self.x)))
        yUnchanged = np.zeros((numValues, len(self.y)))
        xlabelsUnchanged = [f for f in self.x.keys()]
        for i in range(numValues):  # 80
            xUnchanged[i, :] = np.array([self.x[f][i] for f in self.x.keys()])
            yUnchanged[i, :] = np.array([self.y[f][i] for f in self.y.keys()])

        print('Unchanged Labels: ', xlabelsUnchanged)
        print('xUnchanged : ', type(xUnchanged), xUnchanged.shape)
        print(xUnchanged[0])
        print('yUnchanged : ', type(yUnchanged), yUnchanged.shape, )
        print(yUnchanged[0])
        print('')

        xNew, yNew, xlabelsNew = self.asArray(powers, mixVariables)

        print('New Labels: ', xlabelsNew)
        print('xNew : ', type(xNew), xNew.shape)
        print(xNew[0])
        print('yNew : ', type(yNew), yNew.shape)
        print(yNew[0])
        print('')

        self.xLabels = [xlabelsUnchanged, xlabelsNew]

