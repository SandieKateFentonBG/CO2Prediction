import numpy as np
import pandas as pd


def logitize(xQuali, possibleValues):
    output = dict()
    for label, column in xQuali.items():
        for sublabel in possibleValues[label]:
            output['_'.join([label, sublabel])] = [1 if value == possibleValues[label].index(sublabel) else 0 for value in column]
    return output


def scale(xQuanti, method, positiveValue, qinf, qsup): #'standardize', 'robustscale', 'skl_robustscale'
    xQuantiScaled = dict()

    # todo : OUT OF USE

    # todo : fix positive value issue for powers < 1
    for label, column in xQuanti.items():
        if method == 'standardize':
            xQuantiScaled[label] = list((column - np.mean(column)) / np.std(column))
            if positiveValue:
                xQuantiScaled[label] = [x + positiveValue for x in xQuantiScaled[label]]#todo : check this is fine - allows to ensure positive values
        elif method == 'robustscale':
            xQuantiScaled[label] = list((column - np.median(column)) /(np.quantile(column, qinf)-np.quantile(column, qsup)))
            if positiveValue:
                xQuantiScaled[label] = [x + positiveValue for x in xQuantiScaled[label]]
        elif method == 'skl_robustscale':#use sklearn robustscaler
            from sklearn.preprocessing import RobustScaler
            for label, column in xQuanti.items():
                array = np.array(column).reshape(-1, 1)
                rs = RobustScaler(quantile_range=(qinf, qsup)).fit(array)
                xQuantiScaled[label] = list(rs.transform(array)[0])#todo : not working, flatten output - Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
        elif method == 'skl_standardscale':#use sklearn standardscaler
            from sklearn.preprocessing import StandardScaler
            for label, column in xQuanti.items():
                array = np.array(column).reshape(-1, 1)
                rs = StandardScaler().fit(array)
                xQuantiScaled[label] = list(rs.transform(array))


    return xQuantiScaled




def countPowers(powers):
    count = 0
    for powerList in powers.values():
        count += len(powerList)
    return count


class Data:
    def __init__(self, rawData, scalers):
        if scalers['scaling']:  # todo - scale not up to date
            self.x = scale(rawData.xQuanti, scalers['method'],scalers['positiveValue'],scalers['qinf'],scalers['qsup'])
        else:
            self.x = dict(rawData.xQuanti)
        self.x.update(logitize(rawData.xQuali, rawData.possibleQualities))
        self.y = rawData.y

    def asDataframes(self, batchCount=5):
        x, y, xlabels = self.asArray()
        cutoffIndex = batchCount if x.shape[0] % batchCount == 0\
            else [int(x.shape[0] / batchCount * i) for i in range(1, batchCount)]
        return np.split(x, cutoffIndex), np.split(y, cutoffIndex), xlabels

    def asArray(self): #todo : this should maybe build a df with everything then remove all columns thatarent in mix/powers...
        numValues = len(next(iter(self.x.values())))
        x = np.zeros((numValues, len(self.x)))
        y = np.zeros((numValues, len(self.y)))
        xlabels = [f for f in self.x.keys()]
        for i in range(numValues):  # 80
            x[i, :] = np.array([self.x[f][i] for f in self.x.keys()])
            y[i, :] = np.array([self.y[f][i] for f in self.y.keys()])
        return x, y, xlabels


    def asDataframe(self):
        x, y, xlabels = self.asArray()

        return pd.DataFrame(np.hstack((x, y)), columns=xlabels + list(self.y.keys()))

