from sklearn import preprocessing

import pandas as pd

def normalize(df):
    x = df.values
    Scaler = preprocessing.MinMaxScaler()
    x_normalized = Scaler.fit_transform(x)
    xScaled = pd.DataFrame(x_normalized, columns = df.keys())

    return (xScaled, Scaler)

def unscale(elem, scaler):
    return pd.DataFrame(scaler.inverse_transform(elem), columns = elem.keys())

def XYsplit(df, yLabels):
    ydf = df[yLabels]
    xdf = df.drop(columns = yLabels)
    return xdf, ydf

def crossvalidationSplit(x, y, batchCount=5):
    cutoffIndex = [0] + [int(x.shape[0]/batchCount * i) for i in range(1, batchCount)] if x.shape[0] % batchCount == 0\
        else [int(x.shape[0] / batchCount * i) for i in range(batchCount)]
    cutoffIndex += [x.shape[0]]
    xsets = [x.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]
    ysets = [y.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]

    return xsets, ysets

def TrainTestSets(filterDf, yLabels):
    """Normalize and split Train-Test """
    xdf, ydf = XYsplit(filterDf, yLabels)
    xs, ys = crossvalidationSplit(xdf, ydf)
    return xs, ys

def TrainTestDf(xSets, ySets, testSetIndex=1):

    xTrain = pd.concat([batch for batch in xSets if batch is not xSets[testSetIndex]])
    yTrain = pd.concat([batch for batch in ySets if batch is not ySets[testSetIndex]])
    return (xTrain, yTrain), (xSets[testSetIndex], ySets[testSetIndex])

def TrainTestArray(filterDf, yLabels, testSetIndex):
    #todo : shuffle data...
    xs, ys = TrainTestSets(filterDf, yLabels)
    (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xs, ys, testSetIndex)
    return (xTrain.values, yTrain.values.reshape(-1, 1)), (xTest.values, yTest.values.reshape(-1, 1))


