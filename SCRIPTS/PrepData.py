from sklearn import preprocessing

def trackdataprocessing(df, noOutlierdf, filterdf):
    print("Initial dataframe size", df.shape())
    print("dataframe without outliers", noOutlierdf.shape())
    print("dataframe without filtered out features", filterdf.shape())

import pandas as pd

def XYsplit(df, yLabels):
    ydf = df[yLabels]
    xdf = df.drop(columns = yLabels)
    return xdf, ydf

def normalize(xdf, ydf):
    x = xdf.values
    y = ydf.values.reshape(-1, 1)

    x_scaler = preprocessing.MinMaxScaler()
    x_normalized = x_scaler.fit_transform(x)

    y_scaler = preprocessing.MinMaxScaler()
    y_normalized = y_scaler.fit_transform(y)

    x_in = pd.DataFrame(x_normalized, columns = xdf.keys())
    y_in = pd.DataFrame(y_normalized, columns = ydf.keys())

    return (x_in, y_in), (x_scaler, y_scaler)

def crossvalidationSplit(x, y, batchCount=5):
    cutoffIndex = [0] + [int(x.shape[0]/batchCount * i) for i in range(1, batchCount)] if x.shape[0] % batchCount == 0\
        else [int(x.shape[0] / batchCount * i) for i in range(batchCount)]
    cutoffIndex += [x.shape[0]]
    xsets = [x.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]
    ysets = [y.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]

    return xsets, ysets

def TrainTestSplit(xSets, ySets, testSetIndex=1):

    xTrain = pd.concat([batch for batch in xSets if batch is not xSets[testSetIndex]])
    yTrain = pd.concat([batch for batch in ySets if batch is not ySets[testSetIndex]])
    return (xTrain, yTrain), (xSets[testSetIndex], ySets[testSetIndex])