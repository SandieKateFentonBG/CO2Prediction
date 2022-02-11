from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd



def unscale(elem, scaler, scalerParam):

    if scalerParam:
        return pd.DataFrame(scaler.inverse_transform(elem), columns = elem.keys())
    else :
        return elem

def XScaleYSplit(df, yLabels, scalerParam):
    ydf = df[yLabels]
    xdf = df.drop(columns = yLabels)
    xScaler = None
    if scalerParam:
        if scalerParam == 'MinMaxScaler':
            xScaler = preprocessing.MinMaxScaler()
            x_normalized = xScaler.fit_transform(xdf)
            xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
        if scalerParam == 'StandardScaler':
            xScaler = preprocessing.StandardScaler()
            x_normalized = xScaler.fit_transform(xdf)
            xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
        xdf = xScaled
    return xdf, ydf, xScaler

def TrainTest(xdf, ydf, test_size=0.2, random_state=8):

    XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size, random_state=random_state)
    return XTrain, XTest, yTrain, yTest



def crossvalidationSplit(x, y, batchCount=5):
    cutoffIndex = [0] + [int(x.shape[0]/batchCount * i) for i in range(1, batchCount)] if x.shape[0] % batchCount == 0\
        else [int(x.shape[0] / batchCount * i) for i in range(batchCount)]
    cutoffIndex += [x.shape[0]]
    xsets = [x.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]
    ysets = [y.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]

    return xsets, ysets

def TrainTestSets(filterDf, yLabels, scalerParam):
    """Normalize and split Train-Test """
    xdf, ydf, xScaler = XScaleYSplit(filterDf, yLabels, scalerParam)
    xs, ys = crossvalidationSplit(xdf, ydf)
    return xs, ys, xScaler

def TrainTestDf(xSets, ySets, testIdParam=1):

    xTrain = pd.concat([batch for batch in xSets if batch is not xSets[testIdParam]])
    yTrain = pd.concat([batch for batch in ySets if batch is not ySets[testIdParam]])
    return (xTrain, yTrain), (xSets[testIdParam], ySets[testIdParam])

def TrainTestArray(filterDf, yLabels, testSetIndex):
    #todo : shuffle data...
    xs, ys, xScaler = TrainTestSets(filterDf, yLabels)
    (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xs, ys, testSetIndex)
    return (xTrain.values, yTrain.values.reshape(-1, 1)), (xTest.values, yTest.values.reshape(-1, 1))

# def scale(df, scalerParam):
#
#     if scalerParam:
#         x = df.values
#         if scalerParam == 'MinMaxScaler':
#             Scaler = preprocessing.MinMaxScaler()
#             x_normalized = Scaler.fit_transform(x)
#             xScaled = pd.DataFrame(x_normalized, columns = df.keys())
#         if scalerParam == 'MinMaxScaler':
#             Scaler = preprocessing.StandardScaler()
#             x_normalized = Scaler.fit_transform(x)
#             xScaled = pd.DataFrame(x_normalized, columns = df.keys())
#         return (xScaled, Scaler)
#     else :
#         return (df, scalerParam)
