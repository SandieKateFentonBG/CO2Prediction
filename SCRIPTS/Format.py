import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#model

#todo : convert this to a class - find better naming

# #todo : Should I scale my y values (targets)?

#DEFAULT VALUES
random_state = 42
test_size = 0.2

def formatDf(learningDf, xQuantLabels, xQualLabels, yLabels, yUnit):
    xTrainUnsc, xTestUnsc, yTrainDf, yTestDf = TrainTestSplitAsDf(learningDf, yLabels, yUnit)
    xTrainDf, xTestDf, MeanStdDf = scaleXDf(xTrainUnsc, xTestUnsc, xQuantLabels)
    trainDf = pd.concat([xTrainDf, yTrainDf], axis = 1)
    testDf = pd.concat([xTestDf, yTestDf], axis = 1)
    return trainDf, testDf, MeanStdDf

def TrainTestSplitAsDf(df, yLabels, yUnit=None): #, test_size=0.2, random_state=42):
    ydf = df[yLabels]
    xdf = df.drop(columns=yLabels)
    if yUnit:
        ydf = np.multiply(ydf, yUnit)

    XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size,
                                                    random_state=random_state)
    columnsNamesArr = xdf.columns.values
    XTrainDf = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
    XTestDf = pd.DataFrame(data=XTest, columns=columnsNamesArr)
    yTrainDf = pd.DataFrame(data=yTrain, columns=yLabels)
    yTestDf = pd.DataFrame(data=yTest, columns=yLabels)

    return XTrainDf, XTestDf, yTrainDf, yTestDf

def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd

def scaleXDf(XTrain, XTest, xQuantLabels):

    mydict = dict()
    for l in xQuantLabels:
        colMean, colStd = dfColMeanStd(XTrain, l)

        XTrain[l] = (XTrain[l] - colMean) / colStd
        XTest[l] = (XTest[l] - colMean) / colStd
        mydict[l] = [colMean, colStd]

    MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])

    return XTrain, XTest, MeanStdDf



