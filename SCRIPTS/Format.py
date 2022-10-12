import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split

from fast_ml.feature_selection import get_constant_features, recursive_feature_elimination

#model

#todo : convert this to a class - find better naming

# #todo : Should I scale my y values (targets)?

#DEFAULT VALUES
random_state = 42
test_size = 0.5 # proportion with validation
train_size= 0.8

def formatDf(learningDf, xQuantLabels, xQualLabels, yLabels, yUnit):

    XTrain, XValid, XTest, yTrain, yValid, yTest = dataSplitAsDf(learningDf, yLabels, yUnit)
    XTrain, XVal, XTest, MeanStdDf = scaleXDf(XTrain, XTest, XValid, xQuantLabels)
    trainDf = pd.concat([XTrain, yTrain], axis=1)
    validDf = pd.concat([XValid, yValid], axis=1)
    testDf = pd.concat([XTest, yTest], axis=1)

    return trainDf, validDf, testDf, MeanStdDf


# def TrainTestSplitAsDf(df, yLabels, yUnit=None): #, test_size=0.2, random_state=42):
#     ydf = df[yLabels]
#     xdf = df.drop(columns=yLabels)
#     if yUnit:
#         ydf = np.multiply(ydf, yUnit)
#
#     XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size,
#                                                     random_state=random_state)
#     columnsNamesArr = xdf.columns.values
#     XTrainDf = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
#     XTestDf = pd.DataFrame(data=XTest, columns=columnsNamesArr)
#     yTrainDf = pd.DataFrame(data=yTrain, columns=yLabels)
#     yTestDf = pd.DataFrame(data=yTest, columns=yLabels)
#
#     return XTrainDf, XTestDf, yTrainDf, yTestDf

def dataSplitAsDf(df, yLabels, yUnit=None):# train_size=0.8, valid_size=0.1, test_size=0.1 random_state=42):

    ydf = df[yLabels]
    xdf = df.drop(columns=yLabels)
    if yUnit:
        ydf = np.multiply(ydf, yUnit)

    XTrain, XRem, yTrain, yRem = train_test_split(xdf.values, ydf.values, train_size=train_size,
                                                    random_state=random_state)
    XVal, XTest, yVal, yTest = train_test_split(XRem, yRem, test_size=test_size,random_state=random_state)

    columnsNamesArr = xdf.columns.values
    XTrainDf = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
    XValidDf = pd.DataFrame(data=XVal, columns=columnsNamesArr)
    XTestDf = pd.DataFrame(data=XTest, columns=columnsNamesArr)
    yTrainDf = pd.DataFrame(data=yTrain, columns=yLabels)
    yValidDf = pd.DataFrame(data=yVal, columns=yLabels)
    yTestDf = pd.DataFrame(data=yTest, columns=yLabels)
    #
    return XTrainDf, XValidDf, XTestDf, yTrainDf, yValidDf, yTestDf

def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd

def scaleXDf(XTrain, XTest, XVal, xQuantLabels): # = None

    mydict = dict()
    for l in xQuantLabels:
        colMean, colStd = dfColMeanStd(XTrain, l)

        XTrain[l] = (XTrain[l] - colMean) / colStd
        XVal[l] = (XVal[l] - colMean) / colStd
        XTest[l] = (XTest[l] - colMean) / colStd
        mydict[l] = [colMean, colStd]

    MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])

    return XTrain, XVal, XTest, MeanStdDf



