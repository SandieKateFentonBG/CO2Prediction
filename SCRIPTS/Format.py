import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split

from fast_ml.feature_selection import get_constant_features, recursive_feature_elimination

#model

# #todo : Should I scale my y values (targets)?

#DEFAULT VALUES
random_state = 42
test_size = 0.5 # proportion with validation
train_size= 0.8


def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd

class formatedDf:
    def __init__(self, df, xQuantLabels, xQualLabels, yLabels, yUnitFactor):

        xDf = df.drop(columns=yLabels)
        yDf = np.multiply(df[yLabels], yUnitFactor)

        self.random_state = 42
        self.test_size = 0.5  # proportion with validation
        self.train_size = 0.8
        self.ydf = yDf
        self.xdf = xDf

        self.dataSplitAsDf(yLabels)
        self.scaleXDf(xQuantLabels)

        self.trainDf = pd.concat([self.XTrain, self.yTrain], axis=1)
        self.valDf = pd.concat([self.XVal, self.yVal], axis=1)
        self.testDf = pd.concat([self.XTest, self.yTest], axis=1)

    def dataSplitAsDf(self, yLabels):  # train_size=0.8, valid_size=0.1, test_size=0.1 random_state=42):


        XTrain, XRem, yTrain, yRem = train_test_split(self.xdf.values, self.ydf.values, train_size=self.train_size,
                                                      random_state=self.random_state)
        XVal, XTest, yVal, yTest = train_test_split(XRem, yRem, test_size=self.test_size, random_state=self.random_state)

        columnsNamesArr = self.xdf.columns.values

        self.XTrain = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
        self.XVal = pd.DataFrame(data=XVal, columns=columnsNamesArr)
        self.XTest = pd.DataFrame(data=XTest, columns=columnsNamesArr)
        self.yTrain = pd.DataFrame(data=yTrain, columns=yLabels)
        self.yVal = pd.DataFrame(data=yVal, columns=yLabels)
        self.yTest = pd.DataFrame(data=yTest, columns=yLabels)

    def scaleXDf(self, xQuantLabels):  # = None

        mydict = dict()
        for l in xQuantLabels:
            colMean, colStd = dfColMeanStd(self.XTrain, l)

            self.XTrain[l] = (self.XTrain[l] - colMean) / colStd
            self.XVal[l] = (self.XVal[l] - colMean) / colStd
            self.XTest[l] = (self.XTest[l] - colMean) / colStd
            mydict[l] = [colMean, colStd]

        self.MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])






