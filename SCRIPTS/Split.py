

#LIBRARY IMPORTS
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd


class SplitDf:
    def __init__(self, df, xQuantLabels, xQualLabels, yLabels, yUnitFactor, targetLabels,
                 random_state, test_size, train_size, check_size, val_size, fixed_seed,
                 removeUnderrepresenteds, removeUnderrepresentedsDict):

        xDf = df.drop(columns=yLabels)
        yDf = np.multiply(df[yLabels], yUnitFactor)

        self.fixed_seed = fixed_seed # for XVal and XCheck
        self.check_size = check_size #10%
        self.val_size = val_size #10%
        self.ydf = yDf
        self.xdf = xDf
        self.ydf.rename(columns={yLabels[0]:targetLabels[0]})
        targetLabel = targetLabels[0]

        self.splitSets(targetLabels)
        self.scaleXDf(xQuantLabels)

        self.yLabel = targetLabel
        self.RDf = pd.concat([self.XR, self.yR], axis=1)
        self.valDf = pd.concat([self.XVal, self.yVal], axis=1)
        self.checkDf = pd.concat([self.XCheck, self.yCheck], axis=1)

        self.yR = self.RDf[self.yLabel]
        self.yVal = self.valDf[self.yLabel]
        self.yCheck = self.checkDf[self.yLabel]

        self.selector = 'NoSelector'

        self.droppedLabels = ''
        self.selectedLabels = list(self.XR.columns.values)


    def splitSets(self, yLabels):

        XCheck, XRem, yCheck, yRem = train_test_split(self.xdf.values, self.ydf.values, train_size=self.check_size,
                                                      random_state=self.fixed_seed) #10 %
        XVal, XR, yVal, yR = train_test_split(XRem, yRem, train_size=self.val_size, random_state=self.fixed_seed) #10 %
        columnsNamesArr = self.xdf.columns.values

        self.XR = pd.DataFrame(data=XR, columns=columnsNamesArr)
        self.XVal = pd.DataFrame(data=XVal, columns=columnsNamesArr)
        self.XCheck = pd.DataFrame(data=XCheck, columns=columnsNamesArr)

        self.yR = pd.DataFrame(data=yR, columns=yLabels)
        self.yVal = pd.DataFrame(data=yVal, columns=yLabels)
        self.yCheck = pd.DataFrame(data=yCheck, columns=yLabels)


    def scaleXDf(self, xQuantLabels):
        #todo : this scaling depends upon the data distribution - if data is gaussian, standardization / standard scaler makes sense
        # other options robust scale, power transform

        mydict = dict()
        for l in xQuantLabels:
            colMean, colStd = dfColMeanStd(self.XVal, l)

            self.XR[l] = (self.XR[l] - colMean) / colStd
            self.XVal[l] = (self.XVal[l] - colMean) / colStd
            self.XCheck[l] = (self.XCheck[l] - colMean) / colStd
            mydict[l] = [colMean, colStd]

        self.MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])

    def scaleDf(self, xQuantLabels, yLabels):
        #todo : this can be done instead of scaleXDf to ensure my target values have the same std -
        # in whuch case i would have to retransform my target values after prediction

        mydict = dict()
        for l in xQuantLabels:
            colMean, colStd = dfColMeanStd(self.XVal, l)

            self.XR[l] = (self.XR[l] - colMean) / colStd
            self.XVal[l] = (self.XVal[l] - colMean) / colStd
            self.XCheck[l] = (self.XCheck[l] - colMean) / colStd
            mydict[l] = [colMean, colStd]

        for l in yLabels:

            colMean, colStd = dfColMeanStd(self.yVal, l)
            self.yR[l] = (self.yR[l] - colMean) / colStd
            self.yVal[l] = (self.yVal[l] - colMean) / colStd
            self.yCheck[l] = (self.yCheck[l] - colMean) / colStd
            mydict[l] = [colMean, colStd]

        self.MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])


    def split_cv(self, X, y, k):

        kf = KFold(n_splits=k, random_state=None)
        kfolds = []
        for train_index, test_index in kf.split(X):
            fold = []

            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y[train_index], y[test_index]

            fold.append(X_train)
            fold.append(X_test)
            fold.append(y_train)
            fold.append(y_test)

            kfolds.append(fold)
            # kfolds = [fold,..., fold, fold]
            # fold =  [X_train,X_test, y_train, y_test]

        return kfolds


class CrossValDf:

    def __init__(self, SplitDf, fold, i):

        X_train, X_test, y_train, y_test = fold   #, ScaleMean, ScaleStd

        self.random_state = i+1 #!! in cv all folders are numbered 1-5
        self.fixed_seed = SplitDf.fixed_seed  # for XVal and XCheck
        self.check_size = SplitDf.check_size  # 10%
        self.val_size = SplitDf.val_size  # 10%
        self.ydf = SplitDf.ydf
        self.xdf = SplitDf.xdf

        self.yLabel = SplitDf.yLabel
        self.valDf = SplitDf.valDf
        self.checkDf = SplitDf.checkDf

        self.XTrain = X_train
        self.XTest = X_test
        self.XVal = SplitDf.XVal
        self.XCheck = SplitDf.XCheck

        self.yTrain = y_train
        self.yTest = y_test
        self.yVal = SplitDf.yVal
        self.yCheck = SplitDf.yCheck

        self.trainDf = pd.concat([self.XTrain, self.yTrain], axis=1)
        self.testDf = pd.concat([self.XTest, self.yTest], axis=1)

        self.selector = SplitDf.selector
        self.droppedLabels = SplitDf.droppedLabels
        self.selectedLabels = SplitDf.selectedLabels
        self.MeanStdDf = SplitDf.MeanStdDf

