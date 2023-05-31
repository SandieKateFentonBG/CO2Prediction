import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def dfColMeanStd(df, colName):
    colMean = df[colName].mean()
    colStd = df[colName].std()

    return colMean, colStd

class formatedDf:
    def __init__(self, df, xQuantLabels, xQualLabels, yLabels, yUnitFactor, targetLabels,
                 random_state, test_size, train_size, check_size, val_size, fixed_seed):

        xDf = df.drop(columns=yLabels)
        yDf = np.multiply(df[yLabels], yUnitFactor)

        self.random_state = random_state # for Xtest and Xtrain
        self.fixed_seed = fixed_seed # for XVal and XCheck
        self.test_size = test_size  # proportion with validation
        self.train_size = train_size
        self.check_size = check_size
        self.val_size = val_size
        self.ydf = yDf
        self.xdf = xDf
        self.ydf.rename(columns={yLabels[0]:targetLabels[0]})
        targetLabel = targetLabels[0]

        self.dataSplitAsDf(targetLabels)
        self.scaleXDf(xQuantLabels)

        self.yLabel = targetLabel
        self.trainDf = pd.concat([self.XTrain, self.yTrain], axis=1)
        self.valDf = pd.concat([self.XVal, self.yVal], axis=1)
        self.testDf = pd.concat([self.XTest, self.yTest], axis=1)
        self.checkDf = pd.concat([self.XCheck, self.yCheck], axis=1)

        # todo : !
        self.yTrain = self.trainDf[self.yLabel]
        self.yVal = self.valDf[self.yLabel]
        self.yTest = self.testDf[self.yLabel]
        self.yCheck = self.checkDf[self.yLabel]
        # todo : !

        #self.MeanStdDf
        self.selector = 'NoSelector'
        self.droppedLabels = ''
        self.selectedLabels = list(self.XTrain.columns.values)

    def dataSplitAsDf(self, yLabels):  # train_size=0.8, valid_size=0.1, test_size=0.1 random_state=42):


        # XTrain, XRem, yTrain, yRem = train_test_split(self.xdf.values, self.ydf.values, train_size=self.train_size,
        #                                               random_state=self.random_state)
        # XVal, XTest, yVal, yTest = train_test_split(XRem, yRem, test_size=self.test_size, random_state=self.random_state)

        """
        XCheck, yCheck, XVal, yVal > same split for every CV - used for Feature selection and Blender
        XTrain, XTest, yTrain, yTest > different split for every CV - used for Learning

        """
        XCheck, XRem, yCheck, yRem = train_test_split(self.xdf.values, self.ydf.values, train_size=self.check_size,
                                                      random_state=self.fixed_seed) #10 %
        XVal, XR, yVal, yR = train_test_split(XRem, yRem, train_size=self.val_size, random_state=self.fixed_seed) #10 %
        XTrain, XTest, yTrain, yTest = train_test_split(XR, yR, train_size=self.train_size, random_state=self.random_state) #70 % - 10 %

        columnsNamesArr = self.xdf.columns.values

        self.XTrain = pd.DataFrame(data=XTrain, columns=columnsNamesArr)
        self.XVal = pd.DataFrame(data=XVal, columns=columnsNamesArr)
        self.XTest = pd.DataFrame(data=XTest, columns=columnsNamesArr)
        self.XCheck = pd.DataFrame(data=XCheck, columns=columnsNamesArr)


        #todo : ! thius  was chnaged so that all learning dfs have same content  >
        # we want type 'pandas.core.series.Series rather than pandas.core.frame.DataFrame
        # could cause issues

        # todo : will be reformatted just after
        self.yTrain = pd.DataFrame(data=yTrain, columns=yLabels)
        self.yVal = pd.DataFrame(data=yVal, columns=yLabels)
        self.yTest = pd.DataFrame(data=yTest, columns=yLabels)
        self.yCheck = pd.DataFrame(data=yCheck, columns=yLabels)






    def scaleXDf(self, xQuantLabels):
        #todo : this scaling depends upon the data distribution - if data is gaussian, standardization / standard scaler makes sense
        # other options robust scale, power transform

        mydict = dict()
        for l in xQuantLabels:
            colMean, colStd = dfColMeanStd(self.XVal, l)

            self.XTrain[l] = (self.XTrain[l] - colMean) / colStd
            self.XVal[l] = (self.XVal[l] - colMean) / colStd
            self.XTest[l] = (self.XTest[l] - colMean) / colStd
            self.XCheck[l] = (self.XCheck[l] - colMean) / colStd
            mydict[l] = [colMean, colStd]

        self.MeanStdDf = pd.DataFrame(data=mydict, index=['mean', 'std'])







