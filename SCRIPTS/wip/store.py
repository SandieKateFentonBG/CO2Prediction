from RawData import RawData
from Data import *
from FilteredData import *
from Dashboard import *
from PrepData import *
from Model import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


"""
------------------------------------------------------------------------------------------------------------------------
1.DATA
------------------------------------------------------------------------------------------------------------------------
"""

"""Import libraries & Load data"""
rdat = RawData(csvPath, ';', 5, xQualLabels, xQuantLabels, yLabels)

"""Process data"""
dat = Data(rdat, scalers)
df = dat.asDataframe()

"""One hot encoding & Remove outliers"""
noOutlierDf = removeOutliers(df, labels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], cutOffThreshhold=3)

"""Correlation of variables & Feature selection"""
filterDf = filteredData(noOutlierDf, xQuantLabels, yLabels, plot = False)

trackDataProcessing(df, noOutlierDf, filterDf)
print(filterDf.keys())

"""Normalize and split Train-Test """

xdf, ydf = XYsplit(filterDf, yLabels)
(x, y), (xScaler, yScaler) = normalize(xdf, ydf)
xs, ys = crossvalidationSplit(x, y)
(xTrain, yTrain), (xTest, yTest) = TrainTestSplit(xs, ys,  testSetIndex=1)
#print(xTrain.keys())
"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""

#todo: move all this next part to another script - model assessor
#todo: add my own evaluator/accuracy
#todo: add the rotation/crossvalidation
#todo: understand how to regularize linear regression
#todo: look into kernel regresssion
#todo: look into lasso/ ridge - these also allow for feature selection -
# modularize feature selection either earlier with pearson or later with lasso
#todo: look into constrained optim
#todo: understand model saving

"""Build Model """

LRmodel = buildLinearRegressionModel()
#RFmodel = buildRandomForestModel()
#SVMmodel=buildSVMRegressionModel()
#XGBRmodel=buildXGBOOSTRegModel()
#Nmodel=buildNormalModel()
model = LRmodel  #todo: what is this for?

"""Train Model """
model.fit(xTrain, yTrain) #for keras : model.fit(xTrain, yTrain, epochs=8)

"""Evaluate Model """
model.score(xTest, yTest) #Return the coefficient of determination of the prediction
#model.evaluate(xTest, yTest) #for keras : Returns the loss value & metrics values for the model in test mode.

yPred = model.predict(xTest)
print(yPred)
yPredScaled = yScaler.inverse_transform(yPred)
yTestScaled = yScaler.inverse_transform(yTest)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(yPredScaled, yTestScaled)
print(mse)

plt.rcParams['figure.figsize'] = [18, 18]
l1, = plt.plot(yTestScaled, 'g')
l2, = plt.plot(yPredScaled, 'r', alpha=0.7)
plt.legend(['Ground truth', 'Predicted'])
plt.show()