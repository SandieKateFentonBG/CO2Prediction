from RawData import RawData
from Data import *
from FilteredData import *
from Dashboard import *
from ModelAssessor import *
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
#print(filterDf.keys())

"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""
method = 'LRmodel' #'RFmodel','SVMmodel', 'LRmodel'


run = execute(filterDf,yLabels, method, epochs=None, singleTest = 1)
plot(run['yTest'],run['model'].predict(run['xTest']))
xScaler, yScaler = run['scaler']
yPredScaled = yScaler.inverse_transform(run['model'].predict(run['xTest'].reshape(-1, 1)))
yTestScaled = yScaler.inverse_transform(run['yTest'])
plot(yTestScaled,yPredScaled)

# methods = ['LRmodel', 'SVMmodel', 'RFmodel', 'XGBmodel']
# for m in methods:
#     run = execute(filterDf, yLabels, m, epochs=None, singleTest=1)
#     print('Method:', run['method'], 'Evaluation:', run['evalu'], 'Accuracy:', run['acc'],'MSE:', run['mse'] )
#     plot(run['yTest'],run['model'].predict(run['xTest']))


#todo: understand how to regularize linear regression
#todo: understand attributes of model classes
#todo: look into kernel regresssion
#todo: look into lasso/ ridge - these also allow for feature selection -
# modularize feature selection either earlier with pearson or later with lasso
#todo: look into constrained optim
#todo: understand model saving



"""yPred = model.predict(xTest)
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
plt.show()"""