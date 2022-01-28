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
#single run
# method = 'LRmodel' #'RFmodel','SVMmodel', 'LRmodel'
# run = execute(filterDf,yLabels, method, epochs=None, singleTest = 1, display = True)

#multiple run
methods = ['LRmodel', 'SVMmodel', 'RFmodel', 'XGBmodel']
for m in methods:
    run = execute(filterDf, yLabels, m, epochs=None, singleTest=1, display = False)
    print('Method:', run['method'], 'Evaluation:', run['evalu'], 'Accuracy:', run['acc'],'MSE:', run['mse'] )
    # plot(run['yTest'],run['model'].predict(run['xTest']))


#todo: understand how to regularize linear regression
#todo: understand attributes of model classes
#todo: look into kernel regresssion
#todo: look into lasso/ ridge - these also allow for feature selection -
# modularize feature selection either earlier with pearson or later with lasso
#todo: look into constrained optim
#todo: understand model saving

