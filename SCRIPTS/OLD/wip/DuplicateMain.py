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

# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
# poly.fit_transform(dat)

"""One hot encoding & Remove outliers"""
noOutlierDf = removeOutliers(df, labels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], cutOffThreshhold=3)

"""Correlation of variables & Feature selection"""
filterDf = filteredData(noOutlierDf, xQuantLabels, yLabels, plot = False, lt = 0.1)

"""Remove Multi-correlated Features """
# filterDf2 = filteredData(noOutlierDf, xQuantLabels, yLabels, plot = False, lt = 0.1,
#                          removeLabels=['Basement_None', 'Foundations_Raft'])

trackDataProcessing(df, noOutlierDf, filterDf) # filterDf2

"""
------------------------------------------------------------------------------------------------------------------------
2. MODEL
------------------------------------------------------------------------------------------------------------------------
"""
#single run
method = 'Nmodel' #'LRmodel', 'RFmodel','SVMmodel', 'LRmodel'
run = execute(filterDf, yLabels, method, epochs=5, singleTest = 1, display = True)
#
# #multiple run
# methods = ['LRmodel', 'SVMmodel', 'RFmodel', 'XGBmodel']
# for m in methods:
#     run = execute(filterDf, yLabels, m, epochs=None, singleTest=1, display = False)
#     print('Method:', run['method'], 'Evaluation:', run['evalu'], 'Accuracy:', run['acc'],'MSE:', run['mse'] )
#     # plot(run['yTest'],run['model'].predict(run['xTest']))

# for m in methods:
#     run = execute(filterDf2, yLabels, m, epochs=None, singleTest=1, display = False)
#     print('Method:', run['method'], 'Evaluation:', run['evalu'], 'Accuracy:', run['acc'],'MSE:', run['mse'] )
#     # plot(run['yTest'],run['model'].predict(run['xTest']))
#todo : normal model not working
#todo : results very low
#
# #todo: understand how to regularize linear regression
# #todo: understand attributes of model classes
# #todo: look into kernel regresssion
# #todo: look into lasso/ ridge - these also allow for feature selection -
# # modularize feature selection either earlier with pearson or later with lasso
# #todo: look into constrained optim
# #todo: understand model saving
#todo : add higher powers/power up
#todo: support vector regression, kernel regression, lasso regression and fully connected neural networks
#
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# https://scikit-learn.org/stable/auto_examples/miscellaneous/plot_kernel_ridge_regression.html
# svr = GridSearchCV(
#     SVR(kernel="rbf", gamma=0.1),
#     param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)},
# )
#
# kr = GridSearchCV(
#     KernelRidge(kernel="rbf", gamma=0.1),
#     param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3], "gamma": np.logspace(-2, 2, 5)},
# )
