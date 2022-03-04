# PATH

csvPath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/P&M Carbon Database for Release 11-02-2022"
outputPath = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'
#220301_Stdsc_nofilt
displayParams = {"csvPath":"C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/P&M Carbon Database for Release 11-02-2022",
                 "outputPath":'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/', 'showCorr': False,
                 'showResults': False, 'showPlot' : False, 'archive': True, 'reference': 'PMv2_fccb-lt015-2',
                 'Target': 'Calculated tCO2e_per_m2', 'TargetMinMaxVal' : [0, 10000], 'roundNumber': 3,
                 'residualsYLim': [-5000, 5000], 'residualsXLim': [0, 10000], 'fontsize': None}
# DATA

xQualLabels = ['Calculation Design Stage','Value Type','Project Sector', 'Construction Type','Passivhaus?',
               'Basement Type', 'Foundation Type', 'Ground Floor Type', 'Superstructure Type']#,'UK District'
xQuantLabels = ['GIFA (m2)','Storeys (#)']
yLabels = ['A1-A5 Rate (kgCO2e/m2)'] #'Carbon A1-A5 (kgCO2e)',

 #
# if higher orders : baseLabels = ['GIFA (m2)_exp1', 'Storeys_exp1', 'Typical Span (m)_exp1','Typ Qk (kN_per_m2)_exp1']

# FORMAT

processingParams = {'scaler': 'MinMaxScaler', 'cutOffThreshhold' : 3, 'lowThreshold' : 0.15, 'highThreshold' : 1,
                    'removeLabels' : ['Passivhaus?_No','Project Value (œm)'], 'baseLabels' : xQuantLabels } #[], 'Foundations_Raft' 'scaler': None, 'MinMaxScaler', 'StandardScaler'
#main : 'Basement_None', 'Sector_Industrial', 'Foundations_', 'Ground Floor_Raft'


# PARAMS
import numpy as np
modelingParams = {'test_size': 0.2, 'random_state' : 4, 'RegulVal': list(10.0**np.arange(-4,4)), 'epsilonVal': list(10.0**np.arange(-4,4)),
                  'accuracyTol': 0.15, 'CVFold': None, 'rankGridSearchModelsAccordingto' : 'r2', 'plotregulAccordingTo' : 'paramMeanMSETest'} #paramMeanR2Test
powers = {}
mixVariables = []
#[0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100, 200, 500]
# list(10.0**np.arange(-4,4)) #list(np.arange(-5,7))
# powers = {'GIFA (m2)': [1/3, 0.5, 1, 2, 3], 'Storeys':[1/3, 0.5, 1, 2, 3], 'Typical Span (m)': [1/3, 0.5, 1, 2, 3],'Typ Qk (kN_per_m2)': [1/3, 0.5, 1, 2, 3]}#,
# 'GIFA (m2)': [1, 0.5], 'Storeys':[1, 2, 3], 0.5 ,, 1/3, 1/4  1/5, 1/6,'Storeys':[1, 2, 3] ,'Typical Span (m)': [1, 2, 3],'Typ Qk (kN_per_m2)': [1, 2, 3] }
#[['GIFA (m2)','Storeys']], 'Typ Qk (kN_per_m2)'],,['Sector','Type','Basement','Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating' ]], ['Typical Span (m)'],['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], ['Sector_Residential','Basement_Partial Footprint']

# MODEL
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


linearReg = {'model' : LinearRegression(), 'param' : None, 'Linear' : True} #why doies this not have a regul param?
lassoReg = {'model' : Lasso() , 'param': 'alpha', 'Linear' : True} # for overfitting
lassoRegNorm = {'model' : Lasso(normalize=True), 'param': 'alpha', 'Linear' : True}
ridgeReg = {'model' : Ridge(), 'param': 'alpha', 'Linear' : True}
elasticNetReg = {'model' : ElasticNet(), 'param': 'alpha', 'Linear' : True}
elasticNetRegNorm = {'model' : ElasticNet(normalize=True), 'param': 'alpha', 'Linear' : True}
supportVectorLinReg = {'model' : SVR(kernel='linear'), 'param': 'C', 'Linear' : True}
supportVectorRbfReg = {'model' : SVR(kernel='rbf'), 'param': 'C', 'Linear' : False}
supportVectorPolReg = {'model' : SVR(kernel='poly'), 'param': 'C', 'Linear' : False}
kernelRidgeLinReg = {'model' : KernelRidge(kernel='linear'), 'param': 'alpha', 'Linear' : False}
kernelRidgeRbfReg = {'model' : KernelRidge(kernel='rbf'), 'param': 'alpha', 'Linear' : False}
kernelRidgePolReg = {'model' : KernelRidge(kernel='polynomial'), 'param': 'alpha', 'Linear' : False}

# modelsa = [linearReg, lassoReg, ridgeReg, elasticNetReg, supportVectorLinReg, supportVectorRbfReg, supportVectorPolReg,
#         kernelRidgeLinReg, kernelRidgeRbfReg, kernelRidgePolReg]
models = [linearReg, lassoReg, ridgeReg, elasticNetReg,
          kernelRidgeLinReg, kernelRidgeRbfReg, kernelRidgePolReg,
          supportVectorLinReg, supportVectorRbfReg, supportVectorPolReg]

# supportVectorLinRege = {'model' : SVR(kernel='linear'), 'param': 'epsilon', 'Linear' : True}
# supportVectorRbfRege = {'model' : SVR(kernel='rbf'), 'param': 'epsilon', 'Linear' : False}
# supportVectorPolRege = {'model' : SVR(kernel='poly'), 'param': 'epsilon', 'Linear' : False}
#
# models = [supportVectorLinRege, supportVectorRbfRege, supportVectorPolRege]