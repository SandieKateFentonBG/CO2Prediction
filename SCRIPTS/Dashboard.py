# PATH

csvPath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data"
outputPath = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'

displayParams = {"csvPath": "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data", "outputPath":'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/',
                  'showResults' : True, 'showPlot' : True, 'archive': True, 'reference': '220208'}

# DATA

xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']#
xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']

yLabels = ['Calculated tCO2e_per_m2'] #'Calculated Total tCO2e',

scaling = False #True

scalers = {'scaling': False, 'method': 'skl_standardscale','positiveValue': 5, 'qinf': 0.25, 'qsup': 0.75 }#methods : 'standardize', 'robustscale', 'skl_robustscale'

# PARAMS

# modelingParams = {"regularisation": 20, "tolerance": 0.1, "method": "accuracy"} #'mse'; "mae"
modelingParams = {'kernelVal': ['linear', 'rbf', 'polynomial'], 'RegulVal' : [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100]}

powers = {}#'GIFA (m2)': [1], 'Storeys':[1], 'Typical Span (m)': [1],'Typ Qk (kN_per_m2)': [1],
# 'GIFA (m2)': [1, 0.5], 'Storeys':[1, 2, 3], 0.5 ,, 1/3, 1/4  1/5, 1/6,'Storeys':[1, 2, 3] ,'Typical Span (m)': [1, 2, 3],'Typ Qk (kN_per_m2)': [1, 2, 3] }
mixVariables = [] #[['GIFA (m2)','Storeys']], 'Typ Qk (kN_per_m2)'],,['Sector','Type','Basement','Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating' ]], ['Typical Span (m)'],['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], ['Sector_Residential','Basement_Partial Footprint']

# MODEL

