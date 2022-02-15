# PATH

csvPath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data"
outputPath = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'

displayParams = {"csvPath": "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data", "outputPath":'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/',
                  'showResults' : True, 'showPlot' : False, 'archive': False, 'reference': '220215', 'roundNumber' : 3}

# DATA

xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']#
xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']
yLabels = ['Calculated tCO2e_per_m2'] #'Calculated Total tCO2e',

# FORMAT

processingParams = {'scaler': 'MinMaxScaler', 'cutOffThreshhold' : 3, 'lowThreshold' : 0.1, 'highThreshold' : 0.5,
                    'removeLabels' : ['Basement_None', 'Foundations_Raft']} #, 'method': 'skl_standardscale','positiveValue': 5, 'qinf': 0.25, 'qsup': 0.75 #methods : 'standardize', 'robustscale', 'skl_robustscale'
#'scaler': None, 'MinMaxScaler', 'StandardScaler'

# PARAMS

# modelingParams = {"regularisation": 20, "tolerance": 0.1, "method": "accuracy"} #'mse'; "mae"
modelingParams = {'test_size': 0.2, 'random_state' : 8, 'RegulVal': [0.001, 0.01, 0.1, 1, 5, 10, 20, 50, 100], 'TestIndex': 1, 'CVFold': None}#'kernelVal': ['linear', 'rbf', 'polynomial'],
# None,5

powers = {}
# powers = {'GIFA (m2)': [1/3, 0.5, 1, 2, 3], 'Storeys':[1/3, 0.5, 1, 2, 3], 'Typical Span (m)': [1/3, 0.5, 1, 2, 3],'Typ Qk (kN_per_m2)': [1/3, 0.5, 1, 2, 3]}#,
# 'GIFA (m2)': [1, 0.5], 'Storeys':[1, 2, 3], 0.5 ,, 1/3, 1/4  1/5, 1/6,'Storeys':[1, 2, 3] ,'Typical Span (m)': [1, 2, 3],'Typ Qk (kN_per_m2)': [1, 2, 3] }
mixVariables = [] #[['GIFA (m2)','Storeys']], 'Typ Qk (kN_per_m2)'],,['Sector','Type','Basement','Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating' ]], ['Typical Span (m)'],['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)'], ['Sector_Residential','Basement_Partial Footprint']

# MODEL

#scaler
#CVFold