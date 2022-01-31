# PATH

csvPath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data"
# outputPath = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/'

displayParams = {"csvPath": "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data", "outputPath":'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/',
                 'showAccuracy': True, 'showThetas': True, 'showAll' : False, 'showPlot' : True, 'archive': False, 'reference': '211104'}

# DATA

xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']#
xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']

yLabels = ['Calculated tCO2e_per_m2'] #'Calculated Total tCO2e',

scaling = False #True

scalers = {'scaling': False, 'method': 'skl_standardscale','positiveValue': 5, 'qinf': 0.25, 'qsup': 0.75 }#methods : 'standardize', 'robustscale', 'skl_robustscale'

# MODEL

modelingParams = {"regularisation": 20, "tolerance": 0.1, "method": "accuracy"} #'mse'; "mae"

