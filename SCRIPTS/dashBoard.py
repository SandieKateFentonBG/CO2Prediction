"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
reference = 'New_TEST_RUN/'
#VISUALS
displayParams = {'showPlot': True, 'archive': True, 'showCorr' : True, 'TargetMinMaxVal': [0, 800], 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}

# "csvPath": "C:/Users/sfenton/Code/Repositories/CO2Prediction/DATA/210413_PM_CO2_data",
#                  "outputPath":'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/', 'showCorr': False,
#                 'showResults' : False, 'showPlot' : False, 'archive': True, 'reference': 'New_TEST_RUN_2',
#                  , 'roundNumber': 3,
#                  'residualsYLim': [-500, 500], 'residualsXLim': [0, 800], 'fontsize': 14, 'random_state' : random

"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""

"""
________________________________________________________________________________________________________________________
Price&Myers V1
________________________________________________________________________________________________________________________
"""

#PATH
DBpath = "C:/Users/sfenton/Code/Repositories/CO2Prediction/"
DBname = "210413_PM_CO2_data"
#folder =  DATA / RESULTS / SCRIPTS

#DEFAULT LABELS
xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']#
xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']
yLabels = ['Calculated tCO2e_per_m2']


DBdelimiter = ';'
DBfirstLine = 5

processingParams = {'scaler': 'MinMaxScaler', 'cutOffThreshhold' : 3, 'lowThreshold' : 0.15, 'highThreshold' : 1,
                    'yScale' : False, 'yUnitFactor' : 1000 , 'targetLabels' : ['kgCO2e/m2']}

"""
________________________________________________________________________________________________________________________
Price&Myers V2
________________________________________________________________________________________________________________________
"""
#TODO - TODO