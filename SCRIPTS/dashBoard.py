"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test

displayParams = {"reference" : 'TEST_RUN/', 'showPlot': True, 'archive': True, 'showCorr' : True}
# displayParams = {'showPlot': True, 'archive': True, 'showCorr' : True, 'TargetMinMaxVal': [0, 800], 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}


"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
Price&Myers V1
"""
DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/", "DBname" : "210413_PM_CO2_data",
             "DBdelimiter" : ';', "DBfirstLine" : 5 }

xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']
xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']
yLabels = ['Calculated tCO2e_per_m2']

FORMAT_Values = {'yUnitFactor' : 1000 , 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 800]}

"""
________________________________________________________________________________________________________________________
PROCESSING
________________________________________________________________________________________________________________________
"""
#parameters chosen for database processing

PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : 0.5, 'train_size': 0.8,
                'corrMethod' : "spearman", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'corrHighThreshhold' : 0.65,
                     'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]} #'scaler': 'MinMaxScaler', , 'lowThreshold' : 0.15, 'highThreshold' : 1,'yScale' : False,

HYPERPARAMETERS = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : [5, 10, 15, 20, 25]}
"""
________________________________________________________________________________________________________________________
WRAPPING
________________________________________________________________________________________________________________________
"""



"""
________________________________________________________________________________________________________________________
Price&Myers V2
________________________________________________________________________________________________________________________
"""
#TODO - TODO