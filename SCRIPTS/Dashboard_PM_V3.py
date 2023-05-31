# import numpy as np
#
# """
# ________________________________________________________________________________________________________________________
# RUN
# ________________________________________________________________________________________________________________________
# """
# #change when running a test
#
# acronym = 'PM_V3'
#
# displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None}
#
# studyParams = {"sets": [[['Embodied_Carbon[kgCO2e_m2]'],'EC','TestR2']], 'randomvalues': list(range(30, 40)),
#                "Regressors": ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']}
#
#
# """
# ________________________________________________________________________________________________________________________
# DATABASE
# ________________________________________________________________________________________________________________________
# """
# #parameters specific to the database processed
#
# """
# Price&Myers V3
# """
#
#
#
#
#
#
# MyPred_Sample = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "Test-Wood",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym} #Test-Concrete
#
# DB_Values = {"DBpath" : "K:/Temp/Sandie/Pycharm/", "DBname" : "P&M_Carbon_Database_2023",
#              "DBdelimiter" : ';', "DBfirstLine" : 5, 'acronym' : acronym}
#
# xQualLabels = ['Calculation Design Stage','Location','Value Type','Project Sector', 'Type', 'Passivhaus', 'Basement',
#                'Foundation Type', 'Ground Floor Type', 'Superstructure Type', 'Cladding Type', 'Fire Rating']#
# xQuantLabels = ['GIFA (m2)', 'Calculation Year', 'Project Value (poundm)','Storeys (#)']
#
# RemoveOutliersFrom = xQuantLabels
# yLabels = ['A1-C4 Rate (kgCO2e/m2)']
#
# # ['A1-A3 Rate (kgCO2e/m2)',	'A1-A5 Rate (kgCO2e/m2)',	'A1-C4 Rate (kgCO2e/m2)',	'A1-D Rate (kgCO2e/m2)',	'SCORS Rating']
#
# FORMAT_Values = {'yUnitFactor': 1, 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}
#
# #'yUnitFactor' converts from yLabel unit to target Label unit:
# # ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000
#
# """
# ________________________________________________________________________________________________________________________
# PROCESSING
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for database processing
#
#
# PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : float(1/8), 'train_size': float(7/8), 'check_size': 0.1, 'val_size': float(1/9),
#                 'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 40,
#                      'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'accuracyTol' : 0.10, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}
#
# #todo : check 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]
#
#
# """
# ________________________________________________________________________________________________________________________
# GRIDSEARCH
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for gridsearch opimization
#
# """
# ________________________________________________________________________________________________________________________
# FEATURE SELECTION
# ________________________________________________________________________________________________________________________
# """
#
#
# RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : 'list(np.arange(10, len(baseFormatedDf.XTrain)-10, 10))',
#               'RFE_process' : 'short', 'output_feature_count':'rfeCV'}
#
#
# """
# ________________________________________________________________________________________________________________________
# MODEL
# ________________________________________________________________________________________________________________________
# """
# GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
#             'regul_range' : list(10.0 ** np.arange(-4, 4)),
#             'influence_range' : list(10.0 ** np.arange(-4, 4)),
#             'degree' : [2, 3, 4],
#             'margin_range' : list(10.0 ** np.arange(-4, 4)),
#             'kernel_list' : ['poly', 'linear', 'rbf']}
#
#
# LR_param_grid={'alpha': GS_VALUES['regul_range']}
# KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                                                 'coef0' : GS_VALUES['coef0_range']}
# SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                 'epsilon': GS_VALUES['margin_range'],  'coef0' : GS_VALUES['coef0_range']}
#
# """
# ________________________________________________________________________________________________________________________
# HYPERPARAM
# ________________________________________________________________________________________________________________________
# """
#
# # # Example for Single Hyperparameter plot
# KRR_param_grid1={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['linear']}
# KRR_param_grid2={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['polynomial']}
# KRR_param_grid3={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['rbf']}
#
#
# """
# ________________________________________________________________________________________________________________________
# FEATURE SELECTION
# ________________________________________________________________________________________________________________________
# """
#
# BLE_VALUES = {'NBestScore': 'TestR2', 'NCount' : 10, 'Regressor' : 'LR_RIDGE', 'OverallBest' : True,
#               'BestModelNames' : None} #'TestAcc'SVR_RBF
#
#
