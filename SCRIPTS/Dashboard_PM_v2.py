# import numpy as np
#
# """
# ________________________________________________________________________________________________________________________
# RUN
# ________________________________________________________________________________________________________________________
# """
# #change when running a test
#
# displayParams = {"reference" : 'PM_V2_simplecheck-2/', 'showPlot': False, 'archive': True, 'showCorr' : False}
#
# """
# ________________________________________________________________________________________________________________________
# DATABASE
# ________________________________________________________________________________________________________________________
# """
# #parameters specific to the database processed
#
# """
# Price&Myers V2
# """
# DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/", "DBname" : "P&M Carbon Database _11-02-2022_sf",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 }
# # Project Value (œm)
# xQualLabels = ['Calculation Design Stage', 'Value Type', 'Project Sector', 'Construction Type','Passivhaus',
#                'Basement Type', 'Foundation Type', 'Ground Floor Type', 'Superstructure Type', 'Cladding Type'] #
# xQuantLabels = ['GIFA (m2)', 'Storeys (#)',  'Project Value (poundm)']#
# RemoveOutliersFrom = xQuantLabels
# yLabels = ['A1-A5 Rate (kgCO2e/m2)']
#
# # ['Carbon A1-A3 (kgCO2e)','Carbon A1-A5 (kgCO2e)','Carbon A1-C4 (kgCO2e)','Carbon A1-D (kgCO2e)','A1-A3 Rate (kgCO2e/m2)',
# # 'A1-A5 Rate (kgCO2e/m2)	','A1-C4 Rate (kgCO2e/m2)','A1-D Rate (kgCO2e/m2)']
#
#
#
#
# FORMAT_Values = {'yUnitFactor': 1, 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}
#
# """
# ________________________________________________________________________________________________________________________
# PROCESSING
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for database processing
#
# PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : 0.5, 'train_size': 0.8,
#                 'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1,
#                      'corrHighThreshhold' : 0.65, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}
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
# RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : 'list(np.arange(10, len(baseFormatedDf.XTrain)-10, 10))'} #[5, 10, 15, 20, 25]
#
#
#
# """
# ________________________________________________________________________________________________________________________
# MODEL
# ________________________________________________________________________________________________________________________
# """
# GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
#             'regul_range' : list(10.0 ** np.arange(-2, 2)),
#             'influence_range' : list(10.0 ** np.arange(-2, 2)),
#             'degree' : [2, 3],
#             'margin_range' : list(10.0 ** np.arange(-2, 2)),
#             'kernel_list' : [ 'linear', 'rbf']}
#
#
# # GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
# #             'regul_range' : list(10.0 ** np.arange(-4, 4)),
# #             'influence_range' : list(10.0 ** np.arange(-4, 4)),
# #             'degree' : [2, 3, 4],
# #             'margin_range' : list(10.0 ** np.arange(-4, 4)),
# #             'kernel_list' : ['poly', 'linear', 'rbf']}
#
# """
# ________________________________________________________________________________________________________________________
# V1
# ________________________________________________________________________________________________________________________
# """
#
# # LR_param_grid={'alpha': GS_VALUES['regul_range']}
# # KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
# #                                                 'kernel' : GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}
# # SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
# #                 'epsilon': GS_VALUES['margin_range'], 'kernel': GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}
#
# """
# ________________________________________________________________________________________________________________________
# V2 - kernels not in params
# ________________________________________________________________________________________________________________________
# """
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
