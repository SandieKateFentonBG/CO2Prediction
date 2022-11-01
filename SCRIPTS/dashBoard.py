# import numpy as np
#
# """
# ________________________________________________________________________________________________________________________
# RUN
# ________________________________________________________________________________________________________________________
# """
# #change when running a test
#
# displayParams = {"reference" : 'PM_V1_simplecheck-2/', 'showPlot': False, 'archive': True, 'showCorr' : True}
#
# """
# ________________________________________________________________________________________________________________________
# DATABASE
# ________________________________________________________________________________________________________________________
# """
# #parameters specific to the database processed
#
# """
# Price&Myers V1
# """
# DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/", "DBname" : "210413_PM_CO2_data",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 }
#
# xQualLabels = ['Sector','Type','Basement', 'Foundations','Ground Floor','Superstructure','Cladding', 'BREEAM Rating']
# xQuantLabels = ['GIFA (m2)','Storeys','Typical Span (m)', 'Typ Qk (kN_per_m2)']
# RemoveOutliersFrom = xQuantLabels
# yLabels = ['Calculated tCO2e_per_m2']
#
# FORMAT_Values = {'yUnitFactor': 1000, 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 800]}
#
# """
# ________________________________________________________________________________________________________________________
# PROCESSING
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for database processing
#
# PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : 0.5, 'train_size': 0.8,
#                 'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'corrHighThreshhold' : 0.65,
#                      'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]} #'scaler': 'MinMaxScaler', , 'lowThreshold' : 0.15, 'highThreshold' : 1,'yScale' : False,
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
# """
# ________________________________________________________________________________________________________________________
# MODEL
# ________________________________________________________________________________________________________________________
# """
#
# GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
#             'regul_range' : list(10.0 ** np.arange(-2, 2)),
#             'influence_range' : list(10.0 ** np.arange(-2, 2)),
#             'degree' : [2, 3],
#             'margin_range' : list(10.0 ** np.arange(-2, 2)),
#             'kernel_list' : [ 'linear', 'rbf']}
#
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
# V1 - kernels in params
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
