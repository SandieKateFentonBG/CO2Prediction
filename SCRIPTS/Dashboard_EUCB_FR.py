import numpy as np

"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test

displayParams = {"reference" : 'CSTB_large/', 'showPlot': False, 'archive': True, 'showCorr' : False}

"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
EUCB-FR
"""
DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/", "DBname" : "EU-ECB_dataset_feature_engineered_fr_sf",
             "DBdelimiter" : ';', "DBfirstLine" : 5 }
# Project Value (œm)
xQualLabels = ['admin_project_contact',
'bldg_use_type'	,
'bldg_use_subtype',
'bldg_project_status',
'site_country',
'bldg_QTO_type',
'bldg_area_interval',
'bldg_struct_type',
'bldg_roof_type',
'bldg_energy_class_country',
'bldg_certification',
'inv_mat_1_type',
'lca_RSP',
'lca_software',
'lca_database',
'scope_parts',
'scope_handling_D',
'bldg_year_complete_interval']
#incomplete : 'bldg_floors_ag','bldg_energy_class_general',



xQuantLabels = ['bldg_area_gfa', 'bldg_floors_bg'] #,
#incomplete : 'bldg_floors_ag','inv_energy_consumption','bldg_users_total',
RemoveOutliersFrom = ['bldg_area_gfa']
yLabels = ['GHG_sum_em_m2']

#yLabels
#'GHG_sum_em_m2' (max - 1799,718511)
#'GHG_A123_m2_harm_LCM' (max - 1226,516419)
#'GHG_P1_sum_m2a'
FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}

#'yUnitFactor' converts from yLabel unit to target Label unit:
# ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000

#todo : what to do with empty data? deletz line? replace by?
# todo : remove outliers for number floors if count = 0 removes everything - what to do ?

"""
________________________________________________________________________________________________________________________
PROCESSING
________________________________________________________________________________________________________________________
"""
#parameters chosen for database processing

PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : 0.5, 'train_size': 0.8,
                'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1,
                     'corrHighThreshhold' : 0.65, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}
#todo : check 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]


"""
________________________________________________________________________________________________________________________
GRIDSEARCH
________________________________________________________________________________________________________________________
"""
#parameters chosen for gridsearch opimization

"""
________________________________________________________________________________________________________________________
FEATURE SELECTION
________________________________________________________________________________________________________________________
"""

RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : 'list(np.arange(10, len(baseFormatedDf.XTrain)-10, 10))'} #[5, 10, 15, 20, 25]



"""
________________________________________________________________________________________________________________________
MODEL
________________________________________________________________________________________________________________________
"""

# GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
#             'regul_range' : list(10.0 ** np.arange(-2, 2)),
#             'influence_range' : list(10.0 ** np.arange(-2, 2)),
#             'degree' : [2, 3],
#             'margin_range' : list(10.0 ** np.arange(-2, 2)),
#             'kernel_list' : [ 'linear', 'rbf']}

GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
            'regul_range' : list(10.0 ** np.arange(-4, 4)),
            'influence_range' : list(10.0 ** np.arange(-4, 4)),
            'degree' : [2, 3, 4],
            'margin_range' : list(10.0 ** np.arange(-4, 4)),
            'kernel_list' : ['poly', 'linear', 'rbf']}

"""
________________________________________________________________________________________________________________________
V1 - kernels  in params
________________________________________________________________________________________________________________________
"""

# LR_param_grid={'alpha': GS_VALUES['regul_range']}
# KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                                                 'kernel' : GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}
# SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                 'epsilon': GS_VALUES['margin_range'], 'kernel': GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}

"""
________________________________________________________________________________________________________________________
V2 - kernels not in params
________________________________________________________________________________________________________________________
"""

LR_param_grid={'alpha': GS_VALUES['regul_range']}
KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
                                                'coef0' : GS_VALUES['coef0_range']}
SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
                'epsilon': GS_VALUES['margin_range'],  'coef0' : GS_VALUES['coef0_range']}

"""
________________________________________________________________________________________________________________________
HYPERPARAM
________________________________________________________________________________________________________________________
"""

# # Example for Single Hyperparameter plot
KRR_param_grid1={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['linear']}
KRR_param_grid2={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['polynomial']}
KRR_param_grid3={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['rbf']}


