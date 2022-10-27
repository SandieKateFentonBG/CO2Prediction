import numpy as np

"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test

displayParams = {"reference" : '221027_PMV2_/', 'showPlot': False, 'archive': True, 'showCorr' : True}

"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
Price&Myers V1
"""
DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/", "DBname" : "P&M Carbon Database for Release 11-02-2022",
             "DBdelimiter" : ';', "DBfirstLine" : 5 }
# Project Value (œm)
xQualLabels = ['Calculation Design Stage','Value Type','Project Value (œm)', 'Project Sector', 'Construction Type','Passivhaus?',
               'Basement Type', 'Foundation Type', 'Ground Floor Type', 'Superstructure Type', 'Cladding Type']
xQuantLabels = ['GIFA (m2)','Storeys (#)']
yLabels = ['A1-A5 Rate (kgCO2e/m2)']

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}

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

RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : [5, 10, 15, 20, 25, 30]}



"""
________________________________________________________________________________________________________________________
MODEL
________________________________________________________________________________________________________________________
"""



GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
            'regul_range' : list(10.0 ** np.arange(-4, 4)),
            'influence_range' : list(10.0 ** np.arange(-4, 4)),
            'degree' : [2, 3, 4],
            'margin_range' : list(10.0 ** np.arange(-4, 4)),
            'kernel_list' : ['poly', 'linear', 'rbf']}

"""
________________________________________________________________________________________________________________________
V1
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


