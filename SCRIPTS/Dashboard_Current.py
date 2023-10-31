import numpy as np

"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""

# NAMING

acronym = 'rmt_PM_V3_A123-C34_TEST'

# REFIT & SELECTION

base_refit = 'neg_mean_squared_error'
base_select = 'TestMSE'
ble_refit = 'neg_mean_squared_error'
ble_select = ['TestMSEs', True] #smallerisbetter

# base_refit = 'neg_mean_squared_error'
# base_select = 'TestAcc'
# ble_refit = 'neg_mean_squared_error'
# ble_select = ['TestAccs', False]

# base_refit = 'r2'
# base_select = 'TestR2'
# ble_refit = 'r2'
# ble_select = ['TestR2s', False]

# UNITS, SELECTORS, REGRESSORS

set_1 = [['A123-C34_Rate_kgCO2e-m2'],'EC',base_select] # ylabel, content, metric

studyParams = {"sets": [set_1], 'fl_selectors': ['spearman'],
               'RFE_selectors': ['GBR'],
               "Regressors": ['LR','SVR_RBF']}  # 'MLP_SGD''MLP_LBFG_20', 'MLP_LBFG_10', 'MLP_SGD_10', 'MLP_LBFG_100', 'MLP_SGD_100', 'MLP_SGD','MLP_LBFG', ['MLP_LBFG', 'MLP_SGD']['MLP_SGD','MLP_LBFG']['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'] #MLP_SGD #'MLP_LBFG'
# 'LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN','SVR_RBF'


# CROSS VALIDATION

sample_nb = 2
cv=5

# DISPLAY

displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'report_all': False, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None}


"""
________________________________________________________________________________________________________________________
DATABASE #parameters specific to the database processed
________________________________________________________________________________________________________________________
"""

"""
EUCB-FR
"""

# '//fs-par-001/commun/'


DB_Values = {"DBpath" : "//fs-par-001/commun/Temp/Sandie/Pycharm/",
             "DBname" : acronym,# + "-edit" > changed database content : users total "No Data" set to 100000 > make sure they are removed
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym}

xQualLabels = ['Stage', 'Sector', 'Construction', 'Foundation', 'Ground_Floor', 'Superstructure', 'Cladding',
               'Basement','Fire_Rating', 'Passive', 'Calculation_Year','Value_Type']#'Location','SCORS_Rating'

xQuantLabels = ['Floor_Area',  'Value', 'Storeys']

yLabels = set_1[0]
# Calculation_Year	Stage	Location	Location_Precise	Floor_Area	Value	Value_Type	Sector	Building_site
# Storeys	Passive	Basement	Foundation	Ground_Floor	Superstructure	Cladding	Fire_Rating

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1000],
                 'ResidLim' : [-300, 300], 'PredLim' : [0, 900]}#'yUnitFactor' converts yLabel unit to target Label unit: ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000

"""
SAMPLE
"""

MyPred_Sample_SELECTION = {"DBpath" : "//fs-par-001/commun/Temp/Sandie/Pycharm/",
             "DBname" : acronym + "-S-Sel",
                           "DBdelimiter": ';', "DBfirstLine": 5, 'acronym': acronym, 'Cols': 'Cladding',
                           'Rows': 'Superstructure',
                           'col_values': None, 'row_values': None, 'orderFtCols': None,
                           "orderFtRows": ['Concrete_Precast',
                                           'Concrete_PT', 'Timber_Frame_Glulam-CLT', 'Timber_Frame_Softwood',
                                           'Steel_Frame-Precast',
                                           'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other']}

MyPred_Sample_CONCRETE = {"DBpath" : "//fs-par-001/commun/Temp/Sandie/Pycharm/",
                          "DBname": acronym + "-S-CONCRETE",
                          "DBdelimiter": ';', "DBfirstLine": 5, 'acronym': acronym, 'Cols': 'Cladding',
                          'Rows': 'Superstructure',
                          'col_values': None, 'row_values': None, 'orderFtCols': None,
                          "orderFtRows": ['Concrete_Precast',
                                          'Concrete_In-Situ', 'Concrete_PT', 'Timber_Frame_Glulam-CLT',
                                          'Timber_Frame_Softwood', 'Steel_Frame-Precast',
                                          'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other',
                                          'Masonry-Concrete', 'Masonry-Timber', 'Other']}

MyPred_Sample_TIMBER = {"DBpath" : "//fs-par-001/commun/Temp/Sandie/Pycharm/",
                        "DBname": acronym + "-S-TIMBER",
                        "DBdelimiter": ';', "DBfirstLine": 5, 'acronym': acronym, 'Cols': 'Cladding',
                        'Rows': 'Superstructure',
                        'col_values': None, 'row_values': None, 'orderFtCols': None, "orderFtRows": ['Concrete_Precast',
                                                                                                     'Concrete_In-Situ',
                                                                                                     'Concrete_PT',
                                                                                                     'Timber_Frame_Glulam-CLT',
                                                                                                     'Timber_Frame_Softwood',
                                                                                                     'Steel_Frame-Precast',
                                                                                                     'Steel_Frame-Composite',
                                                                                                     'Steel_Frame-Timber',
                                                                                                     'Steel_Frame-Other',
                                                                                                     'Masonry-Concrete',
                                                                                                     'Masonry-Timber',
                                                                                                     'Other']}
MyPred_Sample_GLT = {"DBpath" : "//fs-par-001/commun/Temp/Sandie/Pycharm/",
                     "DBname": acronym + "-S-GLT",
                     "DBdelimiter": ';', "DBfirstLine": 5, 'acronym': acronym, 'Cols': 'Cladding',
                     'Rows': 'Superstructure',
                     'col_values': None, 'row_values': None, 'orderFtCols': None, "orderFtRows": ['Concrete_Precast',
                                                                                                  'Concrete_In-Situ',
                                                                                                  'Concrete_PT',
                                                                                                  'Timber_Frame_Glulam-CLT',
                                                                                                  'Timber_Frame_Softwood',
                                                                                                  'Steel_Frame-Precast',
                                                                                                  'Steel_Frame-Composite',
                                                                                                  'Steel_Frame-Timber',
                                                                                                  'Steel_Frame-Other',
                                                                                                  'Masonry-Concrete',
                                                                                                  'Masonry-Timber',
                                                                                                  'Other']}

"""
DATA ANALYSIS
"""
DAyLabels = ['Carbon_A1-A3_kgCO2e',	'Sequestration_A1-A3_kgCO2e','Carbon_Total_A1-A5_kgCO2e',	'Carbon_C3-C4_kgCO2e',
	'Carbon_D_kgCO2e',	'Carbon_Total_A1-D_kgCO2e',	'A1-A3_Rate_kgCO2e-m2',	'A1-A5_Rate_kgCO2e-m2', 'A1-D_Rate_kgCO2e-m2',
             'A123-C34_Rate_kgCO2e-m2']

# ['Carbon_A1-A3_kgCO2e',	'Sequestration_A1-A3_kgCO2e',	'Carbon_A4_kgCO2e',	'Carbon_A5a_kgCO2e',	'Carbon_A5w_kgCO2e',
# 'Carbon_Total_A1-A5_kgCO2e',	'Carbon_B1_kgCO2e',	'Carbon_C1_kgCO2e',	'Carbon_C2_kgCO2e',	'Carbon_C3-C4_kgCO2e',
# 'Carbon_Total_A1-C4_kgCO2e',	'Carbon_D_kgCO2e',	'Carbon_Total_A1-D_kgCO2e',	'A1-A3_Rate_kgCO2e-m2',	'A1-A5_Rate_kgCO2e-m2',
# 'A1-C4_Rate_kgCO2e-m2',	'A1-D_Rate_kgCO2e-m2',	'A123-C34_Rate_kgCO2e-m2',	'SCORS_Rating']

DAxQuantLabels = xQuantLabels
DARemoveOutliersFrom = xQuantLabels + DAyLabels

#CHANGES   !! LABELS MUST BE IN INITIAL IMPORT!
Summed_Labels = {'Carbon_A123-C34_kgCO2e' : ['Carbon_A1-A3_kgCO2e', 'Carbon_C3-C4_kgCO2e']}
Divided_Labels = {'A123-C34_Rate_kgCO2e-m2' : ['Carbon_A123-C34_kgCO2e', 'Floor_Area']} #SUMMED LABELS MUST BE IN INITIAL IMPORT!
splittingFt = 'Superstructure'
order = ['Concrete_In-Situ', 'Concrete_Precast','Concrete_PT','Timber_Frame_Glulam-CLT',
         'Timber_Frame_Softwood','Steel_Frame-Precast', 'Steel_Frame-Composite','Steel_Frame-Timber',
         'Steel_Frame-Other', 'Masonry-Concrete','Masonry-Timber','Other']
mainTarget = 'A123-C34_Rate_kgCO2e-m2'
labels_1D = ['Carbon_A1-A3_kgCO2e', 'A1-A3_Rate_kgCO2e-m2', 'Carbon_A123-C34_kgCO2e', 'A123-C34_Rate_kgCO2e-m2']

labels_2D_norm = [['Carbon_A1-A3_kgCO2e', 'A1-A3_Rate_kgCO2e-m2', 'Carbon_A1-A3_kgCO2e_normalize', 'A1-A3_Rate_kgCO2e-m2_normalize'],
                    ['Carbon_Total_A1-A5_kgCO2e', 'A1-A5_Rate_kgCO2e-m2', 'Carbon_Total_A1-A5_kgCO2e_normalize', 'A1-A5_Rate_kgCO2e-m2_normalize'],
                  ['Carbon_A123-C34_kgCO2e', 'A123-C34_Rate_kgCO2e-m2', 'Carbon_A123-C34_kgCO2e_normalize', 'A123-C34_Rate_kgCO2e-m2_normalize']]

labels_2D_scale = [['Carbon_A1-A3_kgCO2e', 'A1-A3_Rate_kgCO2e-m2', 'Carbon_A1-A3_kgCO2e_scale', 'A1-A3_Rate_kgCO2e-m2_scale'],
                    ['Carbon_Total_A1-A5_kgCO2e', 'A1-A5_Rate_kgCO2e-m2', 'Carbon_Total_A1-A5_kgCO2e_scale', 'A1-A5_Rate_kgCO2e-m2_scale'],
                  ['Carbon_A123-C34_kgCO2e', 'A123-C34_Rate_kgCO2e-m2', 'Carbon_A123-C34_kgCO2e_scale', 'A123-C34_Rate_kgCO2e-m2_scale']]


exploded_ft = 'Calculation_Year' #qual feature with few different values
splittingFt_focus = 'Concrete_In-Situ' #order[0]
focus = 'Concrete_In-Situ'
splittingFt_2 = 'Cladding'

#
# ## VARIANT
#
# # labels_2D_norm = []
# # labels_2D_scale = []
# # exploded_ft ='Stage'
# # focus = 'Timber_Frame_Glulam-CLT'  #!! no '/' in your name !!'TimberFrameGlulamCLT'
# # mainTarget = 'Storeys'
# # labels_1D = ['Storeys', 'Storeys_scale', 'Storeys_normalize']
# # splittingFt_focus = 'Timber_Frame_Glulam-CLT'
# # splittingFt_2 = 'Foundation'

"""
________________________________________________________________________________________________________________________
PROCESSING  #parameters chosen for database processing
________________________________________________________________________________________________________________________
"""

PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'UnderrepresentedCutOffThreshhold' : 5,
                  'removeUnderrepresenteds' : True, 'removeUnderrepresentedsDict' : dict(),
                'RemoveOutliersFrom' : xQuantLabels, 'removeUnderrepresentedsFrom' : xQualLabels,
                  'random_state' : sample_nb, 'test_size' : float(1/8), 'train_size': float(7/8), 'check_size': 0.1, 'val_size': float(1/9),
                'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 42, 'selectionStoredinCombined' : True,
                     'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75,
                  'accuracyTol' : 0.15, 'accuracyTol_mean' : 0.15, 'accuracyTol_std' : 0.5,
                  'residualsYLim': [-500, 500], 'residualsXLim': [0, 800],
                  'refit' : base_refit, 'grid_select' : base_select}

"""
________________________________________________________________________________________________________________________
LEARNING #parameters chosen for learning
________________________________________________________________________________________________________________________
"""

"""
________________________________________________________________________________________________________________________
FEATURE SELECTION
________________________________________________________________________________________________________________________
"""

RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : 'list(np.arange(10, len(baseFormatedDf.XTrain)-10, 10))',
              'RFE_process' : 'short', 'output_feature_count':'rfeCV'}

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


LR_param_grid={'alpha': GS_VALUES['regul_range']}
KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
                                                'coef0' : GS_VALUES['coef0_range']}
SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
                'epsilon': GS_VALUES['margin_range'],  'coef0' : GS_VALUES['coef0_range']}

MLP_LBFG_param_grid={ 'hidden_layer_sizes': [(20,), (10,), (100,)], 'activation' : ['relu'],
                 'alpha': list(10.0 ** -np.arange(1, 7))} #'solver': ['lbfgs'],
MLP_SGD_param_grid={ 'hidden_layer_sizes': [(20,), (10,), (100,)], 'activation' : ['relu'],
                 'alpha': list(10.0 ** -np.arange(1, 7))} #should be further hypertuned  'solver': ['sgd'],
MLP_ADAM_param_grid={ 'hidden_layer_sizes': [(20,), (10,), (100,)], 'activation' : ['relu'],
                 'alpha': list(10.0 ** -np.arange(1, 7))} #'solver': ['sag'],

MLP_LBFG_20_param_grid={'hidden_layer_sizes': [(20,)], 'activation' : ['relu'],'alpha': list(10.0 ** -np.arange(1, 7))} #, 'solver': ['lbfgs']
MLP_SGD_20_param_grid={'hidden_layer_sizes': [(20,)], 'activation' : ['relu'], 'alpha': list(10.0 ** -np.arange(1, 7))} #'solver': ['sgd'],
MLP_LBFG_10_param_grid={'hidden_layer_sizes': [(10,)], 'activation' : ['relu'],'alpha': list(10.0 ** -np.arange(1, 7))} #, 'solver': ['lbfgs']
MLP_SGD_10_param_grid={'hidden_layer_sizes': [(10,)], 'activation' : ['relu'],'alpha': list(10.0 ** -np.arange(1, 7))} #should be further hypertuned
MLP_LBFG_100_param_grid={'hidden_layer_sizes': [(100,)], 'activation' : ['relu'],'alpha': list(10.0 ** -np.arange(1, 7))} #, 'solver': ['lbfgs']
MLP_SGD_100_param_grid={'hidden_layer_sizes': [(100,)], 'activation' : ['relu'], 'alpha': list(10.0 ** -np.arange(1, 7))} #'solver': ['sgd'],

# hidden_layer_sizes : array-like of shape(n_layers - 2,), default=(100,)
# activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default=’relu’
# solver :{‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
# alpha : float, default=0.0001 # L2 regul_range
# batch_size : int, default=’auto’ => batch_size=min(200, n_samples) # Size of minibatches for stochastic optimizers
# learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default=’constant’ #Only used when solver=’sgd’.
# learning_rate_init : float, default=0.001 #Only used when solver=’sgd’ or ‘adam’.
# power_t : float, default=0.5 #exponent for ‘invscaling’, only used when solver=’sgd’.
# max_iter : int, default=200 # for stochastic solvers (‘sgd’, ‘adam’), determines number of epochs (times each point is used)
# shuffle : bool, default=True #Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.
# random_state : int, RandomState instance, default=None #random number generation for weights and bias initialization, train-test split if early stopping is used, and batch sampling when solver=’sgd’ or ‘adam’.
# tol : float, default=1e-4 #Tolerance for the optimization.
# verbose : bool, default=False # print progress messages to stdout.

"""
________________________________________________________________________________________________________________________
HYPERPARAM
________________________________________________________________________________________________________________________
"""

# # Example for Single Hyperparameter plot
KRR_param_grid1={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['linear']}
KRR_param_grid2={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['polynomial']}
KRR_param_grid3={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['rbf']}

"""
________________________________________________________________________________________________________________________
BLENDER
________________________________________________________________________________________________________________________
"""

BLE_VALUES = {'NBestScore': [set_1], 'NCount' : 10, 'Regressor' : ['SVR_RBF', 'LR_RIDGE'], 'OverallBest' : True,
              'BestModelNames' : None, 'refit' : ble_refit, 'grid_select' : ble_select}





