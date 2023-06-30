import numpy as np

"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test
sample_nb = 2
#READY TO RUN

displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'report_all': False, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None}
cv=5

set_1 = [['Embodied_Carbon[kgCO2e_m2]'],'EC','TestR2'] # ylabel, content, metric
acronym = 'STRUCT_study'
studyParams = {"sets": [set_1], 'fl_selectors': ['spearman', 'pearson'],  #, 'randomvalues': sample_values
               'RFE_selectors': ['GBR', 'DTR', 'RFR'],
               "Regressors": ['KRR_RBF']} #, 'KRR_POL','SVR_RBF'
"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
EUCB-FR
"""

# + "-edit" > changed users total "No Data" to 100000 > make sure they are removed

DB_Values = {"DBpath" : "K:/Temp/Sandie/Pycharm/", #C:/Users/sfenton/Code/Repositories/CO2Prediction/
             "DBname" : "EU-ECB_dataset_feature_engineered_sf_mass_fram_str" + "-edit",#EU-ECB_dataset_feature_engineered_sf_framestr
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym} #"EU-ECB_dataset_feature_engineered_fr_residential_skf_cleaned-with_floors_ag",CSTB_res_nf_SEC_BleR2

xQualLabels = [
'Use_Type','Use_Subtype', 'Site_Country', 'Completion_Year', 'QTO_Type', 'bldg_area_interval', 'Structure', 'Roof',
    'Energy_Class_Country', 'Energy_Class_General', 'Main_Material', 'Second_Material',	'Lifespan',	'Life_Cycle_Scopes', 'LCA_Scope_handling_D']

xQuantLabels = ['Gross_Floor_Area'] #, 'Floors_Above_Ground', 'Floors_Below_Ground', 'Users_Total'

# RemoveOutliersFrom = ['Gross_Floor_Area'] #'Floors_Above_Ground', 'Users_Total'

yLabels = ['Embodied_Carbon[kgCO2e_m2]'] #, 'Embodied_Carbon_Structure[kgCO2e_m2]'

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}

#'yUnitFactor' converts from yLabel unit to target Label unit:
# ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000

"""
SAMPLE
"""
# MyPred_Sample = {"DBpath" : "K:/Temp/Sandie/Pycharm/",  #C:/Users/sfenton/Code/Repositories/CO2Prediction/
#              "DBname" : "Test-Wood",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym} #Test-Concrete



MyPred_Sample_CONCRETEselection = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "EU-ECB-S-Concrete", #todo
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Main_Material', 'Rows':'Structure',
            'col_values' : None, 'row_values' : None,
                          'orderFtCols' : None, #[]todo 'Stone (granite, limestone, etc)','Ceramics (e.g., fired clay bricks)','Timber, wood','Concrete w/o reinforcement','Other'
                          "orderFtRows" : None} #[]todo select relevant structure types : 'massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood'


MyPred_Sample_CONCRETE = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "EU-ECB-S-Concrete", #todo
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Main_Material', 'Rows':'Structure',
            'col_values' : None, 'row_values' : None,
                          'orderFtCols' : None, #[]todo
                          "orderFtRows" : None} #[]todo

MyPred_Sample_TIMBER = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "EU-ECB-S-Wood", #todo
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Gross_Floor_Area', 'Rows':'Structure',
            'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None,
                     "orderFtRows" : None} #[]todo 'massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood'

MyPred_Sample_GLT = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "EU-ECB-S-GLT", #todo
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Gross_Floor_Area', 'Rows':'Structure',
            'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None,
                     "orderFtRows" : None} #todo['massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood']


"""
DATA ANALYSIS
"""
DAyLabels = ['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]'] #, 'Embodied_Carbon_Structure[kgCO2e_m2]'
DAxQuantLabels = xQuantLabels + ['Users_Total']#

DARemoveOutliersFrom = ['Gross_Floor_Area', 'Users_Total'] + DAyLabels

#CHANGES   !! LABELS MUST BE IN INITIAL IMPORT!
Summed_Labels = dict() #SUMMED{'test' : ['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]']}
Divided_Labels = {'GIFA_user[m2_u]' : ['Gross_Floor_Area', 'Users_Total'], 'Embodied_Carbon_user[kgCO2e_u]' : ['Embodied_Carbon[kgCO2e]', 'Users_Total'],
                  'Embodied_Carbon_user_m2[kgCO2e_u_m2]' : ['Embodied_Carbon[kgCO2e_m2]', 'Users_Total']} #SUMMED LABELS MUST BE IN INITIAL IMPORT!
# AddedLabels = [k for k in Summed_Labels.keys()] + [k for k in Divided_Labels.keys()]
splittingFt = 'Structure'
order = ['massive brick','massive concrete','frame concrete','massive wood','frame wood','frame concrete/wood']
mainTarget = 'Embodied_Carbon[kgCO2e_m2]'
labels_1D = ['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]', 'Embodied_Carbon_user[kgCO2e_u]', 'Embodied_Carbon_user_m2[kgCO2e_u_m2]']

labels_2D_norm = [['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]', 'Embodied_Carbon[kgCO2e]_normalize', 'Embodied_Carbon[kgCO2e_m2]_normalize'],
['Embodied_Carbon_user[kgCO2e_u]', 'Embodied_Carbon_user_m2[kgCO2e_u_m2]', 'Embodied_Carbon_user[kgCO2e_u]_normalize', 'Embodied_Carbon_user_m2[kgCO2e_u_m2]_normalize']]

labels_2D_scale = [['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]', 'Embodied_Carbon[kgCO2e]_scale', 'Embodied_Carbon[kgCO2e_m2]_scale'],
['Embodied_Carbon_user[kgCO2e_u]', 'Embodied_Carbon_user_m2[kgCO2e_u_m2]', 'Embodied_Carbon_user[kgCO2e_u]_scale', 'Embodied_Carbon_user_m2[kgCO2e_u_m2]_scale']]

exploded_ft = 'Use_Subtype' #qual feature with few different values
splittingFt_focus = 'massive wood' #order[0]
focus = 'Structure-massive_wood'
splittingFt_2 = 'Main_Material'

"""
________________________________________________________________________________________________________________________
PROCESSING
________________________________________________________________________________________________________________________
"""
#parameters chosen for database processing


PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'UnderrepresentedCutOffThreshhold' : 5,
                  'removeUnderrepresenteds' : True, 'removeUnderrepresentedsDict' : dict(),
                'RemoveOutliersFrom' : ['Gross_Floor_Area'], 'removeUnderrepresentedsFrom' : xQualLabels,
                  'random_state' : sample_nb, 'test_size' : float(1/8), 'train_size': float(7/8), 'check_size': 0.1, 'val_size': float(1/9),
                'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 42, 'selectionStoredinCombined' : True,
                     'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'accuracyTol' : 0.15, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]} #'corrMethod1' : "spearman", 'corrMethod2' : "pearson",

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
MLP_SAG_param_grid={ 'hidden_layer_sizes': [(20,), (10,), (100,)], 'activation' : ['relu'],
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
FEATURE SELECTION
________________________________________________________________________________________________________________________
"""

BLE_VALUES = {'NBestScore': 'TestR2', 'NCount' : 10, 'Regressor' : 'SVR_RBF', 'OverallBest' : True,
              'BestModelNames' : None} #'TestAcc'LR_RIDGE


