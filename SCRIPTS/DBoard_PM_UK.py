import numpy as np

"""
________________________________________________________________________________________________________________________
RUN #change when running a test
________________________________________________________________________________________________________________________
"""

sample_nb = 2
displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'report_all': False, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None}
cv=5

set_1 = [['A123-C34_Rate_kgCO2e-m2'],'EC','TestR2'] # ylabel, content, metric #'A1-A5_Rate_kgCO2e-m2'
acronym = 'PM_A123-C34_study' #'PM_V3_A15'

studyParams = {"sets": [set_1], 'fl_selectors': ['spearman', 'pearson'],
               'RFE_selectors': ['GBR', 'DTR', 'RFR'],
               "Regressors": ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']} #'MLP_SGD''MLP_LBFG_20', 'MLP_LBFG_10', 'MLP_SGD_10', 'MLP_LBFG_100', 'MLP_SGD_100', 'MLP_SGD','MLP_LBFG', ['MLP_LBFG', 'MLP_SGD']['MLP_SGD','MLP_LBFG']['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'] #MLP_SGD #'MLP_LBFG'

# sets = [
#     ['Embodied_Carbon[kgCO2e_m2]','EC','TestR2'],
#     ['Embodied_Carbon[kgCO2e_m2]','EC','TestAcc'],
#     ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS', 'TestR2'],
#     ['Embodied_Carbon_Structure[kgCO2e_m2]','ECS','TestAcc']]

"""
________________________________________________________________________________________________________________________
DATABASE #parameters specific to the database processed
________________________________________________________________________________________________________________________
"""

DB_Values = {"DBpath" : "K:/Temp/Sandie/Pycharm/", #C:/Users/sfenton/Code/Repositories/CO2Prediction/
             "DBname" : "P&M_Carbon_Database_2023_sfedit",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym}

xQualLabels = ['Stage', 'Sector', 'Construction', 'Foundation', 'Ground_Floor', 'Superstructure', 'Cladding',
               'Basement','Fire_Rating', 'Passive', 'Calculation_Year','Value_Type']#'Location','SCORS_Rating'

xQuantLabels = ['Floor_Area',  'Value', 'Storeys']

yLabels = set_1[0]
# Calculation_Year	Stage	Location	Location_Precise	Floor_Area	Value	Value_Type	Sector	Building_site
# Storeys	Passive	Basement	Foundation	Ground_Floor	Superstructure	Cladding	Fire_Rating

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}#'yUnitFactor' converts yLabel unit to target Label unit: ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000

"""
SAMPLE
"""

# MyPred_Sample_SELECTION = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "EU-ECB-S-Sel",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Main_Material', 'Rows':'Structure',
#             'col_values' : None, 'row_values' : None,
#                           'orderFtCols' : ['Stone (granite, limestone, etc)','Ceramics (e.g., fired clay bricks)','Timber, wood','Concrete w/o reinforcement','Other'],
#                           "orderFtRows" : ['frame concrete','frame concrete/wood','frame wood']}
#
# MyPred_Sample_CONCRETE = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "EU-ECB-S-Concrete",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Main_Material', 'Rows':'Structure',
#             'col_values' : None, 'row_values' : None,
#                           'orderFtCols' : ['Stone (granite, limestone, etc)','Ceramics (e.g., fired clay bricks)','Timber, wood','Concrete w/o reinforcement','Other'],
#                           "orderFtRows" : ['massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood']}
#
# MyPred_Sample_TIMBER = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "EU-ECB-S-Wood",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Gross_Floor_Area', 'Rows':'Structure',
#             'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None,
#                      "orderFtRows" : ['massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood']}
#
# MyPred_Sample_GLT = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "EU-ECB-S-GLT",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Gross_Floor_Area', 'Rows':'Structure',
#             'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None,
#                      "orderFtRows" : ['massive brick','massive concrete','frame concrete','frame concrete/wood','massive wood','frame wood']}
#
#
#
#
# MyPred_Sample2 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "PM_V3_Test",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Type', 'Rows':'Superstructure',
#             'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
#                         'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
#                           'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
#                           'Steel Frame/Timber',
#                           'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}
#
# MyPred_Sample3 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "PM_V3_Test",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
#             'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
#                         'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
#                           'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
#                           'Steel Frame/Timber',
#                           'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}
#
# MyPred_Sample4 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
#              "DBname" : "PM_V3_Test",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Floor_Area', 'Rows':'Superstructure',
#             'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
#                         'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
#                           'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
#                           'Steel Frame/Timber',
#                           'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}


# ['Floor_Area', 'Value', 'Storeys',
#  'As-Built', 'Stage_4', 'Stage_3', 'Stage_5', 'Stage_2',
#  'Residential_Single', 'Healthcare', 'Residential_Multi', 'Cultural-Public', 'Educational', 'Commercial', 'Mixed_Use', 'Science-Lab', 'Industrial', 'Other',
#  'New_Build_Brownfield', 'Mixed_New_Build-Refurb', 'New_Build_Greenfield', 'Full_Refurb',
#  'Raft', 'Piled_Ground_Beams', 'Mass_Pads-Strips', 'Reinforced_Pads-Strips', 'Piles_Pile_Caps', 'Other', 'Reused', 'Corbels',
#  'Raft',  'Suspended_RC', 'Ground_Bearing_RC', 'Suspended_Precast', 'Reused', 'Other', 'Timber',
#  'Timber_Frame_Glulam-CLT', 'Concrete_In-Situ', 'Steel_Frame-Precast', 'Masonry-Concrete', 'Steel_Frame-Composite', 'Steel_Frame-Other', 'Masonry-Timber', 'Concrete_Precast', 'Other', 'Steel_Frame-Timber', 'Timber_Frame_Softwood', 'Concrete_PT',
#  'Cladding=Lightweight_Only', 'Cladding=Masonry-SFS', 'Cladding=Masonry-Stone', 'Cladding=SFS-Stone', 'Cladding=Glazed-Curtain_Wall', 'Cladding=Masonry_Only', 'Cladding=Precast-GRC', 'Cladding=Stone', 'Cladding=Other', 'Cladding=Lightweight-SFS', 'Cladding=Timber-SFS', 'Cladding=Timber_Only', 'Cladding=Retained_Full-Partial', 'Cladding=Masonry-Timber',
#  'None', 'Partial_Footprint', 'Full_Footprint', 'Extended_Footprint', 'Other',
#  '90', '60', '120', '30', '240', 'Other',
#  'No', 'Yes',
#  '2022', '2020', '2021',
#  'Whole_Building', 'Structure_Only',
#  'A123-C34_Rate_kgCO2e-m2']



MyPred_Sample_SELECTION = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_Test",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete_Precast',
            'Concrete_PT','Timber_Frame_Glulam-CLT','Timber_Frame_Softwood','Steel_Frame-Precast',
        'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other']}

MyPred_Sample_TIMBER = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_TIMBER",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete_Precast',
            'Concrete_In-Situ','Concrete_PT','Timber_Frame_Glulam-CLT','Timber_Frame_Softwood','Steel_Frame-Precast',
        'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other','Masonry-Concrete',  'Masonry-Timber',  'Other']}

MyPred_Sample_CONCRETE = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_CONCRETE",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete_Precast',
            'Concrete_In-Situ','Concrete_PT','Timber_Frame_Glulam-CLT','Timber_Frame_Softwood','Steel_Frame-Precast',
        'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other','Masonry-Concrete',  'Masonry-Timber',  'Other']}

MyPred_Sample_GLT = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_GLT",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete_Precast',
            'Concrete_In-Situ','Concrete_PT','Timber_Frame_Glulam-CLT','Timber_Frame_Softwood','Steel_Frame-Precast',
        'Steel_Frame-Composite', 'Steel_Frame-Timber', 'Steel_Frame-Other','Masonry-Concrete',  'Masonry-Timber',  'Other']}

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
                'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 40, 'selectionStoredinCombined' : True,
                     'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'accuracyTol' : 0.15, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]} #'corrMethod1' : "spearman", 'corrMethod2' : "pearson",


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
BLENDER
________________________________________________________________________________________________________________________
"""

BLE_VALUES = {'NBestScore': 'TestR2', 'NCount' : 10, 'Regressor' : 'LR_RIDGE', 'OverallBest' : True,
              'BestModelNames' : None} #'TestAcc'SVR_RBF



"""
steps :

------------------------------------------------------------------------------------------------------------------------
Clean data:

Filter the CSTB dataframe using df_id_operation_valid, so only valid entries remain
filtering away low quality cases as defined by CSTB
#Remove cases where "type_structure_principale"
Drop all rows which have a missing value
#Remove cases where "type_structure_principale" is 5 (invalid entry)
------------------------------------------------------------------------------------------------------------------------
Rename directly transferable data:

"nb_occupant" to "bldg_users_total"
"nb_niv_surface" to "bldg_floors_ag"
"nb_niv_ssol" to "bldg_floors_bg"
"sdp" to "bldg_area_hfa"
"niveau_energie" to "bldg_energy_class_country"
"cef" to "inv_energy_consumption"
"periode_etude_reference" to "lca_RSP"
"indicateur_1" to "GHG_sum_em"

------------------------------------------------------------------------------------------------------------------------
"indicateur_1" : Potentiel de réchauffement climatique [GWP)[kg éq. CO2]
remove operational energy and water emissions from total to get sum embodied:
df_CSTB["GHG_sum_em"] = df_CSTB["GHG_sum_em"]-df_CSTB["IND1_ENE"]-df_CSTB["IND1_EAU"]-df_CSTB["IND1_LOT1"]-df_CSTB["IND1_LOT14"]

------------------------------------------------------------------------------------------------------------------------
Translate and regroup/rename data

"usage_principal" to "bldg_use_subtype"

"type_toiture" to "bldg_roof_type"
'3 pans et plus' to'Other'
'2 pans','Gable or saddle roof'
'Terrasse','Flat roof'
'Monopente','Single pitched roof'

"type_travaux" to "bldg_project_status"
'Bâtiments neufs','New Built'
'Extensions ou surélévations','Renovation'

"date_etude_rsenv" to "bldg_year_complete_interval"

------------------------------------------------------------------------------------------------------------------------
Transform and derive data through inference

"type_structure_principale" and "materiau_principal" to "bldg_struct_type"

'Maçonnerie Terre cuite','massive brick'
'Voiles porteurs Béton','Voiles porteurs Béton haute performance'],'massive concrete'
'Maçonnerie Béton','massive brick'
'Maçonnerie Autre, à préciser','massive brick'
'Ossature Bois massif','frame wood'
'Poteaux/poutres Béton','Poteaux/poutres Béton haute performance'],'frame concrete'
'Maçonnerie Béton haute performance','massive brick'
'Poteaux/poutres Autre, à préciser','other'
'Poteaux/poutres Mixte: bois-béton','mix concrete wood'
'Ossature Bois massif reconstitué','frame wood'
'Ossature Mixte: bois-béton','frame concrete/wood'
'Poteaux/poutres Mixte: béton-acier','frame concrete/steel'
'Poteaux/poutres Bois massif','Poteaux/poutres Bois massif reconstitué'],'frame wood'
'Maçonnerie Bois massif reconstitué','Maçonnerie Mixte: bois-béton','Maçonnerie Bois massif'],'massive brick'
'Ossature Acier','frame steel'
'Voiles porteurs Béton cellulaire','massive concrete'
'Ossature Béton','frame concrete'
'Maçonnerie Pierre','massive brick'
'Voiles porteurs Terre cuite','massive brick'
'Maçonnerie Béton cellulaire','massive brick'
'Maçonnerie Bois massif','massive brick'
'Ossature Terre cuite','massive brick'
'Voiles porteurs Bois massif reconstitué','Voiles porteurs Bois massif'],'massive wood'

"materiau_principal" to "inv_mat_1_type"

'Terre cuite','Ceramics (e.g., fired clay bricks)'
'Béton','Concrete w/o reinforcement'
'Pierre','Stone (granite, limestone, etc)'
'Autre, à préciser','Other'
'Bois massif','Timber, wood'
'Béton haute performance','Concrete w/o reinforcement'
'Mixte: bois-béton','Other'
'Bois massif reconstitué','Timber, wood'
'Mixte: béton-acier','Other'
'Acier','Metals (iron, steel)'
'Béton cellulaire','Other'



#Infer bldg_use_type (Building type)
#Copy "bldg_use_subtype"
df_CSTB['bldg_use_type'] = df_CSTB['bldg_use_subtype']

#Replace entries with infered type

'bldg_use_subtype' :
'Residential'
'Single family house' : 341
'Multi-family house' : 114

'Non-residential' :
'Office': 15
'School and Daycare': 14
'Hospital and Health' : 2

df_CSTB['scope_handling_D'] = 'separately considered'

#Infer GHG_sum_m2a (Sum)
df_CSTB['GHG_sum_em_m2a'] = df_CSTB['GHG_sum_em']/df_CSTB['bldg_area_gfa']/df_CSTB['lca_RSP']

#Infer GHG_sum_em_m2a (Sum Operational)
df_CSTB['GHG_sum_op_m2a'] = df_CSTB['co2']

------------------------------------------------------------------------------------------------------------------------
"Lots" decoding work
#Use the "lots" to calculate the impact of individual building parts

# "indicateur_1" to "GHG_sum_em"

# "GHG_P1_sum_m2a" > Ground = Lot2: Foundations and infrastructure
# 'GHG_P2_sum_m2' > Structure = Lot3: Structure and masonry + Lot4: Roof and cover
# "GHG_P34_sum_m2a" > Envelope = Lot6: Exterior surfaces (facades), components (doors and windows) and joineries
# "GHG_P4_sum_m2a" > Internal = Lot5: Interior partitions, suspended ceilings, components and joineries + Lot7: Interior coatings (floors, walls and ceilings)
# "GHG_P56_sum_m2a" > Services =  Lot8: HVAC equipment + Lot10: Electrical equipment + Lot11: Special electrical equipment (systems, controls and communication) + Lot13: Local
# "GHG_P78_sum_m2a" > Appliances = Lot9: Sanitary installations


# a = 50 years > 'GHG_P2_sum_m2' = 50 * 'GHG_P2_sum_m2a'
# https://github.com/mroeck/Embodied-Carbon-of-European-Buildings-Database/blob/develop/01_Preproc_CSTB.ipynb

------------------------------------------------------------------------------------------------------------------------
Add empty data columns
Rearrange columns (drop columns)

Stages included in target
We need to build a string based on:
- Ground (1) (i.e. substructure, foundation, basement walls, etc.)
- Load-bearing structure (2) (i.e. structural frame, walls, floors, roofs, etc.)
- Envelope (3, 4) (i.e. openings, ext. finishes, etc.)
- Internal (4) (i.e. partitions, int. finishes, etc.)
- Services (5,6) (i.e. mechanical, electrical, renew. energy, etc.)
- Appliances (7,8) (i.e. fixed facilities, mobile fittings, etc.)


"""


