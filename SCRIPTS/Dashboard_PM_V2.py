import numpy as np




"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test
sample_nb = 42
sample_values = list(range(40, 50))
displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None}

set_1 = [['A1-A5 Rate (kgCO2e/m2)'],'EC','TestR2'] # ylabel, content, metric #'A123-C34 Rate (kgCO2e/m2)'
acronym = 'PM_V3_A15' #'PM_V3_A123-C34'
studyParams = {"sets": [set_1], 'randomvalues': sample_values,
               "Regressors": ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF']}

"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
Price&Myers V3
"""


DB_Values = {"DBpath" : "K:/Temp/Sandie/Pycharm/", "DBname" : "P&M_Carbon_Database_2023_sf",
             "DBdelimiter" : ';', "DBfirstLine" : 5, 'acronym' : acronym}

xQualLabels = ['Stage','Value_Type','Sector','Construction','Passive','Basement','Foundation','Ground_Floor',
               'Superstructure','Cladding']#'District'  V3 not in : 'Fire_Rating' 'Location
xQuantLabels = ['Floor_Area', 'Value', 'Storeys']  #V3 not in : 'Calculation_Year'


RemoveOutliersFrom = xQuantLabels
yLabels = set_1[0]

# ['A123-C34 Rate (kgCO2e/m2)', 'A1-A3 Rate (kgCO2e/m2)',	'A1-A5 Rate (kgCO2e/m2)',	'A1-C4 Rate (kgCO2e/m2)',	'A1-D Rate (kgCO2e/m2)',	'SCORS Rating']

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels' : ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}

#'yUnitFactor' converts from yLabel unit to target Label unit:
# ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000

"""
SAMPLE
"""


MyPred_Sample2 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_Test",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Type', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}

MyPred_Sample3 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_Test",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}

MyPred_Sample4 = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_Test",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Floor_Area', 'Rows':'Superstructure',
            'col_values' : list(range(100, 1000, 100)), 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}


MyPred_Sample_TIMBER = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_TIMBER",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}

MyPred_Sample_CONCRETE = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_CONCRETE",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}

MyPred_Sample_GLT = {"DBpath" : "K:/Temp/Sandie/Pycharm/",
             "DBname" : "PM_V3_GLT",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : acronym, 'Cols':'Cladding', 'Rows':'Superstructure',
            'col_values' : None, 'row_values' : None, 'orderFtCols' : None, "orderFtRows" : ['Concrete (In-Situ)',
                        'Concrete (Precast)', 'Concrete (PT)', 'Timber Frame (Glulam/CLT)',
                          'Timber Frame (Softwood)', 'Steel Frame/Precast', 'Steel Frame/Composite',
                          'Steel Frame/Timber',
                          'Steel Frame/Other', 'Masonry/Concrete', 'Masonry/Timber', 'Masonry & Timber', 'Other']}



"""
DATA ANALYSIS
"""

DAyLabels = ['Embodied_Carbon[kgCO2e]', 'Embodied_Carbon[kgCO2e_m2]', 'Embodied_Carbon_Structure[kgCO2e_m2]']
DARemoveOutliersFrom = ['Gross_Floor_Area', 'Users_Total'] + DAyLabels

#CHANGES   !! LABELS MUST BE IN INITIAL IMPORT!
Summed_Labels = {'Carbon A123-C34 (kgCO2e)' : ['Carbon A1-A3 (kgCO2e)', 'Carbon C3-C4 (kgCO2e)']}
Divided_Labels = {'A123-C34 Rate (kgCO2e/m2)' : ['Carbon A123-C34 (kgCO2e)', 'Floor_Area']} #SUMMED LABELS MUST BE IN INITIAL IMPORT!
splittingFt = 'Superstructure'
order = ['Concrete (In-Situ)', 'Concrete (Precast)','Concrete (PT)','Timber Frame (Glulam/CLT)',
         'Timber Frame (Softwood)','Steel Frame/Precast', 'Steel Frame/Composite','Steel Frame/Timber',
         'Steel Frame/Other', 'Masonry/Concrete','Masonry/Timber', 'Masonry & Timber','Other']
mainTarget = 'A123C34 Rate (kgCO2e/m2)'
labels_1D = ['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A123C34 (kgCO2e)', 'A123C34 Rate (kgCO2e/m2)']

labels_2D_norm = [['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A1-A3 (kgCO2e)_normalize', 'A1-A3 Rate (kgCO2e/m2)_normalize'],
                    ['Carbon Total A1-A5 (kgCO2e)', 'A1-A5 Rate (kgCO2e/m2)', 'Carbon Total A1-A5 (kgCO2e)_normalize', 'A1-A5 Rate (kgCO2e/m2)_normalize'],
                  ['Carbon A123-C34 (kgCO2e)', 'A123-C34 Rate (kgCO2e/m2)', 'Carbon A123-C34 (kgCO2e)_normalize', 'A123-C34 Rate (kgCO2e/m2)_normalize']]

labels_2D_scale = [['Carbon A1-A3 (kgCO2e)', 'A1-A3 Rate (kgCO2e/m2)', 'Carbon A1-A3 (kgCO2e)_scale', 'A1-A3 Rate (kgCO2e/m2)_scale'],
                    ['Carbon Total A1-A5 (kgCO2e)', 'A1-A5 Rate (kgCO2e/m2)', 'Carbon Total A1-A5 (kgCO2e)_scale', 'A1-A5 Rate (kgCO2e/m2)_scale'],
                  ['Carbon A123-C34 (kgCO2e)', 'A123-C34 Rate (kgCO2e/m2)', 'Carbon A123-C34 (kgCO2e)_scale', 'A123-C34 Rate (kgCO2e/m2)_scale']]


# Carbon A1-A3 (kgCO2e)	Sequestration A1-A3 (kgCO2e)	Carbon A4 (kgCO2e)	Carbon A5a (kgCO2e)	Carbon A5w (kgCO2e)
# Carbon Total A1-A5 (kgCO2e)	Carbon B1 (kgCO2e)	Carbon C1 (kgCO2e)	Carbon C2 (kgCO2e)	Carbon C3-C4 (kgCO2e)
# Carbon Total A1-C4 (kgCO2e)	Carbon D (kgCO2e)	Carbon Total A1-D (kgCO2e)	A1-A3 Rate (kgCO2e/m2)	A1-A5 Rate (kgCO2e/m2)
# A1-C4 Rate (kgCO2e/m2)	A1-D Rate (kgCO2e/m2)	A123-C34 Rate (kgCO2e/m2)	SCORS Rating


exploded_ft = 'Calculation_Year' #qual feature with few different values
splittingFt_focus = 'Concrete (In-Situ)' #order[0]
focus = 'Structure-massive_wood'
splittingFt_2 = 'Cladding'

#
# ## VARIANT
#
# # labels_2D_norm = []
# # labels_2D_scale = []
# # exploded_ft ='Stage'
# # focus = 'Timber Frame (Glulam/CLT)  #!! no '/' in your name !!'TimberFrameGlulamCLT'
# # mainTarget = 'Storeys'
# # labels_1D = ['Storeys', 'Storeys_scale', 'Storeys_normalize']
# # splittingFt_focus = 'Timber Frame (Glulam/CLT)'
# # splittingFt_2 = 'Foundation'

"""
________________________________________________________________________________________________________________________
PROCESSING
________________________________________________________________________________________________________________________
"""
#parameters chosen for database processing


PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : sample_nb, 'test_size' : float(1/8), 'train_size': float(7/8), 'check_size': 0.1, 'val_size': float(1/9),
                'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 40,
                     'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'accuracyTol' : 0.15, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}

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

BLE_VALUES = {'NBestScore': 'TestR2', 'NCount' : 10, 'Regressor' : 'LR_RIDGE', 'OverallBest' : True,
              'BestModelNames' : None} #'TestAcc'SVR_RBF


