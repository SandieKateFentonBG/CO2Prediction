import numpy as np


"""
________________________________________________________________________________________________________________________
RUN
________________________________________________________________________________________________________________________
"""
#change when running a test

displayParams = {"reference" : None, 'showPlot': False, 'archive': True, 'showCorr' : False, 'plot_all': False, "ref_prefix" : None} #'CSTB_study_EC'

studyParams = {"sets": [[['Embodied_Carbon[kgCO2e_m2]'],'EC','TestR2']], 'randomvalues': list(range(40, 50)),
               "Regressors": ['KRR_RBF', 'KRR_POL','SVR_RBF']}

"""
________________________________________________________________________________________________________________________
DATABASE
________________________________________________________________________________________________________________________
"""
#parameters specific to the database processed

"""
EUCB-FR
"""
MyPred_Sample = {"DBpath" : "K:/Temp/Sandie/Pycharm/",  #C:/Users/sfenton/Code/Repositories/CO2Prediction/
             "DBname" : "Test-Wood",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : 'FRAME_study'} #Test-Concrete


DB_Values = {"DBpath" : "K:/Temp/Sandie/Pycharm/", #C:/Users/sfenton/Code/Repositories/CO2Prediction/
             "DBname" : "EU-ECB_dataset_feature_engineered_sf_framestr",
             "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : 'FRAME_study'} #"EU-ECB_dataset_feature_engineered_fr_residential_skf_cleaned-with_floors_ag",CSTB_res_nf_SEC_BleR2

xQualLabels = [
'Use_Type','Use_Subtype', 'Site_Country', 'Completion_Year', 'QTO_Type', 'bldg_area_interval', 'Structure', 'Roof',
    'Energy_Class', 'Main_Material', 'Second_Material',	'Lifespan',	'Life_Cycle_Scopes', 'LCA_Scope_handling_D']

xQuantLabels = ['Gross_Floor_Area'] #, 'Floors_Above_Ground', 'Floors_Below_Ground', 'Users_Total'

RemoveOutliersFrom = ['Gross_Floor_Area'] #'Floors_Above_Ground', 'Users_Total'

yLabels = ['Embodied_Carbon[kgCO2e_m2]'] #, 'Embodied_Carbon_Structure[kgCO2e_m2]'
# yLabels = ['Embodied_Carbon[kgCO2e_m2]'] #, 'Embodied_Carbon_Structure[kgCO2e_m2]'

FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}

#'yUnitFactor' converts from yLabel unit to target Label unit:
# ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000


"""
________________________________________________________________________________________________________________________
PROCESSING
________________________________________________________________________________________________________________________
"""
#parameters chosen for database processing

PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 42, 'test_size' : float(1/8), 'train_size': float(7/8), 'check_size': 0.1, 'val_size': float(1/9),
                'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1, 'fixed_seed' : 42,
                     'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}


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
FEATURE SELECTION
________________________________________________________________________________________________________________________
"""

BLE_VALUES = {'NBestScore': 'TestR2', 'NCount' : 10, 'Regressor' : 'LR_RIDGE', 'OverallBest' : True,
              'BestModelNames' : ['SVR_RBF_RFE_GBR', 'SVR_RBF_RFE_RFR', 'SVR_RBF_NoSelector', 'KRR_POL_RFE_GBR',
                                  'KRR_RBF_RFE_GBR', 'KRR_RBF_NoSelector', 'KRR_RBF_RFE_RFR', 'KRR_POL_RFE_RFR',
                                  'KRR_POL_NoSelector', 'KRR_POL_RFE_DTR']} #'TestAcc'SVR_RBF



