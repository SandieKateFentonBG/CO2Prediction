""""
# GOAL 
# RUN CV_BLender 
"""""

#DASHBOARD IMPORT
from Dashboard_EUCB_FR_v2 import *

#SCRIPT IMPORTS
from Main_BL_Steps import *


ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
# ref_suffix_single, ref_suffix_combined = '_rd' + str(PROCESS_VALUES['random_state']) + '/', '_Combined/'
# ref_single, ref_combined = ref_prefix + ref_suffix_single, ref_prefix + ref_suffix_combined
# displayParams["ref_prefix"], displayParams["reference"] = ref_prefix, ref_single

Run_Blending_CV(displayParams, DB_Values['DBpath'], ref_prefix, ConstructorKey = BLE_VALUES['Regressor'],
                GS_FS_List_Labels=['LR', 'LR_RIDGE', 'KRR_RBF', 'KRR_LIN', 'KRR_POL',
                                   'SVR_RBF', 'SVR_LIN'],
                GS_name_list=['LR_fl_spearman', 'LR_fl_pearson', 'LR_RFE_RFR', 'LR_RFE_DTR', 'LR_RFE_GBR',
                              'LR_NoSelector',
                              'LR_RIDGE_fl_spearman', 'LR_RIDGE_fl_pearson', 'LR_RIDGE_RFE_RFR', 'LR_RIDGE_RFE_DTR',
                              'LR_RIDGE_RFE_GBR', 'LR_RIDGE_NoSelector', 'KRR_LIN_fl_spearman', 'KRR_LIN_fl_pearson',
                              'KRR_LIN_RFE_RFR',
                              'KRR_LIN_RFE_DTR', 'KRR_LIN_RFE_GBR', 'KRR_LIN_NoSelector', 'KRR_RBF_fl_spearman',
                              'KRR_RBF_fl_pearson',
                              'KRR_RBF_RFE_RFR', 'KRR_RBF_RFE_DTR', 'KRR_RBF_RFE_GBR', 'KRR_RBF_NoSelector',
                              'KRR_POL_fl_spearman',
                              'KRR_POL_fl_pearson', 'KRR_POL_RFE_RFR', 'KRR_POL_RFE_DTR', 'KRR_POL_RFE_GBR',
                              'KRR_POL_NoSelector',
                              'SVR_LIN_fl_spearman', 'SVR_LIN_fl_pearson', 'SVR_LIN_RFE_RFR', 'SVR_LIN_RFE_DTR',
                              'SVR_LIN_RFE_GBR',
                              'SVR_LIN_NoSelector', 'SVR_RBF_fl_spearman', 'SVR_RBF_fl_pearson', 'SVR_RBF_RFE_RFR',
                              'SVR_RBF_RFE_DTR',
                              'SVR_RBF_RFE_GBR', 'SVR_RBF_NoSelector'],
                      single=False, predictor='SVR_RBF', ft_selector='RFE_GBR', runBlending = True)