#SCRIPT IMPORTS
from StudyReport import *
from NBestSelecting import *
from SHAPReport import *
from StudyResiduals import *
from FilterReport  import *
from Model_Averaging import *
from Model_Blending_CV import *

def RUN_Training_Report(CV_AllModels, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                  GSName = "All"):

    # FEATURE PROCESSING
    "Assessment of all filtered features - frequency of features selected or dropped"
    reportCV_Filter(CV_AllModels, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                    DBpath=DB_Values['DBpath'])

    # #MODEL PROCESSING
    # reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])

    "Assessment of all Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_All(displayParams, DB_Values["DBpath"], CV_AllModels, GSName, xQuantLabels, xQualLabels, randomValues=None)

    #AVERAGING
    "Computation of average values for models over CV runs, creation of AvgModelObject, identification of overall best models" \
    "Assessment of all models Avg and Std : 'TestAcc-Mean','TestAcc-Std', 'TestMSE-Mean', 'TestMSE-Std','Resid-Mean','Resid-Std' - List"
    RUN_Avg_Model(DB_Values['DBpath'], displayParams, BLE_VALUES, studies=CV_AllModels, ref_combined=None)



def RUN_Combine_Report(CV_AllModels, CV_NBest, CV_BlenderNBest, randomvalues, displayParams):


    "Assessment of NBest Models - ranked table for all seeds : 'TestAcc', 'TestMSE', 'Resid','Variance', 'selectedLabels', 'selector'"
    reportCV_ModelRanking_NBest(CV_AllModels, CV_NBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
    "Assessment of NBest Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_NBest(displayParams, DB_Values["DBpath"], CV_NBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)

    #BLENDER PROCESSING
    "Assessment of Blender models made from NBest models - sheet per seed - Score metrics, weights and increases - list"
    report_BL_NBest_CV(CV_BlenderNBest, displayParams,  DB_Values['DBpath'], randomvalues)
    "Assessment of Model residuals - Nbest, Blender_NBest,  - List, Gaussian Plot, Hist plot, "
    RUN_CombinedResiduals(CV_AllModels, CV_NBest, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'], n= BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])





