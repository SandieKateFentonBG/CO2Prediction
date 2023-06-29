#SCRIPT IMPORTS
from StudyReport import *
from NBestSelecting import *
from SHAPReport import *
from StudyResiduals import *
from FilterReport  import *
from Model_Averaging import *
from Model_Blending_CV import *

def RUN_Training_Report(CV_AllModels, Filters_CV, randomvalues, displayParams, studyParams,
                  GSName = "All"):

    if len(studyParams['fl_selectors'])>0:
        print("CV_Filter")
        # FEATURE PROCESSING
        "Assessment of all filtered features - frequency of features selected or dropped"
        reportCV_Filter(CV_AllModels, Filters_CV, randomvalues, displayParams, studyParams,
                        DBpath=DB_Values['DBpath'])

    print("SHAP_Combined")

    "Assessment of all Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_All(displayParams, DB_Values["DBpath"], CV_AllModels, GSName, xQuantLabels, xQualLabels, randomValues=randomvalues)

    print("Avg_Model")
    #AVERAGING
    "Computation of average values for models over CV runs, creation of AvgModelObject, identification of overall best models" \
    "Assessment of all models Avg and Std : 'TestAcc-Mean','TestAcc-Std', 'TestMSE-Mean', 'TestMSE-Std','Resid-Mean','Resid-Std' - List"
    RUN_Avg_Model(DB_Values['DBpath'], displayParams, BLE_VALUES, studies=CV_AllModels, ref_combined=None)


def RUN_Combine_Report(CV_AllModels, CV_NBest, CV_BlenderNBest, regressors_CV, models_CV, randomvalues, displayParams):


    "Assessment of NBest Models - ranked table for all seeds : 'TestAcc', 'TestMSE', 'Resid','Variance', 'selectedLabels', 'selector'"
    reportCV_ModelRanking_NBest(CV_AllModels, CV_NBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
    "Assessment of NBest Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_NBest(displayParams, DB_Values["DBpath"], CV_NBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)

    #BLENDER PROCESSING
    "Assessment of Blender models made from NBest models - sheet per seed - Score metrics, weights and increases - list"
    for blender_type in CV_BlenderNBest:
        report_BL_NBest_CV(blender_type, displayParams,  DB_Values['DBpath'], randomvalues)

    "Assessment of Model residuals - All, Nbest, Blender_NBest, single regressor, single model   - List, Gaussian Plot, Hist plot, "
    RUN_CombinedResiduals(CV_AllModels, CV_NBest, CV_BlenderNBest, regressors_CV, models_CV, displayParams, FORMAT_Values,
                          DB_Values['DBpath'], randomvalues, setyLim=[-500, 500], setxLim=[0, 1500])



