#SCRIPT IMPORTS
from HelpersArchiver import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Filter import *
from FilterVisualizer import *
from Wrapper import *
from WrapperVisualizer import *
from Model import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *
from ModelMetricsPt import *
from ModelWeightsPt import *
from NBestSelecting import *
from Gridsearch import *
from GridsearchPredTruthPt import *
from GridsearchWeightsPt import *
from GridsearchParamPt import *
from PredictionReport import *
from StudyReport import *
from FeatureReport import *
from GridsearchSHAPPt import *
from Main_FS_Steps import *
from FeatureReport import *
from StudyResiduals import *
from NBestSelecting import *
from SHAPReport import *
from AccuracyCheck import *
from StudyResiduals import *
from FilterReport  import *
from Model_Blending import *


def RUN_CV_Report(CV_AllModels, CV_NBest, CV_BlenderNBest, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                  GSName = "All"): # GSName can be used for studying single model

    #FEATURE PROCESSING
    "Assessment of all filtered features - frequency of features selected or dropped"
    reportCV_Filter(CV_AllModels, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                    DBpath=DB_Values['DBpath'])

    #MODEL PROCESSING
    "Assessment of all models Avg and Std : 'TestAcc-Mean','TestAcc-Std', 'TestMSE-Mean', 'TestMSE-Std','Resid-Mean','Resid-Std' - List"
    reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])

    "Assessment of all Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_All(displayParams, DB_Values["DBpath"], CV_AllModels, GSName, xQuantLabels, xQualLabels, randomValues=None)

    "Assessment of NBest Models - ranked table for all seeds : 'TestAcc', 'TestMSE', 'Resid','Variance', 'selectedLabels', 'selector'"
    reportCV_ModelRanking_NBest(CV_AllModels, CV_NBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
    "Assessment of NBest Models - ranked table for all seeds and samples - features SHAP values Grouped and Ungrouped - List + Plots"
    RUN_SHAP_Combined_NBest(displayParams, DB_Values["DBpath"], CV_NBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)

    #BLENDER PROCESSING
    "Assessment of Blender models made from NBest models - sheet per seed - Score metrics, weights and increases - list"
    report_BL_NBest_CV(CV_BlenderNBest, displayParams,  DB_Values['DBpath'], randomvalues)
    "Assessment of Model residuals - Nbest, Blender_NBest,  - List, Gaussian Plot, Hist plot, "
    RUN_CombinedResiduals(CV_AllModels, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'], n= BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
