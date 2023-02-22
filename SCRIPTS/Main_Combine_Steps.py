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


def RUN_CV_Report(CV_AllModels, CV_BlenderNBest, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                  GSName = "LR"):

    reportCV_Filter(CV_AllModels, CV_Filters_Spearman, CV_Filters_Pearson, randomvalues, displayParams,
                    DBpath=DB_Values['DBpath'])
    reportCV_Scores_NBest(CV_BlenderNBest, displayParams, DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], random_seeds = randomvalues)
    reportCV_ScoresAvg_All(CV_AllModels, displayParams, DB_Values['DBpath'])
    reportCV_ModelRanking_NBest(CV_AllModels, CV_BlenderNBest, seeds = randomvalues, displayParams = displayParams, DBpath =DB_Values['DBpath'], n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])

    RUN_SHAP_Combined_NBest(displayParams, DB_Values["DBpath"], CV_BlenderNBest, CV_AllModels, xQuantLabels, xQualLabels, n = BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'], randomValues = randomvalues)
    RUN_SHAP_Combined_All(displayParams, DB_Values["DBpath"], CV_AllModels, GSName, xQuantLabels, xQualLabels, randomValues=None)

    RUN_CombinedResiduals(CV_AllModels, CV_BlenderNBest, displayParams, FORMAT_Values, DBpath = DB_Values['DBpath'], n= BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])
