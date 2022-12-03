# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from FeatureReport import *
from StudyResiduals import *


studies = []
randomvalues = list(range(33, 43))
for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = 'PMV2_rd' + str(PROCESS_VALUES['random_state']) + '/'
    print('Run Study for random_state:', value)

    #RUN
    rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
    GS_FSs = Run_GS_FS_Study('CSTB_rd' + str(PROCESS_VALUES['random_state']) + '/')

    #IMPORT
    # import_reference = displayParams["reference"]
    # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)

    # GS_FSs = import_Main_GS_FS(import_reference ='CSTB_rd43/')
    # studies.append(GS_FSs)


#SELECT
# studies = [GS_FSs_43, GS_FSs_42]
# [LR, LR_RIDGE, LR_LASSO, LR_ELAST,  KRR_LIN, KRR_RBF,KRR_POL, SVR_LIN, SVR_RBF] = GS_FSs_43
# GS = LR_LASSO.RFE_GBR

#COMBINE
# plotAllResiduals(studies, displayParams, FORMAT_Values, DB_Values['DBpath'], studyFolder = 'JointHistplot')
# ReportResiduals(studies, displayParams, FORMAT_Values, DB_Values['DBpath'], studyFolder ='HistGaussPlot', binwidth = 10, setxLim = [-300, 300], fontsize = 14, sorted = True)

#todo : CHECK
#todo : fix SHAP ranking - ok?
#todo : sort SHAP per ranking > ok?
#todo : script residuals for all > ok?
#todo : understand plotter issue with residuals and SHAP - ok?
#todo: add shap report to SHAP run > ok?
#todo : make for loop for running script >ok

#todo : voting regressor


#todo : make this work for GS



#
# def SHAP_Df(GS, NbFtExtracted):
#
#     Xtrain = GS.learningDf.XTrain
#     explainer = GS.SHAPexplainer
#     shap_values = explainer.shap_values(GS.learningDf.XTest)
#
#     #export SHAP as dataframe
#     df_shap_values = pd.DataFrame(data=shap_values, columns=Xtrain.columns)
#     df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])
#     for col in df_shap_values.columns:
#         importance = df_shap_values[col].abs().mean()
#         df_feature_importance.loc[len(df_feature_importance)] = [col, importance]
#     df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)
#
#     # extract best features and give score
#     featureScoreDict = dict()
#     topNFeatures = df_feature_importance['feature'][:NbFtExtracted]
#
#     for i in range(len(list(topNFeatures))):
#         featureScoreDict[list(topNFeatures)[i]] = NbFtExtracted-i
#
#     return df_feature_importance, featureScoreDict
#
#
# df_feature_importance, featureScoreDict = SHAP_Df(GS, 5)
# print(featureScoreDict)
# print(df_feature_importance)
# print(df_feature_importance['feature'])
# print(df_feature_importance['importance'])