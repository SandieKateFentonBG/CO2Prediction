# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Main_FS_Def import *
from Main_GS_FS_Def import *

import_reference = displayParams["reference"]

# rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = run_FS_Study()
# GS_FSs = run_GS_FS_Study(import_reference)



# rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
GS_FSs = import_Main_GS_FS(import_reference)

[LR, LR_RIDGE, LR_LASSO, LR_ELAST,  KRR_LIN, KRR_RBF,KRR_POL, SVR_LIN, SVR_RBF] = GS_FSs

GS = LR_LASSO.RFE_GBR
print(GS)

#todo : move script in in Model.py
#todo : export feature importance in attributes
#todo : export ranking and score to top n features in attributes
#todo: function that assembles all rankings and returns top 5

#todo > same for grouped shap?

#todo > run 10 times
#todo > combine results for residuals and plot


def SHAP_Df(GS, NbFtExtracted):

    Xtrain = GS.learningDf.XTrain
    explainer = GS.SHAPexplainer
    shap_values = explainer.shap_values(GS.learningDf.XTest)

    #export SHAP as dataframe
    df_shap_values = pd.DataFrame(data=shap_values, columns=Xtrain.columns)
    df_feature_importance = pd.DataFrame(columns=['feature', 'importance'])
    for col in df_shap_values.columns:
        importance = df_shap_values[col].abs().mean()
        df_feature_importance.loc[len(df_feature_importance)] = [col, importance]
    df_feature_importance = df_feature_importance.sort_values('importance', ascending=False)

    # extract best features and give score
    featureScoreDict = dict()
    topNFeatures = df_feature_importance['feature'][:NbFtExtracted]

    for i in range(len(list(topNFeatures))):
        featureScoreDict[list(topNFeatures)[i]] = i

    return df_feature_importance, featureScoreDict


df_feature_importance, featureScoreDict = SHAP_Df(GS, 5)
print(df_feature_importance)