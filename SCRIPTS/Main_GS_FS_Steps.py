#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
from Dashboard_EUCB_FR_v2 import *

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
from ModelBlending import *
from Gridsearch import *
from GridsearchPredTruthPt import *
from GridsearchWeightsPt import *
from GridsearchParamPt import *
from PredictionReport import *
from StudyReport import *
from FeatureReport import *
from GridsearchSHAPPt import *
from Main_FS_Steps import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
import shap






def Run_GS_FS(learning_dfs): #, xQtQlLabels = (xQuantLabels, xQualLabels)
    """
    GOAL -  Calibrate model hyperparameters for different learning Dfs
    Dashboard Input - GS_VALUES ; _param_grids
    """

    #CONSTRUCT
    LR_CONSTRUCTOR = {'name' : 'LR',  'modelPredictor' : LinearRegression(),'param_dict' : dict()}
    LR_LASSO_CONSTRUCTOR = {'name' : 'LR_LASSO',  'modelPredictor' : Lasso(),'param_dict' : LR_param_grid}
    LR_RIDGE_CONSTRUCTOR = {'name' : 'LR_RIDGE',  'modelPredictor' : Ridge(),'param_dict' : LR_param_grid}
    LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
    KRR_LIN_CONSTRUCTOR = {'name': 'KRR_LIN', 'modelPredictor': KernelRidge(kernel='linear'), 'param_dict': KRR_param_grid}
    KRR_RBF_CONSTRUCTOR = {'name': 'KRR_RBF', 'modelPredictor': KernelRidge(kernel='rbf'), 'param_dict': KRR_param_grid}
    KRR_POL_CONSTRUCTOR = {'name' : 'KRR_POL',  'modelPredictor' : KernelRidge(kernel = 'poly'),'param_dict' : KRR_param_grid}
    SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
    SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}

    GS_CONSTRUCTOR = [LR_CONSTRUCTOR, LR_RIDGE_CONSTRUCTOR, LR_LASSO_CONSTRUCTOR, LR_ELAST_CONSTRUCTOR, KRR_LIN_CONSTRUCTOR,
                      KRR_RBF_CONSTRUCTOR,KRR_POL_CONSTRUCTOR, SVR_LIN_CONSTRUCTOR, SVR_RBF_CONSTRUCTOR]

    # CONSTRUCT & REPORT

    GS_FSs = []
    for constructor in GS_CONSTRUCTOR :
        GS_FS = ModelFeatureSelectionGridsearch(predictorName=constructor['name'], learningDfs=learning_dfs,
                                            modelPredictor=constructor['modelPredictor'], param_dict=constructor['param_dict'],
                                                xQtQlLabels = (xQuantLabels, xQualLabels))
        GS_FSs.append(GS_FS)
        reportGS_TxtScores_All(DB_Values['DBpath'], displayParams, GS_FS, objFolder='GS_FS', display=True)
        pickleDumpMe(DB_Values['DBpath'], displayParams, GS_FS, 'GS_FS', constructor['name'])
#
    return GS_FSs

def Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, plot_all = False):

    # SCORES
    for scoreLabel in scoreList:
        heatmap(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', score=scoreLabel, studyFolder='GS_FS/')

    if plot_all:
        for scoreLabel, scoreMax in zip(scoreList, scoreListMax):
            GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None, score=scoreLabel,
                               studyFolder='GS_FS/')
            GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None,
                               score=scoreLabel, colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'], size=[6, 6], showgrid=True,
                               maxScore=scoreMax, absVal=False, ticks=False, lims=False, studyFolder='GS_FS/')

def Plot_GS_FS_Weights(GS_FSs, baseFormatedDf):

    # WEIGHTS                   #ONLY FOR GS with identical weights
    for GS_FS in GS_FSs:
        name = GS_FS.predictorName + '_GS_FS'
        GS_WeightsBarplotAll([GS_FS], GS_FSs, DB_Values['DBpath'], displayParams, target=FORMAT_Values['targetLabels'],
                             content=name, df_for_empty_labels=baseFormatedDf.trainDf, yLim=4, sorted=True,
                             key='WeightsScaled')
    GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
                          DBpath=DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
                          df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')
    for GS_FS in GS_FSs:
        GS_WeightsSummaryPlot([GS_FS], GS_FSs, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
                              DBpath=DB_Values['DBpath'], content=GS_FS.predictorName + '_GS_FS', sorted=True, yLim=4,
                              df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')

def Plot_GS_FS_Metrics(GS_FSs, plot_all = False):
    # METRICS
    GS_MetricsSummaryPlot(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FSs', scatter=True,
                          studyFolder='GS_FS/')
    if plot_all:
        for GS_FS in GS_FSs:
            GS_MetricsSummaryPlot([GS_FS], displayParams, DB_Values['DBpath'], content=GS_FS.predictorName + '_GS_FS',
                                  scatter=True, studyFolder='GS_FS/')

def Plot_GS_FS_PredTruth(GS_FSs, plot_all = False):
    # PREDICTION VS GROUNDTRUTH
    GS_predTruthCombined(displayParams, GS_FSs, DB_Values['DBpath'], content='GS_FSs', scatter=True, fontsize=14,
                         studyFolder='GS_FS/')  # scatter=False for groundtruth as line
    for GS_FS in GS_FSs:
        GS_predTruthCombined(displayParams, [GS_FS], DB_Values['DBpath'], content=GS_FS.predictorName + '_GS_FS',
                             scatter=True, fontsize=14, studyFolder='GS_FS/')
    if plot_all:
        for GS_FS in GS_FSs:
            for learningDflabel in GS_FS.learningDfsList:
                GS = GS_FS.__getattribute__(learningDflabel)
                plotPredTruth(displayParams=displayParams, modelGridsearch=GS, DBpath=DB_Values['DBpath'],
                              TargetMinMaxVal=FORMAT_Values['TargetMinMaxVal'], fontsize=14, studyFolder='GS_FS/')

def Plot_GS_FS_Residuals(GS_FSs, plot_all = False):

    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            if plot_all:
                plotModelHistResiduals(modelGridsearch=GS, displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                       bins=20, binrange=[-200, 200], studyFolder='GS_FS/')
            plotModelYellowResiduals(modelGridsearch=GS, displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                     yLim=PROCESS_VALUES['residualsYLim'], xLim=PROCESS_VALUES['residualsXLim'],
                                     studyFolder='GS_FS/')

def Plot_GS_FS_SHAP(GS_FSs, plot_shap = True, plot_shap_decision = False):

    if plot_shap:

        Model_List = unpackGS_FSs(GS_FSs)

        for GS in Model_List:
            plot_shap_group_cat_SummaryPlot(GS, xQuantLabels, xQualLabels, displayParams=displayParams, DBpath=DB_Values['DBpath'])
            plot_shap_SummaryPlot(GS, displayParams, DBpath=DB_Values['DBpath'], content='', studyFolder='GS_FS/')

            if plot_shap_decision :
                if GS.selectorName != 'NoSelector':
                    plot_shap_group_cat_DecisionPlot(GS, displayParams, DBpath=DB_Values['DBpath'], studyFolder='GS_FS/')
                    plot_shap_DecisionPlot(GS, displayParams, DBpath=DB_Values['DBpath'], studyFolder='GS_FS/')

def Run_Blending(GS_FSs, displayParams, DBpath, n, checkR2, NBestScore ='TestR2'):
    #CONSTRUCT
    LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
    LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}

    # CONSTRUCT & REPORT
    sortedModelsData = sortedModels(GS_FSs, NBestScore= NBestScore)
    nBestModels = selectnBestModels(GS_FSs, sortedModelsData, n, checkR2 = checkR2)
    blendModel = BlendModel(modelList=nBestModels, blendingConstructor=LR_RIDGE_CONSTRUCTOR, NBestScore= NBestScore)
    reportGS_Scores_Blending(blendModel, displayParams, DBpath, NBestScore)
    pickleDumpMe(DBpath, displayParams, blendModel, 'GS_FS', blendModel.GSName + '_' + NBestScore) #TODO : the score was changed here

    return blendModel

def Run_GS_FS_Study(import_FS_ref, importMainGSFS = False):
    """
    MODEL x FEATURE SELECTION GRIDSEARCH
    """
    # #IMPORT Main_FS
    rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_FS_ref,
                                                                                               show=False)
    learning_dfs = [spearmanFilter, pearsonFilter] + RFEs + [baseFormatedDf]


    # # IMPORT Main_GS_FS
    if importMainGSFS : #models already calibrated
        print('')
        print('IMPORTING GS_FS')
        GS_FSs = import_Main_GS_FS(import_FS_ref)
    else:
    # RUN GS_FS
        print('')
        print('RUNNING GS_FS')
        GS_FSs = Run_GS_FS(learning_dfs)

    # BLEND
    print('RUNNING BLENDING')
    blendModel = Run_Blending(GS_FSs, displayParams, DB_Values["DBpath"], 10, checkR2 = True, NBestScore= BLE_VALUES['NBestScore'])

    # REPORT
    print('REPORTING GS_FS')
    reportGS_Details_All(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, df, learningDf,
                         baseFormatedDf, FiltersLs=[spearmanFilter, pearsonFilter], RFEs=RFEs, GSlist=GS_FSs, GSwithFS=True)

    scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
    scoreListMax = [True, False, True, True, True]
    reportGS_Scores_All(DB_Values['DBpath'], displayParams, GS_FSs, scoreList=scoreList, display=True)

    reportGS_FeatureWeights(DB_Values['DBpath'], displayParams, GS_FSs, blender=blendModel)
    reportGS_FeatureSHAP(DB_Values['DBpath'], displayParams, GS_FSs, xQuantLabels, xQualLabels, blender=blendModel)

    print('PLOTTING GS_FS')
    # PLOT
    Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, plot_all = displayParams['plot_all'])
    Plot_GS_FS_Weights(GS_FSs, baseFormatedDf)
    Plot_GS_FS_Metrics(GS_FSs, plot_all = False)
    Plot_GS_FS_PredTruth(GS_FSs, plot_all = False)
    Plot_GS_FS_Residuals(GS_FSs, plot_all=False)
    Plot_GS_FS_SHAP(GS_FSs, plot_shap = True, plot_shap_decision = displayParams['plot_all'])

    return GS_FSs, blendModel

def import_Main_GS_FS(import_reference, GS_FS_List_Labels = ['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_LIN', 'KRR_RBF', 'KRR_POL', 'SVR_LIN','SVR_RBF']): #'SVR_POL'

    GS_FSs = []
    for FS_GS_lab in GS_FS_List_Labels:
        path = DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/GS_FS/' + FS_GS_lab + '.pkl'
        # path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/' + import_reference + 'RECORDS/GS_FS/' + FS_GS_lab + '.pkl'
        GS_FS = pickleLoadMe(path=path, show=False)

        GS_FSs.append(GS_FS)

    return GS_FSs

def import_Main_Blender(import_reference, NBestScore, label = 'LR_RIDGE_Blender'):
    path = DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/GS_FS/' + label + '_' + NBestScore + '.pkl'
    Blender = pickleLoadMe(path=path, show=False)

    return Blender

def unpackGS_FSs(GS_FSs, remove = ''):
    Model_List = []
    for GS_FS in GS_FSs:
        for learningDflabel in GS_FS.learningDfsList:
            GS = GS_FS.__getattribute__(learningDflabel)
            if GS.predictorName != remove:
                Model_List.append(GS)
    return Model_List
