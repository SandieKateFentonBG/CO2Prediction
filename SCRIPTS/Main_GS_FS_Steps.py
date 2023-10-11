#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
# from Dashboard_EUCB_FR_v2 import *
# from Dashboard_EUCB_Structures import *
from Dashboard_Current import *

#SCRIPT IMPORTS
from HelpersArchiver import *
from HelpersFormatter import *
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

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
import shap






def Run_GS_FS(learning_dfs, regressors): #, xQtQlLabels = (xQuantLabels, xQualLabels)
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


    MLP_LBFG_CONSTRUCTOR = {'name': 'MLP_LBFG', 'modelPredictor': MLPRegressor(solver='lbfgs'), 'param_dict': MLP_LBFG_param_grid}
    MLP_SGD_CONSTRUCTOR = {'name': 'MLP_SGD', 'modelPredictor': MLPRegressor(solver='sgd'), 'param_dict': MLP_SGD_param_grid}
    MLP_ADAM_CONSTRUCTOR = {'name': 'MLP_ADAM', 'modelPredictor': MLPRegressor(solver='adam'), 'param_dict': MLP_ADAM_param_grid}

    MLP_LBFG_20_CONSTRUCTOR = {'name': 'MLP_LBFG_20', 'modelPredictor': MLPRegressor(solver='lbfgs'), 'param_dict': MLP_LBFG_20_param_grid}
    MLP_SGD_20_CONSTRUCTOR = {'name': 'MLP_SGD_20', 'modelPredictor': MLPRegressor(solver='sgd'), 'param_dict': MLP_SGD_20_param_grid}
    MLP_LBFG_10_CONSTRUCTOR = {'name': 'MLP_LBFG_10', 'modelPredictor': MLPRegressor(solver='lbfgs'), 'param_dict': MLP_LBFG_10_param_grid}
    MLP_SGD_10_CONSTRUCTOR = {'name': 'MLP_SGD_10', 'modelPredictor': MLPRegressor(solver='sgd'), 'param_dict': MLP_SGD_10_param_grid}
    MLP_LBFG_100_CONSTRUCTOR = {'name': 'MLP_LBFG_100', 'modelPredictor': MLPRegressor(solver='lbfgs'), 'param_dict': MLP_LBFG_100_param_grid}
    MLP_SGD_100_CONSTRUCTOR = {'name': 'MLP_SGD_100', 'modelPredictor': MLPRegressor(solver='sgd'), 'param_dict': MLP_SGD_100_param_grid}



    All_CONSTRUCTOR = [LR_CONSTRUCTOR, LR_RIDGE_CONSTRUCTOR, LR_LASSO_CONSTRUCTOR, LR_ELAST_CONSTRUCTOR, KRR_LIN_CONSTRUCTOR,
                      KRR_RBF_CONSTRUCTOR,KRR_POL_CONSTRUCTOR, SVR_LIN_CONSTRUCTOR, SVR_RBF_CONSTRUCTOR,
                        MLP_SGD_CONSTRUCTOR, MLP_SGD_20_CONSTRUCTOR, MLP_SGD_10_CONSTRUCTOR, MLP_SGD_100_CONSTRUCTOR,
                       MLP_LBFG_CONSTRUCTOR, MLP_LBFG_20_CONSTRUCTOR,  MLP_LBFG_10_CONSTRUCTOR, MLP_LBFG_100_CONSTRUCTOR,
                       MLP_SAG_CONSTRUCTOR] #

    GS_CONSTRUCTOR = [elem for elem in All_CONSTRUCTOR if elem['name'] in regressors]

    # CONSTRUCT & REPORT

    GS_FSs = []
    for constructor in GS_CONSTRUCTOR :
        GS_FS = ModelFeatureSelectionGridsearch(predictorName=constructor['name'], learningDfs=learning_dfs,
                                            modelPredictor=constructor['modelPredictor'], param_dict=constructor['param_dict']
                                                , acc = PROCESS_VALUES['accuracyTol'],
                                                refit = PROCESS_VALUES['refit'],
                                                xQtQlLabels = (xQuantLabels, xQualLabels))
        GS_FSs.append(GS_FS)
        reportGS_TxtScores_All(DB_Values['DBpath'], displayParams, GS_FS, objFolder='GS_FS', display=True)


        pickleDumpMe(DB_Values['DBpath'], displayParams, GS_FS, 'GS_FS', constructor['name'])
#
    return GS_FSs

def Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, combined = False, plot_all = False):

    # SCORES
    for scoreLabel in scoreList:
        heatmap(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', score=scoreLabel, studyFolder='GS_FS/', combined = combined)

    if plot_all:
        for scoreLabel, scoreMax in zip(scoreList, scoreListMax):
            GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None, score=scoreLabel,
                               studyFolder='GS_FS/', combined = combined)
            GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', yLim=None,
                               score=scoreLabel, colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'], size=[6, 6], showgrid=True,
                               maxScore=scoreMax, absVal=False, ticks=False, lims=False, studyFolder='GS_FS/', combined = combined)

def Plot_GS_FS_Weights(GS_FSs, baseFormatedDf, NBestModel = None):

    # WEIGHTS                   #ONLY FOR GS with identical weights
    for GS_FS in GS_FSs:
        if GS_FS.isLinear == True:
            name = GS_FS.predictorName + '_GS_FS'
            GS_WeightsBarplotAll([GS_FS], GS_FSs, DB_Values['DBpath'], displayParams, target=FORMAT_Values['targetLabels'],
                                 content=name, df_for_empty_labels=baseFormatedDf.trainDf, yLim=4, sorted=True,
                                 key='WeightsScaled')
    GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
                          DBpath=DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
                          df_for_empty_labels=baseFormatedDf.trainDf, fontsize=14, studyFolder='GS_FS/')
    if NBestModel:
        GS_WeightsSummaryPlot_NBest(NBestModel, target=FORMAT_Values['targetLabels'], displayParams=displayParams,
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

def Plot_GS_FS_Hyperparam(import_reference, Plot=False):
    if Plot:

        GS_FSs_alpha = import_Main_GS_FS(import_reference,
                                         GS_FS_List_Labels=['LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN',
                                                            'KRR_POL'])  #
        GS_FSs_gamma = import_Main_GS_FS(import_reference,
                                         GS_FS_List_Labels=['KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'])  #
        GS_FSs_degree = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_POL'])  #
        GS_FSs_epsilon = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['SVR_LIN', 'SVR_RBF'])  #
        GS_FSs_C = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['SVR_LIN', 'SVR_RBF'])  #
        GS_FSs_coef0 = import_Main_GS_FS(import_reference, GS_FS_List_Labels=['KRR_POL'])  #

        for ML, key, log, maxVal, absVal in zip(
                [GS_FSs_alpha, GS_FSs_gamma, GS_FSs_degree, GS_FSs_epsilon, GS_FSs_C, GS_FSs_coef0],
                ['alpha', 'gamma', 'degree', 'epsilon', 'C', 'coef0'],
                [True, True, False, True, True, True],
                [True, True, True, True, True, True], [True, True, True, True, True, True]):
            Model_List = unpackGS_FSs(ML, remove='')

            ParameterPlot2D(Model_List, displayParams, DB_Values['DBpath'], yLim=None,
                            paramKey=key, score='mean_test_r2', log=log, studyFolder='GS/')
            ParameterPlot3D(Model_List, displayParams, DB_Values['DBpath'],
                            colorsPtsLsBest=['b', 'g', 'c', 'y'], paramKey=key, score='mean_test_r2',
                            size=[6, 6], showgrid=False, log=log, maxScore=maxVal, absVal=absVal, ticks=False,
                            lims=False,
                            studyFolder='GS/')

def Run_NBest(GS_FSs, OverallBest = False):

    if OverallBest:
        BestModelNames = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + displayParams['ref_prefix'] + '_Combined/' +
                              'RECORDS/GS_FS/BestModelNames.pkl')

        NBestModels = OBestModel(GS_FSs, NBestScore=BLE_VALUES['NBestScore'], NCount=BLE_VALUES['NCount'], BestModelNames = BestModelNames)
    else:
        NBestModels = NBestModel(GS_FSs, NBestScore =  BLE_VALUES['NBestScore'], NCount = BLE_VALUES['NCount']) #todo model selection changed for MSE
    pickleDumpMe(DB_Values['DBpath'], displayParams, NBestModels, 'NBEST', NBestModels.GSName)
    reportGS_Scores_NBest(NBestModels, displayParams, DBpath=DB_Values['DBpath'])

    return NBestModels

def Run_NBest_Study(import_FS_ref, import_GS_FS_ref, importNBest = False, OverallBest = False):

    # # IMPORT Main_GS_FS
    print('IMPORTING GS_FS')
    GS_FSs = import_Main_GS_FS(import_GS_FS_ref)

    baseFormatedDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_FS_ref +'RECORDS/DATA/baseFormatedDf.pkl', show = False) #check this

    # NBEST

    # # IMPORT NBEST
    if importNBest : #models already calibrated
        print('')
        print('IMPORTING NBEST')
        NBestModels = import_NBest(import_GS_FS_ref, OverallBest = OverallBest)
    else:
        print('')
        print('RUNNING NBEST')
        NBestModels = Run_NBest(GS_FSs, OverallBest = OverallBest)

    # REPORT
    print('REPORTING GS_FS & NBEST')
    if displayParams['report_all']:
        reportGS_FeatureWeights(DB_Values['DBpath'], displayParams, GS_FSs, NBestModel=NBestModels)
    reportGS_FeatureSHAP(DB_Values['DBpath'], displayParams, GS_FSs, xQuantLabels, xQualLabels, NBestModel=NBestModels)

    print('PLOTTING GS_FS & NBEST')
    # PLOT
    if displayParams['plot_all']:
        Plot_GS_FS_Weights(GS_FSs, baseFormatedDf, NBestModels)

    return NBestModels

def Run_GS_FS_Study(import_FS_ref, importMainGSFS = False, importMainFS=True, FS=None):
    """
    MODEL x FEATURE SELECTION GRIDSEARCH
    """
    # #IMPORT Main_FS
    if importMainFS:
        rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList = import_Main_FS(import_FS_ref,show=False)
    else:
        [rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList] = FS

    learning_dfs = [baseFormatedDf] #not sure order counts
    if len(filterList)>0:
        learning_dfs += filterList
    if len(RFEList)>0:
        learning_dfs += RFEList

    # todo : this was changed from ldf = filterList + RFEList + [baseFormatedDf] to be sure no empty learning dfs are provided  > check it works


    # # IMPORT Main_GS_FS
    if importMainGSFS : #models already calibrated
        print('')
        print('IMPORTING GS_FS')
        GS_FSs = import_Main_GS_FS(import_FS_ref, GS_FS_List_Labels=studyParams['Regressors'])
    else:
    # RUN GS_FS
        print('')
        print('RUNNING GS_FS')
        GS_FSs = Run_GS_FS(learning_dfs, regressors=studyParams['Regressors'])

    # REPORT
    print('REPORTING GS_FS')
    reportGS_Details_All(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, rdat, dat, df,
                         learningDf, baseFormatedDf, FiltersLs=filterList, RFEs=RFEList, GSlist=GS_FSs, GSwithFS=True)

    scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
    scoreListMax = [True, False, True, True, True]
    reportGS_Scores_All(DB_Values['DBpath'], displayParams, GS_FSs, scoreList=scoreList, display=False)

    print('PLOTTING GS_FS')
    # PLOT
    Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, plot_all = displayParams['plot_all'])

    Plot_GS_FS_Metrics(GS_FSs, plot_all = False)
    Plot_GS_FS_PredTruth(GS_FSs, plot_all = False)
    Plot_GS_FS_Residuals(GS_FSs, plot_all=False)
    Plot_GS_FS_SHAP(GS_FSs, plot_shap = True, plot_shap_decision = displayParams['plot_all'])
    Plot_GS_FS_Hyperparam(import_FS_ref, Plot=False)

    return GS_FSs


def report_GS_FS(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, BLE_VALUES,
                 rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList, GS_FSs):

    # REPORT
    print('REPORTING GS_FS')
    reportGS_Details_All(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, BLE_VALUES, rdat,
                         dat, df, learningDf, baseFormatedDf, FiltersLs=filterList, RFEs=RFEList, GSlist=GS_FSs, GSwithFS=True)

    scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
    scoreListMax = [True, False, True, True, True]
    reportGS_Scores_All(DB_Values['DBpath'], displayParams, GS_FSs, scoreList=scoreList, display=False)

    print('PLOTTING GS_FS')
    # PLOT
    Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, plot_all = displayParams['plot_all'])

    Plot_GS_FS_Metrics(GS_FSs, plot_all = False)
    Plot_GS_FS_PredTruth(GS_FSs, plot_all = False)
    Plot_GS_FS_Residuals(GS_FSs, plot_all=False)
    Plot_GS_FS_SHAP(GS_FSs, plot_shap = True, plot_shap_decision = displayParams['plot_all'])
    Plot_GS_FS_Hyperparam(displayParams['reference'], Plot=False)

def import_Main_GS_FS(import_reference, GS_FS_List_Labels = studyParams['Regressors']): #'SVR_POL'

    # if Combined:import_reference = ref_prefix + '_Combined/'
    # is single:import_reference = ref_prefix + 'rd' + str(PROCESS_VALUES['random_state']) + '/'

    GS_FSs = []
    for FS_GS_lab in GS_FS_List_Labels: #9
        path = DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/GS_FS/' + FS_GS_lab + '.pkl'
        print("importing : ", path)
        GS_FS = pickleLoadMe(path=path, show=False) #6
        GS_FSs.append(GS_FS) #54

    return GS_FSs


