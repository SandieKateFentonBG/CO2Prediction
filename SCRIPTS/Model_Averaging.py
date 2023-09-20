
from GridsearchParamPt import GS_ParameterPlot2D
from StudyResiduals import *
from Main_GS_FS_Steps import *


def avgModel( DBpath, displayParams, studies = None, ResultsDf = None):

    "#todo : fix this : either provide studies = list of GS OR provide ResultsDf"

    if studies:
        AvgDict, lazy_models = computeCV_Scores_Avg_All(studies) # creates ResultsDf
    else:
        AvgDict = ResultsDf

    # studies = 10 * 9 * 6 - GS_FSs
    # Gsearch = 9 * 6 - GS_FS
    # PSearch = 6 - FS
    # Model - 1

    #create empty Gridsearch
    new = studies[0]

    for predictor in new : #9
        for learningDflabel in predictor.learningDfsList:  # 6
            GS = predictor.__getattribute__(learningDflabel)

            TestAcc, TestAccStd, TestMSE, TestMSEStd, TestR2, TestR2Std , Resid, ResidStd, ResidVariance, \
            ResidVarianceStd, TrainScore, TrainScoreStd, TestScore, TestScoreStd= AvgDict.loc[GS.GSName, :]

            setattr(GS, 'TestAcc', TestAcc)
            setattr(GS, 'TestMSE', TestMSE)
            setattr(GS, 'TestR2', TestR2)
            setattr(GS, 'ResidMean', Resid)
            setattr(GS, 'ResidVariance', ResidVariance)
            setattr(GS, 'TrainScore', TrainScore)
            setattr(GS, 'TestScore', TestScore)


            setattr(predictor, learningDflabel, GS)

        pickleDumpMe(DBpath, displayParams, predictor, 'GS_FS', predictor.predictorName, combined=True)


    return new



def RUN_Avg_Model(DBpath, displayParams, BLE_VALUES, studies = None, ref_combined =  None):
    print(BLE_VALUES['NBestScore'])

    if not studies:
        All_CV = import_Main_GS_FS(ref_combined,
                               GS_FS_List_Labels=studyParams['Regressors'])
    else:
        All_CV = studies

    # CREATE AVG MODEL
    ResultsDf, lazy_models = computeCV_Scores_Avg_All(All_CV)

    GS_FSs = avgModel(DBpath, displayParams, studies=All_CV, ResultsDf = ResultsDf) #

    # REPORT
    reportCV_ScoresAvg_All(ResultsDf, displayParams, DBpath, NBestScore=BLE_VALUES['NBestScore'])

    #PLOT
    scoreList = ['TestAcc', 'TestMSE', 'TestR2', 'TrainScore', 'TestScore']
    scoreListMax = [True, False, True, True, True]
    Plot_GS_FS_Scores(GS_FSs, scoreList, scoreListMax, combined=True, plot_all=displayParams['plot_all'])

    # FIND NBEST
    BestModelNames = find_Overall_Best_Models(DBpath, displayParams, ResultsDf, lazy_labels = lazy_models, n=BLE_VALUES['NCount'], NBestScore=BLE_VALUES['NBestScore'])

    return GS_FSs










