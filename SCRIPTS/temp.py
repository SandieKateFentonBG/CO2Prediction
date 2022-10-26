
# GS_predTruthCombined(displayParams, [LR_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
# GS_predTruthCombined(displayParams, [LR_RIDGE_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_RIDGE_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
# GS_predTruthCombined(displayParams, [LR_LASSO_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_LASSO_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
# GS_predTruthCombined(displayParams, [LR_ELAST_FS_GS], DBpath = DB_Values['DBpath'], content = 'LR_ELAST_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
# GS_predTruthCombined(displayParams, [KRR_FS_GS], DBpath = DB_Values['DBpath'], content = 'KRR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
# GS_predTruthCombined(displayParams, [SVR_FS_GS], DBpath = DB_Values['DBpath'], content = 'SVR_FS_GS', scatter=True, fontsize=14, studyFolder = 'GS_FS/')
#
# #
#

# GS_MetricsSummaryPlot(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FSs', scatter=True, studyFolder = 'GS_FS/')
#
# GS_MetricsSummaryPlot([LR_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_RIDGE_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_RIDGE_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_LASSO_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_LASSO_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([LR_ELAST_FS_GS], displayParams, DB_Values['DBpath'], content = 'LR_ELAST_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([KRR_FS_GS], displayParams, DB_Values['DBpath'], content = 'KRR_FS_GS', scatter=True, studyFolder = 'GS_FS/')
# GS_MetricsSummaryPlot([SVR_FS_GS], displayParams, DB_Values['DBpath'], content = 'SVR_FS_GS', scatter=True, studyFolder = 'GS_FS/')


#ONLY FOR identical weights
# GS_WeightsBarplotAll([LR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_RIDGE_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_RIDGE_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_LASSO_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_LASSO_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([LR_ELAST_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'LR_ELAST_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([KRR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'KRR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')
# GS_WeightsBarplotAll([SVR_FS_GS], GS_FSs, DB_Values['DBpath'], displayParams, target = FORMAT_Values['targetLabels'], content = 'SVR_FS_GS',
#                      df_for_empty_labels=baseFormatedDf.trainDf, yLim = 4, sorted = True, key ='WeightsScaled')

# GS_WeightsSummaryPlot([LR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_RIDGE_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_RIDGE_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_LASSO_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_LASSO_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([LR_ELAST_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='LR_ELAST_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([KRR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='KRR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')
#
# GS_WeightsSummaryPlot([SVR_FS_GS], GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='SVR_FS_GS', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')

# GS_WeightsSummaryPlot(GS_FSs, GS_FSs, target = FORMAT_Values['targetLabels'],
#                       df_for_empty_labels=baseFormatedDf.trainDf, displayParams =displayParams,
#                       DBpath = DB_Values['DBpath'], content='GS_FSs', sorted=True, yLim=4,
#                           fontsize=14,  studyFolder='GS_FS/')


GS_FSs = [LR_FS_GS, LR_RIDGE_FS_GS, LR_LASSO_FS_GS, LR_ELAST_FS_GS, KRR_FS_GS]
# GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,score ='TestAcc', studyFolder = 'GS_FS/')
# GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,score ='TestMSE', studyFolder = 'GS_FS/')
# GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,score ='TestR2', studyFolder = 'GS_FS/')
# GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,score ='TrainScore', studyFolder = 'GS_FS/')
# GS_ParameterPlot2D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,score ='TestScore', studyFolder = 'GS_FS/')


# GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,
#                     score ='TestAcc',
#                     colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'],
#                     size=[6, 6], showgrid=True,  maxScore=True, absVal = False, ticks=False, lims=False,
#                     studyFolder = 'GS_FS/')
#
# GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,
#                     score ='TestMSE',
#                     colorsPtsLsBest=['b', 'g', 'c', 'y', 'r' ],
#                     size=[6, 6], showgrid=True,  maxScore=False, absVal = False, ticks=False, lims=False,
#                     studyFolder = 'GS_FS/')
#
# GS_ParameterPlot3D(GS_FSs, displayParams, DB_Values['DBpath'], content = 'GS_FS', yLim = None,
#                     score ='TestR2',
#                     colorsPtsLsBest=['b', 'g', 'c', 'y', 'r'],
#                     size=[6, 6], showgrid=True,  maxScore=True, absVal = False, ticks=False, lims=False,
#                     studyFolder = 'GS_FS/')

# heatmap(GS_FSs, displayParams, DB_Values['DBpath'], content='GS_FS', score='TestAcc', studyFolder='GS_FS/')

# reportGridsearchAsTable (DB_Values['DBpath'], displayParams, GS_FSs, scoreList = ['TestAcc', 'TestMSE', 'TestR2'], objFolder ='GS_FS', display = True)


#
LR_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_FS_GS.pkl', show = False)
LR_RIDGE_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_RIDGE_FS_GS.pkl', show = False)
LR_LASSO_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_LASSO_FS_GS.pkl', show = True)
LR_ELAST_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/LR_ELAST_FS_GS.pkl', show = True)
KRR_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/KRR_FS_GS.pkl', show = True)
SVR_GS_FS = pickleLoadMe(path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/221025_RUN/RECORDS/GS_FS/SVR_FS_GS.pkl', show = True)
#>>>
GS_FSs = [LR_GS_FS, LR_RIDGE_GS_FS, LR_LASSO_GS_FS, LR_ELAST_GS_FS, KRR_GS_FS, SVR_GS_FS]
