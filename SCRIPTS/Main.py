# todo : choose database

#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
from Dashboard_EUCB_FR import *

#SCRIPT IMPORTS
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from FeatureReport import *
from StudyResiduals import *
from ModelBlending import *

# 'CSTB_rd' 'PMV2_rd''CSTB_A123_rd''CSTB_rd'
DBname = 'CSTB_rd' #DB_Values['acronym'] + yLabels[0] + '_rd'
#todo : change database link !

studies = []
randomvalues = list(range(42, 43))
for value in randomvalues:
    PROCESS_VALUES['random_state'] = value
    displayParams["reference"] = DBname + str(PROCESS_VALUES['random_state']) + '/'
    print('Run Study for random_state:', value)

    #RUN
    # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = Run_FS_Study()
    # GS_FSs, blendModel = Run_GS_FS_Study(DBname + str(PROCESS_VALUES['random_state']) + '/')

    #IMPORT
    import_reference = displayParams["reference"]
    # rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
    GS_FSs = import_Main_GS_FS(import_reference)
    # Blender = import_Main_Blender(import_reference)

    studies.append(GS_FSs)

    Model_List = unpackGS_FSs(GS_FSs)
    for m in Model_List:
        print(m.GSName)
        # print(len(m.Resid), m.Resid)

        for i in range(len(m.Resid)):
            if abs(m.Resid[i]) > 100:
                print(i, 'resid :', m.Resid[i], 'Ypred :', m.yPred[i])
                print('Y', m.learningDf.yTest.iloc[[i]])
                # print('X', m.learningDf.XTest.iloc[[i]])

                # print('Ypred', m.yPred[i])
                # print('resid', m.Resid[i])
        print([elem for elem in m.Resid if abs(elem) > 100])



#SELECT

# import_reference = displayParams["reference"]
# rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(import_reference, show = False)
# GS_FSs = import_Main_GS_FS(import_reference)
#

# PREDICT
# PredictionDict = computePrediction(GS)

# BLEND
# nBestModels, blendModel = Run_Blending(GS_FSs, displayParams, DB_Values["DBpath"], 10, display = True)

#COMBINE
plotAllResiduals(studies, displayParams, FORMAT_Values, DB_Values['DBpath'], studyFolder = 'JointHistplot')
# ReportResiduals(studies, displayParams, FORMAT_Values, DB_Values['DBpath'], studyFolder ='HistGaussPlot', binwidth = 0.2, setxLim = [-3, 3], fontsize = 14, sorted = True)
# ReportResiduals(studies, displayParams, FORMAT_Values, DB_Values['DBpath'], studyFolder ='HistGaussPlot', binwidth = 10, setxLim = [-300, 300], fontsize = 14, sorted = True)



#todo : add mean and variance of residuals to GS_FS > rerun all
#todo : merge blending to other models


#todo : make this work for GS


