
from Main_DA_Steps import *
from Main_FS_Steps import *
from Main_GS_FS_Steps import *
from Main_Combine_Steps import *


#1. Check db > no empty boxes, No data everywhere (vs unknown, No Data)

for set in studyParams['sets']:

    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
    displayParams["ref_prefix"] = acronym + '_' + studyParams['sets'][0][1]

    # # 0 ANALYZE
    print("Analyze Data")

    # # RUN
    Run_DA(path=DB_Values['DBpath'], dbName=DB_Values['DBname'], delimiter=DB_Values['DBdelimiter'],
           firstLine=DB_Values['DBfirstLine'], xQualLabels=xQualLabels, xQuantLabels=DAxQuantLabels, yLabels=DAyLabels,
           Summed_Labels=Summed_Labels, Divided_Labels=Divided_Labels, splittingFt=splittingFt, order=order, mainTarget=mainTarget,
           labels_1D=labels_1D, labels_2D_norm=labels_2D_norm, labels_2D_scale=labels_2D_scale,
           exploded_ft=exploded_ft, splittingFt_focus=splittingFt_focus, splittingFt_2=splittingFt_2)

    # IMPORT
    DA = import_DataAnalysis(displayParams["ref_prefix"], name = 'DataAnalysis' + splittingFt)

    # 1 SELECT DATA
    print("Select Data for :", set)

    # RUN
    Run_FS_CVStudy(cv=cv)

    # IMPORT
    rdat, dat, df, learningDf = import_input_data()

    #2 MODEL DATA

    All_CV, Filters_CV = [], []

    for i in range(1, cv+1):


        displayParams["reference"] = displayParams["ref_prefix"] + '_rd' + str(i) +'/'
        baseFormatedDf, filterList, RFEList = import_selected_data(displayParams["reference"], show = False)

        # RUN
        learning_dfs = [baseFormatedDf]  # not sure order counts
        if len(filterList) > 0:
            learning_dfs += filterList
        if len(RFEList) > 0:
            learning_dfs += RFEList

        # print('Fitting regression for fold : ', str(i))
        GS_FSs = Run_GS_FS(learning_dfs, regressors=studyParams['Regressors'])

        # IMPORT
        GS_FSs = import_Main_GS_FS(displayParams["reference"] , GS_FS_List_Labels = studyParams['Regressors'])

        report_GS_FS(displayParams, DB_Values, FORMAT_Values, PROCESS_VALUES, RFE_VALUES, GS_VALUES, BLE_VALUES,
                     rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList, GS_FSs)

        All_CV.append(GS_FSs)
        Filters_CV.append(filterList)

    #3 SELECT MODELS

    RUN_Training_Report(All_CV, Filters_CV, randomvalues=list(range(1, cv+1)), displayParams = displayParams,
                        studyParams = studyParams, GSName="All")



