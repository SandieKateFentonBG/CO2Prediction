""""
# GOAL 
# RUN FS
# RUN GS
"""""

#DASHBOARD IMPORT
# from Dashboard_EUCB_FR_v2 import *
# from Dashboard_EUCB_Structures import *
from Dashboard_Current import *

#SCRIPT IMPORTS
from Main_GS_FS_Steps import *
from Main_Combine_Steps import *

#LIBRARY IMPORTS


Studies_CV_BlenderNBest = []

for set in studyParams['sets']:
    yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set

    print("Study for :", set)
    All_CV, NBest_CV, Filters_CV, Blenders_NBest_CV, Blenders_Single_CV = [],[],[],[],[]
    randomvalues = studyParams['randomvalues']

    for value in randomvalues:

        PROCESS_VALUES['random_state'] = value

        ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
        ref_suffix_single, ref_suffix_combined = '_rd'+ str(PROCESS_VALUES['random_state']) + '/', '_Combined/'
        ref_single, ref_combined = ref_prefix + ref_suffix_single, ref_prefix + ref_suffix_combined
        displayParams["ref_prefix"], displayParams["reference"] = ref_prefix, ref_single

        # ">>IMPORT"

        print('Import Study for random_state:', value)

        # FEATURE PROCESSING
        # rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList = import_Main_FS(ref_single, show = False)
        #
        # # MODEL PROCESSING
        # GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels = studyParams['Regressors'])
        #
        # # GS_FSs = import_Main_GS_FS(ref_single, GS_FS_List_Labels=['KRR_LIN'])

        # # ">>RUN"

        print('Run Study for random_state:', value)
        # #
        # # # FEATURE PROCESSING
        rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList = Run_FS_Study()

        # todo :  spearmanFilter, pearsonFilter was changed to filterList
        # #
        # # # MODEL PROCESSING
        GS_FSs = Run_GS_FS_Study(ref_single, importMainGSFS=False)

        # "STORE"

        All_CV.append(GS_FSs)

        Filters_CV.append(filterList)

        #
        # = filterList[i] for
        #
        # for name, filter in zip(studyParams['fl_selectors'], filterList):
        #     if s == 'pearson':
        #         Filters_Pearson_CV.append(pearsonFilter)
        #     if s == 'spearman'
        #         Filters_Spearman_CV.append(spearmanFilter)

    # "AVG & NBEST"

    RUN_Training_Report(All_CV, Filters_CV, randomvalues, displayParams, studyParams, GSName="All")



