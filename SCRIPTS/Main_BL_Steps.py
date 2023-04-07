#DASHBOARD IMPORT
# from Dashboard_EUCB_FR_v2 import *
from Dashboard_EUCB_Structures import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

#SCRIPT IMPORTS
# from Model_Blending import *
from Model_Blending_CV import *
from HelpersFormatter import *
from HelpersArchiver import *
from Main_GS_FS_Steps import import_Main_GS_FS


def Run_Blending_NBest(modelList, displayParams, DBpath, ref_single, ConstructorKey ='LR_RIDGE'):

    #CONSTRUCT
    LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
    LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}
    SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
    SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
    LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
    CONSTRUCTOR_DICT = {'LR': LR_CONSTRUCTOR, 'LR_RIDGE' : LR_RIDGE_CONSTRUCTOR,
                        'SVR_RBF': SVR_RBF_CONSTRUCTOR, 'SVR_LIN': SVR_LIN_CONSTRUCTOR,
                        'LR_ELAST': LR_ELAST_CONSTRUCTOR}

    CONSTRUCTOR = CONSTRUCTOR_DICT[ConstructorKey]


    # CONSTRUCT & REPORT
    print('RUNNING BLENDING')
    blendModel = Model_Blender(modelList, CONSTRUCTOR, Gridsearch = True, Type='NBest')
    report_Blending_NBest(blendModel, displayParams, DBpath)
    pickleDumpMe(DBpath, displayParams, blendModel, 'BLENDER', blendModel.GSName)

    # LOAD
    blendModel = import_Blender_NBest(ref_single, label = ConstructorKey + '_Blender_NBest')


    # PLOT
    print('PLOTTING BLENDER')
    blendModel.plotBlenderYellowResiduals(displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                       yLim=PROCESS_VALUES['residualsYLim'], xLim=PROCESS_VALUES['residualsXLim'],
                                       studyFolder='BLENDER/')

    return blendModel


def assemble_blending(randomvalues, ref_prefix, GS_FS_List_Labels=['LR', 'LR_RIDGE', 'LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'],
                      single = False, predictor = 'SVR_RBF', ft_selector = 'RFE_GBR') :

    Blending_List = []

    if single :
    # SINGLE
        for value in randomvalues:
            ref_suffix_single = '_rd' + str(value) + '/'
            GS_FSs = import_Main_GS_FS(ref_prefix + ref_suffix_single, GS_FS_List_Labels=[predictor])
            Blending_List.append(GS_FSs[0].__getattribute__(ft_selector))

        Blending_Models = Blending_List
    else:
        # MULTIPLE
        for value in randomvalues:
            ref_suffix_single = '_rd' + str(value) + '/'
            GS_FSs = import_Main_GS_FS(ref_prefix + ref_suffix_single, GS_FS_List_Labels=GS_FS_List_Labels)
            # STORE
            Model_List = unpackGS_FSs(GS_FSs)
            Blending_List.append(Model_List)

        Blending_Models = repackGS_FSs(Blending_List)

    return Blending_Models




def Run_Blending_CV(displayParams, DBpath, ref_prefix, ConstructorKey = 'LR_RIDGE',
    GS_FS_List_Labels=['LR', 'LR_RIDGE','LR_LASSO', 'LR_ELAST', 'KRR_RBF', 'KRR_LIN', 'KRR_POL', 'SVR_LIN', 'SVR_RBF'],
    GS_name_list = ['LR_fl_spearman', 'LR_fl_pearson', 'LR_RFE_RFR', 'LR_RFE_DTR', 'LR_RFE_GBR', 'LR_NoSelector',
                    'LR_RIDGE_fl_spearman','LR_RIDGE_fl_pearson', 'LR_RIDGE_RFE_RFR', 'LR_RIDGE_RFE_DTR',
                    'LR_RIDGE_RFE_GBR', 'LR_RIDGE_NoSelector','LR_LASSO_fl_spearman', 'LR_LASSO_fl_pearson',
                    'LR_LASSO_RFE_RFR', 'LR_LASSO_RFE_DTR', 'LR_LASSO_RFE_GBR','LR_LASSO_NoSelector',
                    'LR_ELAST_fl_spearman', 'LR_ELAST_fl_pearson', 'LR_ELAST_RFE_RFR', 'LR_ELAST_RFE_DTR',
    'LR_ELAST_RFE_GBR', 'LR_ELAST_NoSelector', 'KRR_LIN_fl_spearman', 'KRR_LIN_fl_pearson', 'KRR_LIN_RFE_RFR',
    'KRR_LIN_RFE_DTR', 'KRR_LIN_RFE_GBR', 'KRR_LIN_NoSelector', 'KRR_RBF_fl_spearman', 'KRR_RBF_fl_pearson',
    'KRR_RBF_RFE_RFR', 'KRR_RBF_RFE_DTR', 'KRR_RBF_RFE_GBR', 'KRR_RBF_NoSelector', 'KRR_POL_fl_spearman',
    'KRR_POL_fl_pearson', 'KRR_POL_RFE_RFR', 'KRR_POL_RFE_DTR', 'KRR_POL_RFE_GBR', 'KRR_POL_NoSelector',
    'SVR_LIN_fl_spearman', 'SVR_LIN_fl_pearson', 'SVR_LIN_RFE_RFR', 'SVR_LIN_RFE_DTR', 'SVR_LIN_RFE_GBR',
    'SVR_LIN_NoSelector', 'SVR_RBF_fl_spearman', 'SVR_RBF_fl_pearson', 'SVR_RBF_RFE_RFR', 'SVR_RBF_RFE_DTR',
    'SVR_RBF_RFE_GBR', 'SVR_RBF_NoSelector'],single=False, predictor='SVR_RBF', ft_selector='RFE_GBR', runBlending = True):

    if runBlending :
        #CONSTRUCT

        LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
        LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}
        SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
        SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
        LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
        CONSTRUCTOR_DICT = {'LR': LR_CONSTRUCTOR, 'LR_RIDGE' : LR_RIDGE_CONSTRUCTOR,
                            'SVR_RBF': SVR_RBF_CONSTRUCTOR, 'SVR_LIN': SVR_LIN_CONSTRUCTOR,
                            'LR_ELAST': LR_ELAST_CONSTRUCTOR}

        CONSTRUCTOR = CONSTRUCTOR_DICT[ConstructorKey]

        # IMPORT MODELLIST
        Blending_Models = assemble_blending(randomvalues=studyParams['randomvalues'],ref_prefix = ref_prefix,
        GS_FS_List_Labels=GS_FS_List_Labels, single=single, predictor=predictor, ft_selector=ft_selector)

        # CONSTRUCT BLENDER & ARCHIVE
        for ModelList in Blending_Models:
            print('RUNNING BLENDING')
            blendModel = Model_Blender(ModelList, CONSTRUCTOR, Gridsearch = True, Type=ModelList[0].GSName)
            pickleDumpMe(DBpath, displayParams, blendModel, 'BLENDER', blendModel.GSName, combined=True)

    blender_name_list =[ConstructorKey + '_Blender_' + name for name in GS_name_list]

    # IMPORT BLENDER
    blendModels = []
    for name in blender_name_list : #change here
        blendModel = import_Blender_CV(name, ref_prefix)
        blendModels.append(blendModel)

    #REPORT
    report_Blender_CV(blendModels, displayParams, DBpath)

    # PLOT
    print('PLOTTING BLENDER')
    for blendModel in blendModels:
        blendModel.plotBlenderYellowResiduals(displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                           yLim=PROCESS_VALUES['residualsYLim'], xLim=PROCESS_VALUES['residualsXLim'],
                                           studyFolder='BLENDER/')
        blendModel.plot_Blender_CV_Residuals(displayParams, FORMAT_Values, DBpath)

    return blendModels


def import_Blender_CV(blendmodelName, ref_prefix, ref_suffix_combined = '_Combined/'):
    path = DB_Values['DBpath'] + 'RESULTS/' + ref_prefix + ref_suffix_combined + 'RECORDS/BLENDER/' + blendmodelName + '.pkl'
    Blender = pickleLoadMe(path=path, show=False)

    return Blender

def import_Blender_NBest(ref_single, label ='LR_RIDGE_Blender_NBest'):
    path = DB_Values['DBpath'] + 'RESULTS/' + ref_single + 'RECORDS/BLENDER/' + label + '.pkl'
    print(path)
    Blender = pickleLoadMe(path=path, show=False)

    return Blender


def report_Blender_CV(studies_Blender, displayParams, DBpath):

    import pandas as pd
    if displayParams['archive']:
        import os
        ref_prefix = displayParams['ref_prefix']
        path, folder, subFolder = DBpath, "RESULTS/", ref_prefix + '_Combined/' + 'RECORDS/'
        outputPathStudy = path + folder + subFolder

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        AllDfs = []
        sheetNames = []
        for blendModel in studies_Blender:

            BlendingDf = construct_Blending_Df(blendModel)
            AllDfs.append(BlendingDf)
            sheetNames.append(blendModel.GSName)

        columns = ['TrainScore', 'TestScore', 'TestMSE',  'TestAcc', 'ResidMean', 'ResidVariance','ModelWeights'] #'TestR2',

        BestBlenderDf, AvgBlenderDf, BestModel, AvgModel = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)
        BestBlender_BestModel, AvgBlender_AvgModel = pd.DataFrame(columns=columns), pd.DataFrame(columns=columns)


        for df, blendmodel in zip(AllDfs, studies_Blender):
            BestBlenderDf.loc[blendmodel.GSName, :] = df.loc[blendmodel.GSName, :]
            AvgBlenderDf.loc[blendmodel.GSName, :] = df.loc['Blender_Mean', :]
            BestModel.loc[blendmodel.GSName, :] = df.iloc[0, :]
            AvgModel.loc[blendmodel.GSName, :] = df.loc['NBest_Avg', :]
            BestBlender_BestModel.loc[blendmodel.GSName, :] = df.loc['BestBlender-BestModel', :]
            AvgBlender_AvgModel.loc[blendmodel.GSName, :] = df.loc['AvgBlender-NBestAvg', :]

        # summarize and sort
        SummaryDf = pd.concat([BestBlenderDf, AvgBlenderDf, BestModel, AvgModel, BestBlender_BestModel, AvgBlender_AvgModel],
                              axis = 1, keys = ['BestBlenderDf', 'AvgBlenderDf', 'BestModel', 'AvgModel', 'BestBlender-BestModel', 'AvgBlender-AvgModel'])
        sortedDf = SummaryDf.sort_values(('AvgBlender-AvgModel', 'ResidVariance'), ascending=True)
        sheetNames = ['SummaryDf'] + ['SortedSummaryDf'] + sheetNames
        dflist = [SummaryDf] + [sortedDf] + AllDfs

        with pd.ExcelWriter(outputPathStudy + ref_prefix + '_' + BLE_VALUES['Regressor'] + "_BL_Scores_CV" + ".xlsx", mode='w') as writer:
            for df, name in zip(dflist, sheetNames):
                df.to_excel(writer, sheet_name=name)

        return AllDfs, sheetNames








