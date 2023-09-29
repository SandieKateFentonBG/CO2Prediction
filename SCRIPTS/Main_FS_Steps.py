# todo : choose database - untoogle it and untoggle import line
#DASHBOARD IMPORT
# from dashBoard import *
# from Dashboard_PM_v2 import *
# from Dashboard_EUCB_FR import *
# from Dashboard_EUCB_FR_v2 import *
# from Dashboard_EUCB_Structures import *
from Dashboard_Current import *

#SCRIPT IMPORTS
from HelpersArchiver import *
from Raw import *
from Features import *
from Data import *
from Format import *
from Split import *
from Filter import *
from FilterVisualizer import *
from Wrapper import *
from WrapperReport import *
from WrapperVisualizer import *
from ProcessingReport import *
from Model import *
from ModelPredTruthPt import *
from ModelResidualsPt import *
from ModelParamPt import *
from ModelMetricsPt import *
from ModelWeightsPt import *
from Gridsearch import *

#LIBRARY IMPORTS
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
BASE
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""



def A_RawData():
    """
    GOAL - Import libraries & Load data
    """

    # CONSTRUCT
    rdat = RawData(path=DB_Values['DBpath'], dbName=DB_Values['DBname'], delimiter=DB_Values['DBdelimiter'],
                   firstLine=DB_Values['DBfirstLine'], xQualLabels=xQualLabels, xQuantLabels=xQuantLabels,
                   yLabels=yLabels, updateLabels=None)

    # VISUALIZE
    for i in range(len(xQualLabels)):
        rdat.visualize(displayParams, DBpath=DB_Values['DBpath'], combined = True,
                       yLabel=yLabels[0], xLabel=xQualLabels[i], changeFigName=str(i))
    for i in range(len(xQuantLabels)):
        rdat.visualize(displayParams, DBpath=DB_Values['DBpath'], combined = True,
                       yLabel=yLabels[0], xLabel=xQuantLabels[i], changeFigName=xQuantLabels[i])

    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, rdat, 'DATA', 'rdat', combined = True)

    return rdat

def B_encodeFeatures(rdat):
    """
    GOAL - Process data & One hot encoding
    """

    # CONSTRUCT
    dat = Features(rdat)
    df = dat.asDataframe()

    # shuffle the DataFrame rows and reindex them #todo this was changed
    df = df.sample(frac=1).reset_index(drop=True)

    # REPORT
    dfAsTable(DB_Values['DBpath'], displayParams, df, objFolder='DATA', name = "DF", combined = True)

    # STOCK
    # pickleDumpMe(DB_Values['DBpath'], displayParams, dat, 'DATA', 'dat', combined = True)
    pickleDumpMe(DB_Values['DBpath'], displayParams, df, 'DATA', 'df', combined = True)

    return dat, df

def C_cleanData(dat):
    """
    GOAL - Remove outliers - on Quantitative features and Underrepresented on qualitative
    Dashboard Input - PROCESS_VALUES : OutlierCutOffThreshhold
    """
    # CONSTRUCT #todo : this part was changed

    df = dat.asDataframe()

    nouOutlDf = removeOutliers(df, labels=PROCESS_VALUES['RemoveOutliersFrom'] + yLabels,
                                cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])
    # REPORT
    print("Outliers removed ", nouOutlDf.shape)

    if PROCESS_VALUES['removeUnderrepresenteds']:

        learningDf, removedDict, removed_list = dat.removeUnderrepresenteds(df =nouOutlDf,
            cutOffThreshhold=PROCESS_VALUES['UnderrepresentedCutOffThreshhold'],
            removeUnderrepresentedsFrom=PROCESS_VALUES['removeUnderrepresentedsFrom'])

        # REPORT
        print("Underrepresented removed", learningDf.shape)
        print("Features removed:")

        PROCESS_VALUES['removeUnderrepresentedsDict'] = removed_list

    else:
         learningDf = nouOutlDf

    dfAsTable(DB_Values['DBpath'], displayParams, learningDf, objFolder='DATA', name = "learningDf", combined = True)


    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, dat, 'DATA', 'dat', combined = True) #overwrite previous dat
    pickleDumpMe(DB_Values['DBpath'], displayParams, learningDf, 'DATA', 'learningDf', combined = True)

    return learningDf

"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
V1 - MAYBE OLD
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

# def D_format(learningDf, combined):
#     """
#     GOAL - Train Test Validate Check Split - Scale
#     Dashboard Input - PROCESS_VALUES : test_size  # proportion with validation, random_state, yUnit
#     """
#
#     # CONSTRUCT
#     baseFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
#                                 yUnitFactor=FORMAT_Values['yUnitFactor'], targetLabels=FORMAT_Values['targetLabels'],
#                                 random_state=PROCESS_VALUES['random_state'], test_size=PROCESS_VALUES['test_size'],
#                                 train_size=PROCESS_VALUES['train_size'], check_size=PROCESS_VALUES['check_size'],
#                                 val_size=PROCESS_VALUES['val_size'], fixed_seed = PROCESS_VALUES['fixed_seed'])
#
#     #todo : migration of selection to combined : combined = PROCESS_VALUES['selectionStoredinCombined']
#
#     dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.trainDf, objFolder='DATA', name = "trainDf",
#               combined = combined)
#     dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.checkDf, objFolder='DATA', name="checkDf", combined = combined)
#     dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.valDf, objFolder='DATA', name = "valDf", combined = combined)
#     dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.testDf, objFolder='DATA', name = "testDf", combined = combined)
#
#     # STOCK
#     pickleDumpMe(DB_Values['DBpath'], displayParams, baseFormatedDf, 'DATA', 'baseFormatedDf', combined = combined)
#
#     return baseFormatedDf
#
# def E_FS_Filter(baseFormatedDf, combined):
#
#     """
#     GOAL - Remove uncorrelated and redundant features
#     Dashboard Input - PROCESS_VALUES : corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold
#     """
#     """
#     SPEARMAN
#     """
#
#     fl_selectors = []
#
#     if studyParams['fl_selectors'] :
#
#         for fl in studyParams['fl_selectors'] :
#
#             # CONSTRUCT
#             myFilter = FilterFeatures(baseFormatedDf, baseLabels=xQuantLabels, method=fl,
#                                             corrRounding=PROCESS_VALUES['corrRounding'],
#                                             lowThreshhold=PROCESS_VALUES['corrLowThreshhold'],
#                                             highThreshhold=PROCESS_VALUES['corrHighThreshhold'])
#             # todo : migration of selection to combined : combined = PROCESS_VALUES['selectionStoredinCombined']
#
#             # STOCK
#             pickleDumpMe(DB_Values['DBpath'], displayParams, myFilter, 'FS', fl, combined = combined)
#
#             # VISUALIZE
#             plotCorrelation(myFilter, myFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
#                             filteringName="nofilter", combined = combined)
#             plotCorrelation(myFilter, myFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
#                             filteringName="dropuncorr", combined = combined)
#             plotCorrelation(myFilter, myFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
#                             filteringName="dropcolinear", combined = combined)
#
#             fl_selectors.append(myFilter)
#
#             # """
#             # PEARSON
#             # """
#             # # CONSTRUCT
#             # pearsonFilter = FilterFeatures(baseFormatedDf, baseLabels=xQuantLabels, method=PROCESS_VALUES['corrMethod2'],
#             #                                corrRounding=PROCESS_VALUES['corrRounding'],
#             #                                lowThreshhold=PROCESS_VALUES['corrLowThreshhold'],
#             #                                highThreshhold=PROCESS_VALUES['corrHighThreshhold'])
#             # # STOCK
#             # pickleDumpMe(DB_Values['DBpath'], displayParams, pearsonFilter, 'FS', 'pearsonFilter')
#             #
#             # # VISUALIZE
#             # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
#             #                 filteringName="nofilter")
#             # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
#             #                 filteringName="dropuncorr")
#             # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
#             #                 filteringName="dropcolinear")
#
#     return fl_selectors
#
# def F_FS_RFE(baseFormatedDf, combined):
#     """
#     GOAL - select the optimal number of features or combination of features
#     """
#
#     RFEList = []
#
#     if studyParams['RFE_selectors']:
#
#         rfe_hyp_feature_count = list(np.arange(10, len(baseFormatedDf.XVal) - 10, 10))
#
#         # for RFE in studyParams['RFE_selectors'] :
#
#         # CONSTRUCT
#         random_state = PROCESS_VALUES['fixed_seed']
#         RFR_RFE_CONSTRUCTOR = {'method' : 'RFR', 'estimator' : RandomForestRegressor(random_state=random_state) }
#         DTR_RFE_CONSTRUCTOR = {'method' : 'DTR', 'estimator' : DecisionTreeRegressor(random_state=random_state)}
#         GBR_RFE_CONSTRUCTOR = {'method' : 'GBR', 'estimator' : GradientBoostingRegressor(random_state=random_state)}
#
#         All_CONSTRUCTOR = [RFR_RFE_CONSTRUCTOR, DTR_RFE_CONSTRUCTOR, GBR_RFE_CONSTRUCTOR]
#         RFE_CONSTRUCTOR = [elem for elem in All_CONSTRUCTOR if elem['method'] in studyParams['RFE_selectors']]
#
#         for constructor in RFE_CONSTRUCTOR:
#
#             MyRFE = WrapFeatures(method=constructor['method'], estimator=constructor['estimator'], formatedDf=baseFormatedDf,
#                                  rfe_hyp_feature_count=rfe_hyp_feature_count, min_features_to_select = RFE_VALUES['RFE_n_features_to_select'],
#                                  output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
#                                  cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
#
#             pickleDumpMe(DB_Values['DBpath'], displayParams, MyRFE, 'FS', constructor['method'], combined = combined)
#
#             RFEList.append(MyRFE)
#             #
#             #
#             # RFR_RFE = WrapFeatures(method=RFE, estimator=RandomForestRegressor(random_state=PROCESS_VALUES['fixed_seed']),
#             #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
#             #                        min_features_to_select = RFE_VALUES['RFE_n_features_to_select'],
#             #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
#             #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
#             # DTR_RFE = WrapFeatures(method='DTR', estimator=DecisionTreeRegressor(random_state=PROCESS_VALUES['fixed_seed']),
#             #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
#             #                        min_features_to_select=RFE_VALUES['RFE_n_features_to_select'],
#             #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
#             #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
#             # GBR_RFE = WrapFeatures(method='GBR',
#             #                        estimator=GradientBoostingRegressor(random_state=PROCESS_VALUES['fixed_seed']),
#             #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
#             #                        min_features_to_select=RFE_VALUES['RFE_n_features_to_select'],
#             #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
#             #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
#
#             # RFEs = [RFR_RFE, DTR_RFE, GBR_RFE]
#
#         # STOCK
#         pickleDumpMe(DB_Values['DBpath'], displayParams, RFEList, 'FS', 'RFEs', combined = combined)
#
#         # REPORT
#         reportRFE(DB_Values['DBpath'], displayParams, RFEList, objFolder='FS', display=True, process=RFE_VALUES['RFE_process'], combined = combined)
#
#         #VISUALIZE
#         if RFE_VALUES['RFE_process'] == 'long':
#
#             RFEHyperparameterPlot2D(RFEList,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
#                                       title ='Influence of Feature Count on Model Performance', xlabel='Feature Count',
#                                     log = False, combined = combined)
#
#             RFEHyperparameterPlot3D(RFEList, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
#                                         colorsPtsLsBest=['b', 'g', 'c', 'y'],
#                                         title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
#                                         zlabel='R2 Test score', size=[6, 6],
#                                         showgrid=False, log=False, max=True, ticks=False, lims=False, combined = combined)
#     return RFEList
#
# def Run_Data_Processing(combined):
#     rdat = A_RawData(combined=combined)
#     dat, df = B_encodeFeatures(rdat, combined=combined)
#     learningDf = C_cleanData(dat, combined=combined)
#     baseFormatedDf = D_format(learningDf, combined=combined)
#
#     return rdat, dat, df, learningDf, baseFormatedDf
#
# def Run_FS_Study(combined):
#
#     rdat, dat, df, learningDf, baseFormatedDf = Run_Data_Processing(combined=combined)
#
#     filterList = E_FS_Filter(baseFormatedDf, combined=combined)
#
#     RFEList = F_FS_RFE(baseFormatedDf, combined=combined)
#
#     reportProcessing(DB_Values['DBpath'], displayParams, df, learningDf, baseFormatedDf,
#                      filterList, RFEList, combined=combined)
#     # todo : 'dat' was added to all functions
#
#     return rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList
#
#     # todo :  spearmanFilter, pearsonFilter was changed to filterList


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
V2
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""


def D_processData(learningDf, cv):

    """
    GOAL - Train Test Validate Check Split - Scale with CV
    Dashboard Input - PROCESS_VALUES : test_size  # proportion with validation, random_state, yUnit
    """

    # FIXED SELECTION SPLIT
    splitDf = SplitDf(learningDf, xQuantLabels, xQualLabels, yLabels,
                      yUnitFactor=FORMAT_Values['yUnitFactor'], targetLabels=FORMAT_Values['targetLabels'],
                      random_state=PROCESS_VALUES['random_state'], test_size=PROCESS_VALUES['test_size'],
                      train_size=PROCESS_VALUES['train_size'], check_size=PROCESS_VALUES['check_size'],
                      val_size=PROCESS_VALUES['val_size'], fixed_seed=PROCESS_VALUES['fixed_seed'],
                      removeUnderrepresenteds=PROCESS_VALUES['removeUnderrepresenteds'],
                      removeUnderrepresentedsDict=PROCESS_VALUES['removeUnderrepresentedsDict'])

    kfolds = splitDf.split_cv(X=splitDf.XR, y=splitDf.yR, k=cv)

    # kfolds = [fold,..., fold, fold]
    # fold =  [X_train,X_test, y_train, y_test]

    # CV REGRESSION SPLIT
    baseFormatedDfs = []
    for i in range(len(kfolds)):

        baseFormatedDf = CrossValDf(splitDf, kfolds[i], i)

        dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.trainDf, objFolder='DATA', name="trainDf",
                  combined=False, number=baseFormatedDf.random_state)
        dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.checkDf, objFolder='DATA', name="checkDf",
                  combined=False, number=baseFormatedDf.random_state)
        dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.valDf, objFolder='DATA', name="valDf",
                  combined=False, number=baseFormatedDf.random_state)
        dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.testDf, objFolder='DATA', name="testDf",
                  combined=False, number=baseFormatedDf.random_state)

        # STOCK
        pickleDumpMe(DB_Values['DBpath'], displayParams, baseFormatedDf, 'DATA', 'baseFormatedDf', combined=False
                     , number=baseFormatedDf.random_state) # baseFormatedDfs are different for every cv !!
        baseFormatedDfs.append(baseFormatedDf)

    return splitDf, baseFormatedDfs

def E_FS_FilterData(splitDf):

    """
    GOAL - Remove uncorrelated and redundant features
    Dashboard Input - PROCESS_VALUES : corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold
    """
    """
    SPEARMAN
    """

    fl_selectorsData = []

    if studyParams['fl_selectors'] :

        for fl in studyParams['fl_selectors'] :

            # CONSTRUCT
            myFilterData = FilterFeaturesCV(splitDf, baseLabels=xQuantLabels, method=fl,
                                            corrRounding=PROCESS_VALUES['corrRounding'],
                                            lowThreshhold=PROCESS_VALUES['corrLowThreshhold'],
                                            highThreshhold=PROCESS_VALUES['corrHighThreshhold'])

            # VISUALIZE
            plotCorrelation(myFilterData, myFilterData.correlationMatrix_All, DB_Values['DBpath'], displayParams,
                            filteringName="nofilter", combined = True)
            plotCorrelation(myFilterData, myFilterData.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
                            filteringName="dropuncorr", combined = True)
            plotCorrelation(myFilterData, myFilterData.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
                            filteringName="dropcolinear", combined = True)

            fl_selectorsData.append(myFilterData)

    return fl_selectorsData

def F_FS_RFEData(splitDf):
    """
    GOAL - select the optimal number of features or combination of features
    """

    RFEData = []

    if studyParams['RFE_selectors']:

        rfe_hyp_feature_count = list(np.arange(10, len(splitDf.XVal) - 10, 10))

        # for RFE in studyParams['RFE_selectors'] :

        # CONSTRUCT
        random_state = PROCESS_VALUES['fixed_seed']
        RFR_RFE_CONSTRUCTOR = {'method' : 'RFR', 'estimator' : RandomForestRegressor(random_state=random_state) }
        DTR_RFE_CONSTRUCTOR = {'method' : 'DTR', 'estimator' : DecisionTreeRegressor(random_state=random_state)}
        GBR_RFE_CONSTRUCTOR = {'method' : 'GBR', 'estimator' : GradientBoostingRegressor(random_state=random_state)}

        All_CONSTRUCTOR = [RFR_RFE_CONSTRUCTOR, DTR_RFE_CONSTRUCTOR, GBR_RFE_CONSTRUCTOR]
        RFE_CONSTRUCTOR = [elem for elem in All_CONSTRUCTOR if elem['method'] in studyParams['RFE_selectors']]

        for constructor in RFE_CONSTRUCTOR:

            MyRFE = WrapFeaturesCV(method=constructor['method'], estimator=constructor['estimator'], splitDf=splitDf,
                                 rfe_hyp_feature_count=rfe_hyp_feature_count, min_features_to_select = RFE_VALUES['RFE_n_features_to_select'],
                                 output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
                                 cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))


            RFEData.append(MyRFE)


        # REPORT
        reportRFE(DB_Values['DBpath'], displayParams, RFEData, objFolder='FS', display=True, process=RFE_VALUES['RFE_process'], combined = True)

        #VISUALIZE
        if RFE_VALUES['RFE_process'] == 'long':

            RFEHyperparameterPlot2D(RFEData,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
                                      title ='Influence of Feature Count on Model Performance', xlabel='Feature Count',
                                    log = False, combined = True)

            RFEHyperparameterPlot3D(RFEData, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
                                        colorsPtsLsBest=['b', 'g', 'c', 'y'],
                                        title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
                                        zlabel='R2 Test score', size=[6, 6],
                                        showgrid=False, log=False, max=True, ticks=False, lims=False, combined = True)
    return RFEData

def Run_FS_CVStudy(cv):

    # ! Split Df > fixed split > same for all random seeds
    # != BaseformattedDf > additional Xtrain and Xtest cv split different for every random seed
    print('Process data')
    rdat = A_RawData()
    dat, df = B_encodeFeatures(rdat)

    learningDf = C_cleanData(dat)
    print('learningDf', learningDf)

    print('Split data into ', str(cv), 'folds')

    splitDf, baseFormatedDfs = D_processData(learningDf, cv=cv)  # this is a list of len cv

    print('Filter features')
    fl_selectorsData = E_FS_FilterData(splitDf)
    print('RFE of features')
    RFEData = F_FS_RFEData(splitDf)

    for baseFormatedDf in baseFormatedDfs:
        filterList = []
        RFEList = []

        if studyParams['fl_selectors']:

            for fl_name, fl in zip(studyParams['fl_selectors'], fl_selectorsData):

                # FILTER OUT
                fl.updateFilterCV(baseFormatedDf)
                # STOCK
                pickleDumpMe(DB_Values['DBpath'], displayParams, fl, 'FS', fl_name, combined=False, number=baseFormatedDf.random_state)
                filterList.append(fl)

        if studyParams['RFE_selectors']:

            for RFE_name, RFE in zip(studyParams['RFE_selectors'], RFEData):
                # FILTER OUT
                RFE.updateWrapFeaturesCV(baseFormatedDf)
                # STOCK
                pickleDumpMe(DB_Values['DBpath'], displayParams, RFE, 'FS', RFE.method, combined=False, number=baseFormatedDf.random_state)
                RFEList.append(RFE)
            pickleDumpMe(DB_Values['DBpath'], displayParams, RFEList, 'FS', 'RFEs', combined=False, number=baseFormatedDf.random_state)
        # REPORT
        reportProcessing(DB_Values['DBpath'], displayParams, df, learningDf, baseFormatedDf, filterList, RFEList, combined=False, number=baseFormatedDf.random_state)

def import_input_data(show = False):

    import_reference = displayParams['ref_prefix'] + '_Combined/'

    rdat = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/rdat.pkl', show = show)
    dat = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/DATA/dat.pkl', show=show)
    df = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/df.pkl', show = show)
    learningDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/learningDf.pkl', show = show)

    return rdat, dat, df, learningDf

def import_selected_data(import_reference, show = False):


    baseFormatedDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/baseFormatedDf.pkl', show = show)

    filterList = []
    RFEList = []
    if studyParams['fl_selectors']:

        for s in studyParams['fl_selectors']:
            myFilter = pickleLoadMe(
                path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/' + s + '.pkl', show=show)
            filterList.append(myFilter)

    if studyParams['RFE_selectors']:
        for s in studyParams['RFE_selectors']:
            myRFE = pickleLoadMe(
                path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/' + s + '.pkl', show=show)
            RFEList.append(myRFE)

    return baseFormatedDf, filterList, RFEList


"""
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
V2
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
"""

def import_Processed_Data(import_reference, show = False):
    rdat = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/rdat.pkl', show = show)
    try:
        dat = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/DATA/dat.pkl', show=show)
    except Exception:
        dat = None #if the training was made before this existed
    df = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/df.pkl', show = show)
    learningDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/learningDf.pkl', show = show)
    baseFormatedDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/baseFormatedDf.pkl', show = show)

    return rdat, dat, df, learningDf, baseFormatedDf


def import_Processed_Data(import_reference, show = False):
    rdat = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/rdat.pkl', show = show)
    try:
        dat = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/DATA/dat.pkl', show=show)
    except Exception:
        dat = None #if the training was made before this existed
    df = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/df.pkl', show = show)
    learningDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/learningDf.pkl', show = show)
    baseFormatedDf = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/DATA/baseFormatedDf.pkl', show = show)

    return rdat, dat, df, learningDf, baseFormatedDf

def import_Main_FS(import_reference, show = False):

    # #IMPORT
    rdat, dat, df, learningDf, baseFormatedDf = import_Processed_Data(import_reference, show = False)

    filterList = []
    RFEList = []
    if studyParams['fl_selectors']:

        for s in studyParams['fl_selectors']:
            try:
                myFilter = pickleLoadMe(
                    path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/' + s + '.pkl', show=show)
            except Exception:  # if the training was made before this existed
                myFilter = pickleLoadMe(
                    path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/' + s + 'Filter.pkl',
                    show=show)
            filterList.append(myFilter)

    if studyParams['RFE_selectors']:
        try:
            for s in studyParams['RFE_selectors']:
                myRFE = pickleLoadMe(
                    path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/' + s + '.pkl', show=show)
                RFEList.append(myRFE)

        except Exception:  # if the training was made before this existed
            RFEList = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/FS/RFEs.pkl', show = show)
    #path = 'C:/Users/sfenton/Code/Repositories/CO2Prediction/RESULTS/' + import_reference + 'RECORDS/FS/RFEs.pkl'

    # try:
    #     spearmanFilter = pickleLoadMe(
    #         path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/spearman.pkl', show=show)
    # except Exception: #if the training was made before this existed
    #     spearmanFilter = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/FS/spearmanFilter.pkl', show = show)
    # try:
    #     pearsonFilter = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/FS/pearson.pkl', show = show)
    # except Exception: #if the training was made before this existed
    #     pearsonFilter = pickleLoadMe(path = DB_Values['DBpath'] + 'RESULTS/'+ import_reference +'RECORDS/FS/pearsonFilter.pkl', show = show)
    # RFEList = pickleLoadMe(path=DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/FS/RFEs.pkl',
    #                        show=show)

    # todo :  spearmanFilter, pearsonFilter was changed to filterList

    return rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList