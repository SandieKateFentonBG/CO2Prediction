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
        rdat.visualize(displayParams, DBpath=DB_Values['DBpath'], dbName=DB_Values['DBname'],
                       yLabel=yLabels[0], xLabel=xQualLabels[i], changeFigName=str(i))
    for i in range(len(xQuantLabels)):
        rdat.visualize(displayParams, DBpath=DB_Values['DBpath'], dbName=DB_Values['DBname'],
                       yLabel=yLabels[0], xLabel=xQuantLabels[i], changeFigName=xQuantLabels[i])

    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, rdat, 'DATA', 'rdat')

    return rdat

def B_features(rdat):
    """
    GOAL - Process data & One hot encoding
    """

    # CONSTRUCT
    dat = Features(rdat)
    df = dat.asDataframe()

    # REPORT
    print("Full df", df.shape)
    print(df)

    dfAsTable(DB_Values['DBpath'], displayParams, df, objFolder='DATA', name = "DF")

    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, dat, 'DATA', 'dat')
    pickleDumpMe(DB_Values['DBpath'], displayParams, df, 'DATA', 'df')

    return dat, df

def C_data(dat):
    """
    GOAL - Remove outliers - only exist/removed on Quantitative features
    Dashboard Input - PROCESS_VALUES : OutlierCutOffThreshhold
    """
    # CONSTRUCT

    if PROCESS_VALUES['removeUnderrepresenteds']:

        df, removedDict = dat.removeUnderrepresenteds(
            cutOffThreshhold=PROCESS_VALUES['UnderrepresentedCutOffThreshhold'],
            removeUnderrepresentedsFrom=PROCESS_VALUES['removeUnderrepresentedsFrom'])

        # REPORT
        print("Underrepresented removed", df.shape)
        print("Features removed:")
        for k, v in removedDict.items():
            print(k, v)

    else:
        df = dat.asDataframe()

    learningDf = removeOutliers(df, labels=PROCESS_VALUES['RemoveOutliersFrom'] + yLabels,
                                cutOffThreshhold=PROCESS_VALUES['OutlierCutOffThreshhold'])
    dfAsTable(DB_Values['DBpath'], displayParams, learningDf, objFolder='DATA', name = "learningDf")

    # REPORT
    print("Outliers removed ", learningDf.shape)

    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, learningDf, 'DATA', 'learningDf')


    return learningDf

def D_format(learningDf):
    """
    GOAL - Train Test Validate Check Split - Scale
    Dashboard Input - PROCESS_VALUES : test_size  # proportion with validation, random_state, yUnit
    """

    # CONSTRUCT
    baseFormatedDf = formatedDf(learningDf, xQuantLabels, xQualLabels, yLabels,
                                yUnitFactor=FORMAT_Values['yUnitFactor'], targetLabels=FORMAT_Values['targetLabels'],
                                random_state=PROCESS_VALUES['random_state'], test_size=PROCESS_VALUES['test_size'],
                                train_size=PROCESS_VALUES['train_size'], check_size=PROCESS_VALUES['check_size'],
                                val_size=PROCESS_VALUES['val_size'], fixed_seed = PROCESS_VALUES['fixed_seed'])

    dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.trainDf, objFolder='DATA', name = "trainDf")
    dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.checkDf, objFolder='DATA', name="checkDf")
    dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.valDf, objFolder='DATA', name = "valDf")
    dfAsTable(DB_Values['DBpath'], displayParams, baseFormatedDf.testDf, objFolder='DATA', name = "testDf")

    # STOCK
    pickleDumpMe(DB_Values['DBpath'], displayParams, baseFormatedDf, 'DATA', 'baseFormatedDf')

    return baseFormatedDf

def E_FS_Filter(baseFormatedDf):

    """
    GOAL - Remove uncorrelated and redundant features
    Dashboard Input - PROCESS_VALUES : corrMethod, corrRounding, corrLowThreshhold, corrHighThreshhold
    """
    """
    SPEARMAN
    """

    fl_selectors = []

    if studyParams['fl_selectors'] :

        for fl in studyParams['fl_selectors'] :

            # CONSTRUCT
            myFilter = FilterFeatures(baseFormatedDf, baseLabels=xQuantLabels, method=fl,
                                            corrRounding=PROCESS_VALUES['corrRounding'],
                                            lowThreshhold=PROCESS_VALUES['corrLowThreshhold'],
                                            highThreshhold=PROCESS_VALUES['corrHighThreshhold'])
            # STOCK
            pickleDumpMe(DB_Values['DBpath'], displayParams, myFilter, 'FS', fl)

            # VISUALIZE
            plotCorrelation(myFilter, myFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
                            filteringName="nofilter")
            plotCorrelation(myFilter, myFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
                            filteringName="dropuncorr")
            plotCorrelation(myFilter, myFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
                            filteringName="dropcolinear")

            fl_selectors.append(myFilter)

            # """
            # PEARSON
            # """
            # # CONSTRUCT
            # pearsonFilter = FilterFeatures(baseFormatedDf, baseLabels=xQuantLabels, method=PROCESS_VALUES['corrMethod2'],
            #                                corrRounding=PROCESS_VALUES['corrRounding'],
            #                                lowThreshhold=PROCESS_VALUES['corrLowThreshhold'],
            #                                highThreshhold=PROCESS_VALUES['corrHighThreshhold'])
            # # STOCK
            # pickleDumpMe(DB_Values['DBpath'], displayParams, pearsonFilter, 'FS', 'pearsonFilter')
            #
            # # VISUALIZE
            # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_All, DB_Values['DBpath'], displayParams,
            #                 filteringName="nofilter")
            # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_NoUncorrelated, DB_Values['DBpath'], displayParams,
            #                 filteringName="dropuncorr")
            # plotCorrelation(pearsonFilter, pearsonFilter.correlationMatrix_NoRedundant, DB_Values['DBpath'], displayParams,
            #                 filteringName="dropcolinear")

    return fl_selectors

def F_FS_RFE(baseFormatedDf):
    """
    GOAL - select the optimal number of features or combination of features
    """

    RFEList = []

    if studyParams['RFE_selectors']:

        rfe_hyp_feature_count = list(np.arange(10, len(baseFormatedDf.XVal) - 10, 10))

        # for RFE in studyParams['RFE_selectors'] :

        # CONSTRUCT
        random_state = PROCESS_VALUES['fixed_seed']
        RFR_RFE_CONSTRUCTOR = {'method' : 'RFR', 'estimator' : RandomForestRegressor(random_state=random_state) }
        DTR_RFE_CONSTRUCTOR = {'method' : 'DTR', 'estimator' : DecisionTreeRegressor(random_state=random_state)}
        GBR_RFE_CONSTRUCTOR = {'method' : 'GBR', 'estimator' : GradientBoostingRegressor(random_state=random_state)}

        All_CONSTRUCTOR = [RFR_RFE_CONSTRUCTOR, DTR_RFE_CONSTRUCTOR, GBR_RFE_CONSTRUCTOR]
        RFE_CONSTRUCTOR = [elem for elem in All_CONSTRUCTOR if elem['method'] in studyParams['RFE_selectors']]

        for constructor in RFE_CONSTRUCTOR:

            MyRFE = WrapFeatures(method=constructor['method'], estimator=constructor['estimator'], formatedDf=baseFormatedDf,
                                 rfe_hyp_feature_count=rfe_hyp_feature_count, min_features_to_select = RFE_VALUES['RFE_n_features_to_select'],
                                 output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
                                 cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))

            pickleDumpMe(DB_Values['DBpath'], displayParams, MyRFE, 'FS', constructor['method'])

            RFEList.append(MyRFE)
            #
            #
            # RFR_RFE = WrapFeatures(method=RFE, estimator=RandomForestRegressor(random_state=PROCESS_VALUES['fixed_seed']),
            #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
            #                        min_features_to_select = RFE_VALUES['RFE_n_features_to_select'],
            #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
            #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
            # DTR_RFE = WrapFeatures(method='DTR', estimator=DecisionTreeRegressor(random_state=PROCESS_VALUES['fixed_seed']),
            #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
            #                        min_features_to_select=RFE_VALUES['RFE_n_features_to_select'],
            #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
            #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))
            # GBR_RFE = WrapFeatures(method='GBR',
            #                        estimator=GradientBoostingRegressor(random_state=PROCESS_VALUES['fixed_seed']),
            #                        formatedDf=baseFormatedDf, rfe_hyp_feature_count=rfe_hyp_feature_count,
            #                        min_features_to_select=RFE_VALUES['RFE_n_features_to_select'],
            #                        output_feature_count=RFE_VALUES['output_feature_count'], process=RFE_VALUES['RFE_process'],
            #                        cv = KFold(n_splits=5, shuffle=True, random_state= PROCESS_VALUES['fixed_seed']))

            # RFEs = [RFR_RFE, DTR_RFE, GBR_RFE]

        # STOCK
        pickleDumpMe(DB_Values['DBpath'], displayParams, RFEList, 'FS', 'RFEs')

        # REPORT
        reportRFE(DB_Values['DBpath'], displayParams, RFEList, objFolder='FS', display=True, process=RFE_VALUES['RFE_process'])

        #VISUALIZE
        if RFE_VALUES['RFE_process'] == 'long':

            RFEHyperparameterPlot2D(RFEList,  displayParams, DBpath = DB_Values['DBpath'], yLim = None, figTitle = 'RFEPlot2d',
                                      title ='Influence of Feature Count on Model Performance', xlabel='Feature Count', log = False)

            RFEHyperparameterPlot3D(RFEList, displayParams, DBpath = DB_Values['DBpath'], figTitle='RFEPlot3d',
                                        colorsPtsLsBest=['b', 'g', 'c', 'y'],
                                        title='Influence of Feature Count on Model Performance', ylabel='Feature Count',
                                        zlabel='R2 Test score', size=[6, 6],
                                        showgrid=False, log=False, max=True, ticks=False, lims=False)
    return RFEList

def Run_Data_Processing():
    rdat = A_RawData()
    dat, df = B_features(rdat)
    learningDf = C_data(dat)
    baseFormatedDf = D_format(learningDf)

    return rdat, dat, df, learningDf, baseFormatedDf

def Run_FS_Study():

    rdat, dat, df, learningDf, baseFormatedDf = Run_Data_Processing()
    print(baseFormatedDf.yLabel)

    filterList = E_FS_Filter(baseFormatedDf)

    RFEList = F_FS_RFE(baseFormatedDf)

    reportProcessing(DB_Values['DBpath'], displayParams, df, learningDf, baseFormatedDf,
                     filterList, RFEList)
    # todo : 'dat' was added to all functions

    return rdat, dat, df, learningDf, baseFormatedDf, filterList, RFEList

    # todo :  spearmanFilter, pearsonFilter was changed to filterList

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