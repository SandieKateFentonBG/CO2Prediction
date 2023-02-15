from Predict import *


def Run_Prediction_Individual( MyPred_Sample, ArchPath, Model_List=None, Blender_List=None, precomputed = False):

    sample = Sample(displayParams["reference"], MyPred_Sample)

    pickleDumpMe(ArchPath, displayParams, sample, 'PREDICTIONS', MyPred_Sample["DBname"])

    names, avgs, preds= [],[],[]

    if Model_List:

        for model in Model_List:
            # COMPUTE EXPLAINER

            if precomputed : #explainer already computed
                explainer =  model.SHAPexplainer
            else :
                import shap
                masker = shap.maskers.Independent(model.learningDf.XTrain)
                try:
                    explainer = shap.Explainer(model.Estimator)
                except Exception:
                    explainer = shap.KernelExplainer(model.Estimator.predict, model.learningDf.XTrain)

            # ARCHIVE
            names.append(model.GSName)
            avgs.append(explainer.expected_value)
            preds.append(sample.SamplePrediction(model))

            # PLOT
            sample.SHAP_WaterfallPlot(model, explainer, DB_Values['DBpath'])
            sample.SHAP_ScatterPlot(model, explainer, DB_Values['DBpath'])
            sample.SHAP_ForcePlot(model, explainer, DB_Values['DBpath'])

    if Blender_List :

        for model in Blender_List:
            # COMPUTE EXPLAINER

            if precomputed:  # explainer already computed
                explainer = model.SHAPexplainer
            else:
                import shap

                try:
                    explainer = shap.Explainer(model.Estimator)
                except Exception:
                    explainer = shap.KernelExplainer(model.Estimator.predict, model.blendXtrain)

            # ARCHIVE
            names.append(model.GSName)
            avgs.append(explainer.expected_value)
            preds.append(sample.SamplePredictionBlender(model))

            # PLOT

    SheetNames = ['Input', 'Predictions']

    #create empty dfs
    PredDf = pd.DataFrame(columns=['avg', 'pred'], index=names)
    PredDf.loc[:, 'avg'] = avgs
    PredDf.loc[:, 'pred'] = preds

    print(PredDf)

    AllDfs = [sample.input.T, PredDf]

    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + sample.name + '_Pred_Records' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, SheetNames):
                df.to_excel(writer, sheet_name=name)


def create_Gridsearch_Samples_1D(MyPred_Sample, feature1, feature2):
    from copy import copy, deepcopy

    sample = Sample(displayParams["reference"], MyPred_Sample)
    samples = []

    for i in range(len(sample.possibleQualities[feature1])) :

        newSample = copy(sample)
        value = newSample.possibleQualities[feature1][i]
        newxQuanti = newSample.xQuantiDict.copy()
        newxQuali = newSample.xQualiDict.copy()
        newxQuali[feature1] = [value]
        newSample.createSample(newxQuali, newxQuanti)
        samples.append(newSample)

    for s in samples:
        print("sample.xQuali", s.xQuali)

    return samples



def create_Gridsearch_Samples_2D(MyPred_Sample, feature1, feature2):
    from copy import copy, deepcopy

    sample = Sample(displayParams["reference"], MyPred_Sample)

    samples_ls = []

    for j in range(len(sample.possibleQualities[feature2])):
        samples = []
        for i in range(len(sample.possibleQualities[feature1])):
            newSample = copy(sample)
            value_j = newSample.possibleQualities[feature2][j]
            value_i = newSample.possibleQualities[feature1][i]
            newxQuanti = newSample.xQuantiDict.copy()
            newxQuali = newSample.xQualiDict.copy()

            newxQuali[feature2] = [value_j]
            newxQuali[feature1] = [value_i]

            newSample.createSample(newxQuali, newxQuanti)
            samples.append(newSample)

        samples_ls.append(samples)

    return samples_ls #list of sublists len(samples_ls) = feature2; len(samples_ls[0]) = feature1



def create_Gridsearch_Predictions_2D (MyPred_Sample, feature1, feature2, model):

    samples_ls = create_Gridsearch_Samples_2D(MyPred_Sample, feature1, feature2) #list of sublists len(samples_ls) = feature2; len(samples_ls[0]) = feature1
    cols = samples_ls[0][0].possibleQualities[feature1]
    idxs = samples_ls[0][0].possibleQualities[feature2]
    PredDf = pd.DataFrame(columns=cols, index=idxs)

    for sub_ls, name in zip(samples_ls, idxs) : #f2
        preds = []
        for elem in sub_ls : #f1

            if hasattr(model, 'modelList'):
                preds.append(elem.SamplePredictionBlender(model)[0])
            else:
                preds.append(elem.SamplePrediction(model)[0])

        PredDf.loc[name, :] = preds

    return PredDf
def Pred_HeatMap(PredDf):

    # fig = plt.figure()
    # fig, ax = plt.subplots()
    # im = ax.imshow(results)
    fig, ax = plt.subplots(figsize=(24,15))
    xticklabels = list(range(len(PredDf)))
    print(type(PredDf))
    print(PredDf)
    print(type(PredDf.values))
    print(PredDf.values)
    print(type(PredDf.values[0][0]))
    sns.heatmap(PredDf.round(2)) #, annot=True, cbar=True, cbar_kws={"shrink": .80},xticklabels = xticklabels,
                # fmt=".001f", ax=ax, cmap="bwr", center=0, vmin=-1, vmax=1, square=True)

def Run_Prediction_Gridsearch(MyPred_Sample, feature1, feature2, Model_List=None, Blender_List=None):

    AllDfs = []
    SheetNames = []
    for model in Model_List + Blender_List:
        SheetNames.append(model.GSName)
        PredDf = create_Gridsearch_Predictions_2D(MyPred_Sample, feature1, feature2, model)
        Pred_HeatMap(PredDf)



        AllDfs.append(PredDf)

    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + 'Prediction_Gridsearch' + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, SheetNames):
                df.to_excel(writer, sheet_name=name)










#REFERENCE

# for set in studyParams['sets']:
#     yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = set
#     DBname = DB_Values['acronym'] + '_' + yLabelsAc + '_rd'
#
#     for value in studyParams['randomvalues']:
#         PROCESS_VALUES['random_state'] = value

yLabels, yLabelsAc, BLE_VALUES['NBestScore'] = studyParams['sets'][0]
displayParams["reference"] = DB_Values['acronym'] + '_' + yLabelsAc + '_rd' + str(PROCESS_VALUES['random_state']) + '/'

print(displayParams["reference"])

# MODEL

GS_FSs = import_Main_GS_FS(displayParams["reference"], GS_FS_List_Labels = studyParams['Regressors'])
# Model_List_All = unpackGS_FSs(GS_FSs, remove='')
LRidge = [GS_FSs[1].RFE_RFR]
#
Blender = import_Main_Blender(displayParams["reference"], n = BLE_VALUES['NCount'], NBestScore = BLE_VALUES['NBestScore'], label = BLE_VALUES['Regressor'] + '_Blender')
B_M = Blender.modelList

# Run_Prediction(MyPred_Sample, DB_Values['DBpath'], Model_List=B_M + LRidge, Blender_List=[Blender], precomputed = False)

# create_Gridsearch_Samples_1D(MyPred_Sample, feature1='Structure', feature2='Main_Material')

# create_Gridsearch_Samples_2D(MyPred_Sample, feature1='Structure', feature2='Main_Material')


Run_Prediction_Gridsearch(MyPred_Sample, feature1='Structure', feature2='Main_Material', Model_List=LRidge, Blender_List=[Blender])