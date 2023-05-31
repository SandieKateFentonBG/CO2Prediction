from Sample import *

def Run_Model_Predictions_Explainer(MyPred_Sample, ArchPath, Model_List=None, Blender_List=None, precomputed = False, dbName=MyPred_Sample['DBname']):

    sample = Sample(displayParams["reference"], MyPred_Sample, dbName)

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

            sample.SHAP_WaterfallPlot(model, explainer, DB_Values['DBpath'], Grouped = True)
            sample.SHAP_ScatterPlot(model, explainer, DB_Values['DBpath'])
            sample.SHAP_ForcePlot(model, explainer, DB_Values['DBpath'], Grouped = True)


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

    AllDfs = [sample.input.T, PredDf]

    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + 'Model_Predictions_Explainer_' + sample.name + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, SheetNames):
                df.to_excel(writer, sheet_name=name)

def create_Feature_Samples_1D(MyPred_Sample, feature1, feature2):
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

def create_Feature_Samples_2D(MyPred_Sample, feature1, feature2, feature1_values = None, feature2_values = None):
    from copy import copy, deepcopy

    sample = Sample(displayParams["reference"], MyPred_Sample)

    samples_ls = []

    if feature2 in sample.xQuali.keys() and feature1 in sample.xQuali.keys():

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

    if feature2 in sample.xQuali.keys() and feature1 in sample.xQuanti.keys():

        for j in range(len(sample.possibleQualities[feature2])):
            samples = []
            for i in range(len(feature1_values)):
                newSample = copy(sample)
                newxQuanti = newSample.xQuantiDict.copy()
                newxQuali = newSample.xQualiDict.copy()

                newxQuali[feature2] = [newSample.possibleQualities[feature2][j]]
                newxQuanti[feature1] = [feature1_values[i]]

                newSample.createSample(newxQuali, newxQuanti)
                samples.append(newSample)

            samples_ls.append(samples)


    if feature2 in sample.xQuanti.keys() and feature1 in sample.xQuali.keys():

        for j in range(len(feature2_values)):
            samples = []
            for i in range(len(sample.possibleQualities[feature1])):
                newSample = copy(sample)
                newxQuanti = newSample.xQuantiDict.copy()
                newxQuali = newSample.xQualiDict.copy()

                newxQuanti[feature2] = [feature2_values[j]]
                newxQuali[feature1] = [newSample.possibleQualities[feature1][i]]

                newSample.createSample(newxQuali, newxQuanti)
                samples.append(newSample)

            samples_ls.append(samples)


    if feature2 in sample.xQuanti.keys() and feature1 in sample.xQuanti.keys():

        for j in range(len(feature2_values)):
            samples = []
            for i in range(len(feature1_values)):
                newSample = copy(sample)
                newxQuanti = newSample.xQuantiDict.copy()
                newxQuali = newSample.xQualiDict.copy()

                newxQuanti[feature2] = [feature2_values[j]]
                newxQuanti[feature1] = [feature1_values[i]]

                newSample.createSample(newxQuali, newxQuanti)
                samples.append(newSample)

            samples_ls.append(samples)



    return samples_ls #list of sublists len(samples_ls) = feature2; len(samples_ls[0]) = feature1

def create_Feature_Predictions_2D (MyPred_Sample, feature1, feature2, model, feature1_values = None, feature2_values = None):

    samples_ls = create_Feature_Samples_2D(MyPred_Sample, feature1, feature2, feature1_values = feature1_values, feature2_values = feature2_values) #list of sublists len(samples_ls) = feature2; len(samples_ls[0]) = feature1

    # consider 4 options quali - quanti :

    if feature2 in samples_ls[0][0].xQuali.keys() and feature1 in samples_ls[0][0].xQuali.keys():

        cols = samples_ls[0][0].possibleQualities[feature1]
        idxs = samples_ls[0][0].possibleQualities[feature2]

    if feature2 in samples_ls[0][0].xQuali.keys() and feature1 in samples_ls[0][0].xQuanti.keys():

        cols = [str(elem) for elem in feature1_values]
        idxs = samples_ls[0][0].possibleQualities[feature2]

    if feature2 in samples_ls[0][0].xQuanti.keys() and feature1 in samples_ls[0][0].xQuali.keys():
        cols = samples_ls[0][0].possibleQualities[feature1]
        idxs = [str(elem) for elem in feature2_values]

    if feature2 in samples_ls[0][0].xQuanti.keys() and feature1 in samples_ls[0][0].xQuanti.keys():

        cols = [str(elem) for elem in feature1_values]
        idxs = [str(elem) for elem in feature2_values]

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

def Plot_Feature_Predictions_2D(modelName, PredDf, f1, f2, displayParams, DBpath, studyFolder='PREDICTIONS/'):

    fig, ax = plt.subplots(figsize=(12,8))#
    PredDf = PredDf.astype(float)
    fontsize = 12

    figFolder = 'HEATMAP'
    figTitle = modelName +'_Feature_Predictions_' + f1 + f2

    ylabel, xlabel = f1, f2
    yLabels, xLabels = PredDf.columns, PredDf.index
    title = f1 + "-" + f2 + "( %s )" % studyParams['sets'][0][0][0]

    # title = modelName + 'Influence of Features on target - (%s)' % studyParams['sets'][0][0][0]
    heatmap  = sns.heatmap(PredDf, annot=True, fmt=".001f", ax=ax, cbar_kws={"shrink": .70,  "orientation": "horizontal"}, cmap="bwr",
                linewidths=3, linecolor="white", square=True) #'label': title,

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(yLabels))+(1/2))
    ax.set_xticklabels(yLabels, fontsize =fontsize)#, fontsize = 12
    ax.set_yticks(np.arange(len(xLabels))+(1/2))
    ax.set_yticklabels(xLabels, fontsize =fontsize)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_yticklabels(), rotation=0, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",rotation_mode="anchor")
    #set labels to axis
    # plt.xlabel(title, fontsize =12, fontweight='bold')
    # plt.ylabel("", fontsize =12) #xlabel

    cbar = heatmap.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=14)
    cbar.set_label(title, fontsize=fontsize)



    fig.tight_layout()

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + figFolder
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        plt.savefig(outputFigPath + '/' + figTitle + '.png')
        print(outputFigPath)
    if displayParams['showPlot']:
        plt.show()
    plt.close()

def Run_Feature_Predictions_2D(MyPred_Sample, feature1, feature2, Model_List=None, Blender_List=None, feature1_values = None, feature2_values = None):

    AllDfs = []
    SheetNames = []
    for model in Model_List + Blender_List:
        SheetNames.append(model.GSName)
        PredDf = create_Feature_Predictions_2D(MyPred_Sample, feature1, feature2, model, feature1_values = feature1_values, feature2_values = feature2_values)
        Plot_Feature_Predictions_2D(model.GSName, PredDf, feature1, feature2, displayParams, DB_Values['DBpath'])
        AllDfs.append(PredDf)

    if displayParams['archive']:

        import os
        reference = displayParams['reference']
        outputPathStudy = DB_Values['DBpath'] + "RESULTS/" + reference + 'RECORDS/' + 'PREDICTIONS/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with pd.ExcelWriter(outputPathStudy + 'Feature_Predictions_' + feature1 + feature2 + ".xlsx", mode='w') as writer:
            for df, name in zip(AllDfs, SheetNames):
                df.to_excel(writer, sheet_name=name)

