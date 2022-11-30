import shap
import pandas as pd
import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def computeSHAP(GS):
    """Plot shap summary for a fitted estimator and a set of test with its labels."""
    import shap

    clf = GS.Estimator
    Xtest = GS.learningDf.XTest  # for SHAP VALUES
    Xtrain = GS.learningDf.XTrain  # for average values

    # compute initial SHAP values
    sample = shap.sample(Xtrain, 30)
    masker = shap.maskers.Independent(Xtrain)
    try:
        explainer = shap.Explainer(clf, masker)
    except Exception:
        explainer = shap.KernelExplainer(clf.predict, sample)
    shap_values = explainer.shap_values(Xtest)
    return shap_values, explainer


def plot_shap(GS, displayParams, DBpath, content='', studyFolder='GS_FS/'):
    """Plot shap summary for a fitted estimator and a set of test with its labels."""

    # shap_values, explainer = computeSHAP(GS)

    explainer = GS.SHAPexplainer
    shap_values = explainer.shap_values(GS.learningDf.XTest)

    # plot & save SHAP
    shap_summary = shap.summary_plot(shap_values=shap_values, features=GS.learningDf.XTest, plot_type="dot", show = False)
    plt.suptitle(GS.GSName, ha="right", size = 'large' )
    plt.gcf().set_size_inches(12, 6)
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'SHAP'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAP_' + GS.GSName + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

    return explainer


def plot_shap_group_cat(GS, xQuantLabels, xQualLabels, displayParams, DBpath, content='', studyFolder='GS_FS/'):
    """Plot shap summary for a fitted estimator and a set of test with its labels - categorical features will be grouped."""

    # shap_values, explainer = computeSHAP(GS)

    explainer = GS.SHAPexplainer
    shap_values = explainer.shap_values(GS.learningDf.XTest)

#todo : toggle this - and add in Model.py
    #find re-mapping to group sub-categories into single category
    transformList  = []
    for sLabel in GS.learningDf.selectedLabels:
        if sLabel in xQuantLabels:
            transformList.append(sLabel)
        else:
            for qLabel in xQualLabels:
                if qLabel in sLabel:
                    transformList.append(qLabel)
    remap_dict = {i:transformList.count(i) for i in transformList} #dictionary for remapping
    keyList = list(remap_dict.keys())
    lengthList = list(remap_dict.values())

    #compute new SHAP values
    new_shap_values = []
    for values in shap_values:
        # split shap values into a list for each feature
        values_split = np.split(values, np.cumsum(lengthList[:-1]))
        # sum values within each list
        values_sum = [sum(l) for l in values_split]
        new_shap_values.append(values_sum)
    # replace SHAP values
    shap_values = np.array(new_shap_values)
    # todo : toggle end

    # plot & save SHAP values
    shap_summary = shap.summary_plot(shap_values=shap_values, feature_names = keyList, plot_type="dot", show = False)
    plt.suptitle(GS.GSName, ha="right", size = 'large' )
    plt.gcf().set_size_inches(12, 6)
    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'SHAP-GROUPED'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAP_GROUPED_' + GS.GSName + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()

    return explainer, remap_dict

def SHAP_Df(GS):
    Xtrain = GS.learningDf.XTrain
    explainer = GS.SHAPexplainer
    shap_values = explainer.shap_values(GS.learningDf.XTest)
    shap_importance = pd.Series(shap_values.values, Xtrain.columns).abs().sort_values(ascending=False)



# todo : LATER - Shap advanced uses - grouping and correlation
# https://towardsdatascience.com/you-are-underutilizing-shap-values-feature-groups-and-correlations-8df1b136e2c2
# https://www.kaggle.com/code/estevaouyra/shap-advanced-uses-grouping-and-correlation/notebook
# def plot_SHAP_by_group
#     from itertools import repeat, chain
#     revert_dict = lambda d: dict(chain(*[zip(val, repeat(key)) for key, val in d.items()]))

# def grouped_shap(shap_vals, features, groups):
#     groupmap = revert_dict(groups)
#     shap_Tdf = pd.DataFrame(shap_vals, columns=pd.Index(features, name='features')).T
#     shap_Tdf['group'] = shap_Tdf.reset_index().features.map(groupmap).values
#     shap_grouped = shap_Tdf.groupby('group').sum().T
#     return shap_grouped
