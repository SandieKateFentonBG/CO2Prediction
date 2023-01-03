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


def plot_shap_SummaryPlot(GS, displayParams, DBpath, content='', studyFolder='GS_FS/'):
    """Plot shap summary for a fitted estimator and a set of test with its labels."""

    # shap_values, explainer = computeSHAP(GS)

    explainer = GS.SHAPexplainer
    shap_values = GS.SHAPvalues

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
        plt.savefig(outputFigPath + '/SHAP_summary' + GS.GSName + '.png')
    if displayParams['showPlot']:
        plt.show()
    plt.close()



def plot_shap_DecisionPlot(GS, displayParams, DBpath,  studyFolder='GS_FS/'):
    """Plot shap decision for a fitted estimator and a set of test with its labels."""

    # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.decision_plot.html
    shap_values = GS.SHAPvalues
    # y_pred = GS.yPred
    expected_value = GS.SHAPexplainer.expected_value
    #todo : this doesn't work for every model - ex : no selector
    misclassified = abs(GS.yPred - GS.learningDf.yTest) < abs(GS.learningDf.yTest) * GS.accuracyTol

    # plot & save SHAP
    plt.gcf().set_size_inches(14, 6)



    shap_decision = shap.decision_plot(base_value = expected_value, shap_values = shap_values,auto_size_plot=False,
                                       features = GS.learningDf.XTest,highlight=misclassified,
                                       show=displayParams['showPlot'],title = GS.GSName)
    plt.yticks(fontsize=14)
    plt.suptitle(GS.GSName, ha="right", size = 'large' )

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'SHAP'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAP_Decison_' + GS.GSName + '.png')

    plt.close()

def plot_shap_group_cat_DecisionPlot(GS, displayParams, DBpath, studyFolder='GS_FS/'):
    """Plot shap decision for a fitted estimator and a set of test with its labels."""

    # https://shap-lrjball.readthedocs.io/en/latest/generated/shap.decision_plot.html

    expected_value = GS.SHAPexplainer.expected_value
    misclassified = abs(GS.yPred - GS.learningDf.yTest) < abs(GS.learningDf.yTest) * GS.accuracyTol

    # plot & save SHAP
    plt.gcf().set_size_inches(14, 6)
    shap_decision = shap.decision_plot(base_value=expected_value, shap_values=GS.SHAPGroupvalues, auto_size_plot=False,
                                       feature_names=list(GS.SHAPGroup_RemapDict.keys()), highlight=misclassified,
                                       show=displayParams['showPlot'], title=GS.GSName)
    plt.suptitle(GS.GSName, ha="right", size='large')

    reference = displayParams['reference']
    if displayParams['archive']:
        path, folder, subFolder = DBpath, "RESULTS/", reference + 'VISU/' + studyFolder + 'SHAP-GROUPED'
        import os
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        plt.savefig(outputFigPath + '/SHAP_GROUPED_Decison_' + GS.GSName + '.png')

    plt.close()





def plot_shap_group_cat_SummaryPlot(GS, xQuantLabels, xQualLabels, displayParams, DBpath, content='', studyFolder='GS_FS/'):
    """Plot shap summary for a fitted estimator and a set of test with its labels - categorical features will be grouped."""

    # shap_values, explainer = computeSHAP(GS)

    explainer = GS.SHAPexplainer
    shap_values = explainer.shap_values(GS.learningDf.XTest)

    # plot & save SHAP values
    shap_summary = shap.summary_plot(shap_values=GS.SHAPGroupvalues, feature_names = list(GS.SHAPGroup_RemapDict.keys()),
                                     plot_type="dot", show = False)
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
