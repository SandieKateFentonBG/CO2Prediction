from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error, r2_score
from Archiver import *
from PrepData import TrainTestDf
from GridSearch import *

"""Models"""
linearReg = {'model' : LinearRegression(), 'param' : None, 'Linear' : True} #why doies this not have a regul param?
lassoReg = {'model' : Lasso() , 'param': 'alpha', 'Linear' : True} # for overfitting
ridgeReg = {'model' : Ridge(), 'param': 'alpha', 'Linear' : True}
elasticNetReg = {'model' : ElasticNet(), 'param': 'alpha', 'Linear' : True}
supportVectorLinReg = {'model' : SVR(kernel='linear'), 'param': 'C', 'Linear' : True}
supportVectorRbfReg = {'model' : SVR(kernel='rbf'), 'param': 'C', 'Linear' : False}
supportVectorPolReg = {'model' : SVR(kernel='poly'), 'param': 'C', 'Linear' : False}
# kernelRidgeReg = {'model' : KernelRidge(), 'param': 'alpha'}
kernelRidgeLinReg = {'model' : KernelRidge(kernel='linear'), 'param': 'alpha', 'Linear' : False}
kernelRidgeRbfReg = {'model' : KernelRidge(kernel='rbf'), 'param': 'alpha', 'Linear' : False}
kernelRidgePolReg = {'model' : KernelRidge(kernel='polynomial'), 'param': 'alpha', 'Linear' : False}
models = [linearReg, lassoReg, ridgeReg, elasticNetReg, supportVectorLinReg, supportVectorRbfReg, supportVectorPolReg,
        kernelRidgeLinReg, kernelRidgeRbfReg, kernelRidgePolReg] #linearReg,
# #
modelsa = [linearReg, lassoReg]

def emptyWeights(df, target): #keys = df.keys()
    # weights = list(df.keys()[0: -1])+['intercept']
    weights = list(df.keys()) #+['intercept']
    weights.remove(target)
    weightsDict = dict()
    for w in weights:
        weightsDict[w]= None

    return weightsDict


def modelWeightsDict(df, target, features, weights, intercept):
    weightsDict = emptyWeights(df, target)
    for i, j in zip(features, weights):
        weightsDict[i]=j
    # weightsDict['intercept'] = intercept
    return weightsDict

def modelWeightsList(df, target, features, weights, intercept):
    weightsDict = modelWeightsDict(df, target, features, weights, intercept)
    ks = list(weightsDict.keys())
    ws = list(weightsDict.values())
    return ks, ws

def unpack_results_for_points(results):
    rlist = []
    clist = []
    vlist = []
    for r in range(len(results)):
        for c in range(len(results[0])):
            # for d in range(len(results[0][0])):
            rlist.append(results[r][c][0])
            clist.append(results[r][c][1])
            vlist.append(results[r][c][2])
    return rlist, clist, vlist

# def assembleWeights():
#     rlist = []
#     clist = []
#     vlist = []
#     for m in models:
#         for c in range(len)


    #
    #
    # colors = ['r', 'g', 'b', 'y']
    # title = 'matrix completion'
    # xlabel, ylabel, zlabel = 'model', 'feature', 'weight'
    # xli =
    # yli, zli =
    # for m in models:
