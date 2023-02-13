import numpy as np
import pandas as pd

def Highest3DPoints(points, max = True, absVal = False):

    best = points[0][0]
    for i in range(len(points)):

        for j in range(len(points[0])):
            if max:
                if absVal:
                    if abs(points[i][j][2])>abs(best[2]):
                        best = points[i][j]
                else:
                    if points[i][j][2]>best[2]:
                        best = points[i][j]
            else:
                if absVal:
                    if abs(points[i][j][2])<abs(best[2]):
                        best = points[i][j]
                else:
                    if points[i][j][2]<best[2]:
                        best = points[i][j]

    return best

def unpackResPts(results):
    xli = []
    yli = []
    zli = []
    for r in range(len(results)):
        for c in range(len(results[0])):
            # for d in range(len(results[0][0])):
            xli.append(results[r][c][0])
            yli.append(results[r][c][1])
            zli.append(results[r][c][2])
    xl, yl, zl = np.array(xli), np.array(yli), np.array(zli)
    return xl, yl, zl

def unpackResLines(results):
    rlist = []
    for r in range(len(results)):
        row = []
        x, y, z = [], [], []
        for c in range(len(results[0])):
            x.append(results[r][c][0])
            y.append(results[r][c][1])
            z.append(results[r][c][2])
            row = [x, y, z]
        rlist.append(row)
    return rlist

def Construct3DPoints(ResultsList, key ='gamma', score ='mean_test_r2'): #x : featureCount, y : valScore

    pts = []
    labels =[]

    for j in range(len(ResultsList)):

        #todo : check !!
        lab = ResultsList[j].predictorName + '-' + ResultsList[j].selectorName
        labels.append(lab)
        # labels.append(ResultsList[j].predictorName)
        modelRes = []
        for i in range(len(ResultsList[j].param_dict[key])): #x : gamma value
            paramRes = [j, ResultsList[j].param_dict[key][i], ResultsList[j].Grid.cv_results_[score][i]] #y : Score

            modelRes.append(paramRes)
        pts.append(modelRes)
    return pts, labels


def GS_Construct3DPoints(GS_FSs, score): #x : featureCount, y : valScore

    pts = []
    labels =[]

    for j in range(len(GS_FSs)): # j is the number of predictor types : KRR, SVR,LASSO
        myModel = GS_FSs[j]
        lab = GS_FSs[j].predictorName
        labels.append(lab) #7

        modelRes = []
        for i in range(len(GS_FSs[j].learningDfsList)): #GS_FS learningDfs : DTR, RFR, DTC
            AllLearningDfs = myModel.learningDfsList
            learningDflabel = AllLearningDfs[i]
            MyModelwithDf = myModel.__getattribute__(learningDflabel)
            MyScore = MyModelwithDf.__getattribute__(score)
            paramRes = [j, i, MyScore]

            modelRes.append(paramRes)
        pts.append(modelRes)
    return pts, labels

def GS_ConstructDataframe(GS_FSs, score):

    # xLabel = 'Predictor'
    # ylabel = 'Feature Selection'
    # zlabel = score

    yLabels = GS_FSs[0].learningDfsList
    pts, xLabels = GS_Construct3DPoints(GS_FSs, score)
    xl, yl, zl = unpackResPts(pts)
    results = np.reshape(zl, (len(xLabels), len(yLabels)))

    df = pd.DataFrame(results, index=xLabels, columns=yLabels)

    return df

def showAttributes(obj):

    for attr in dir(obj):
        print("obj.%s = %r" % (attr, getattr(obj, attr)))