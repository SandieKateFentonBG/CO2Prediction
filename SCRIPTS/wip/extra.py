def computeCorrelationWith(dat, yLabels, round = 2):
    # todo : this does not work...
    x, y, xLabels = dat.asArray()
    xdf = pd.DataFrame(x, columns= xLabels)
    ydf = pd.DataFrame(y, columns= yLabels)
    print(xdf.shape)
    print(ydf.shape)
    return xdf.corrwith(ydf, axis = 0).round(round)


def highCorMatrix(fullMatrix, keepLabel):
    """This function is useless"""

    return fullMatrix.loc[keepLabel]


def highCorrDataframe(df, dropLabels):
    return df.drop(columns=dropLabels)

def computeYLabelCor(correlationMatrix, yLabel = 'Calculated tCO2e_per_m2'):

    """
    To visualize correlation values
    """

    #helper :
    # query row index correlationMatrix.index[0]
    # query col index correlationMatrix.columns[0]
    # query col  correlationMatrix.index[0]
    #reshape col np.array(yLabelCor).reshape(55,1)

    return correlationMatrix.loc[yLabel]



    # # highCorMatrix = correlationMatrix.loc[lowThreshhold <= abs((correlationMatrix[yLabel])) <= highThreshhold]
    #
    # highCorMatrix = correlationMatrix.loc[abs((correlationMatrix[yLabel])) >= lowThreshhold] #abs((correlationMatrix[yLabel])) <= highThreshhold and
    #
    # # for label in correlationMatrix.index:
    # #     print(label)
    # #     mc = correlationMatrix.loc[abs(correlationMatrix[label]) > highThreshhold]
    # # print('mc', mc)
    #
    # hc = correlationMatrix.loc[abs((correlationMatrix[yLabel])) >= lowThreshhold]
    # lc = correlationMatrix.loc[(abs((correlationMatrix[yLabel])) < lowThreshhold)]
    # # test = correlationMatrix.loc[(abs((correlationMatrix[label])) > highThreshhold) for label in correlationMatrix.index]
    # na = correlationMatrix.loc[correlationMatrix[yLabel].isna()]
    # # print('lc', lc)
    # # print('hc', hc.shape, hc)
    # # print('na', na)
    # ls = []
    # for label in correlationMatrix.index:
    #     print(label)#, correlationMatrix[label]
    #     #mc = hc.loc[abs(hc[label]) > highThreshhold]
    #     a = correlationMatrix.loc[(abs((correlationMatrix[label])) > highThreshhold)]
    #     if abs((correlationMatrix[label])) > highThreshhold:
    #         ls.append(label)
    #     # mc = correlationMatrix.loc[lambda correlationMatrix: correlationMatrix[label] > highThreshhold]
    # print('ls', ls, a)
    # lowCorMatrix = correlationMatrix.loc[(abs((correlationMatrix[yLabel])) < lowThreshhold)] \
    #                + correlationMatrix.loc[correlationMatrix['Calculated tCO2e_per_m2'].isna()]
    # #+ correlationMatrix.loc[(abs((correlationMatrix[yLabel])) > highThreshhold)]

def rotateSearchEval(xSets, ySets, modelingParams, displayParams, models):
    for i in range(5):

        (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xSets, ySets, i)
        (xTrainArr, yTrainArr), (xTestArr, yTestArr) = (xTrain.values, yTrain.values.reshape(-1, 1)), (
        xTest.values, yTest.values.reshape(-1, 1))
        print('Rotation', i)
        searchEval(modelingParams, displayParams, models, xTrainArr, yTrainArr, xTestArr, yTestArr)
