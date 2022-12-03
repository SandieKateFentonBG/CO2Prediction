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

def unscale(elem, scaler, scalerParam):

    if scalerParam:
        return pd.DataFrame(scaler.inverse_transform(elem), columns = elem.keys())
    else :
        return elem


# def XScaleYSplit(df, yLabels, scalerParam):
#     ydf = df[yLabels]
#     xdf = df.drop(columns = yLabels)
#     xScaler = None
#     if scalerParam:
#         if scalerParam == 'MinMaxScaler':
#             xScaler = preprocessing.MinMaxScaler()
#             x_normalized = xScaler.fit_transform(xdf)
#             xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
#         if scalerParam == 'StandardScaler':
#             xScaler = preprocessing.StandardScaler()
#             x_normalized = xScaler.fit_transform(xdf)
#             xScaled = pd.DataFrame(x_normalized, columns = xdf.keys())
#         xdf = xScaled
#     return xdf, ydf, xScaler
#
# def TrainTest(xdf, ydf, test_size, random_state):
#
#     XTrain, XTest, yTrain, yTest = train_test_split(xdf.values, ydf.values, test_size=test_size, random_state=random_state)
#     return XTrain, XTest, yTrain, yTest




#
#
# def crossvalidationSplit(x, y, batchCount=5):
#     cutoffIndex = [0] + [int(x.shape[0]/batchCount * i) for i in range(1, batchCount)] if x.shape[0] % batchCount == 0\
#         else [int(x.shape[0] / batchCount * i) for i in range(batchCount)]
#     cutoffIndex += [x.shape[0]]
#     xsets = [x.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]
#     ysets = [y.iloc[cutoffIndex[n]:cutoffIndex[n + 1]] for n in range(len(cutoffIndex) - 1)]
#
#     return xsets, ysets
#
# def TrainTestSets(filterDf, yLabels, scalerParam):
#     """Normalize and split Train-Test """
#     xdf, ydf, xScaler = XScaleYSplit(filterDf, yLabels, scalerParam)
#     xs, ys = crossvalidationSplit(xdf, ydf)
#     return xs, ys, xScaler
#
# def TrainTestDf(xSets, ySets, testIdParam=1):
#
#     xTrain = pd.concat([batch for batch in xSets if batch is not xSets[testIdParam]])
#     yTrain = pd.concat([batch for batch in ySets if batch is not ySets[testIdParam]])
#     return (xTrain, yTrain), (xSets[testIdParam], ySets[testIdParam])
#
# def TrainTestArray(filterDf, yLabels, testSetIndex):
#     #todo : shuffle data...
#     xs, ys, xScaler = TrainTestSets(filterDf, yLabels)
#     (xTrain, yTrain), (xTest, yTest) = TrainTestDf(xs, ys, testSetIndex)
#     return (xTrain.values, yTrain.values.reshape(-1, 1)), (xTest.values, yTest.values.reshape(-1, 1))
