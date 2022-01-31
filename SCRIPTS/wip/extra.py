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

Method: LRmodel Evaluation: 0.20838623374056542 Accuracy: 0.07142857142857142 MSE: 0.0148902751375687
Method: SVMmodel Evaluation: 0.2529888632278974 Accuracy: 0.14285714285714285 MSE: 0.014051298539089804
Method: RFmodel Evaluation: -0.1817663607525839 Accuracy: 0.0 MSE: 0.022229055392857144
Method: XGBmodel Evaluation: -0.6050890964875479 Accuracy: 0.07142857142857142 MSE: 0.030191762914865864


initial size (80, 5)
without outliers (70, 5)
without uncorrelated features (70, 5)

Method: LRmodel Evaluation: 0.09426283881941555 Accuracy: 0.0 MSE: 0.017036939107346633
Method: SVMmodel Evaluation: 0.09832980529422752 Accuracy: 0.0 MSE: 0.016960439364206267
Method: RFmodel Evaluation: -0.21732492876546194 Accuracy: 0.07142857142857142 MSE: 0.02289791296428571
Method: XGBmodel Evaluation: -0.6357782960928142 Accuracy: 0.07142857142857142 MSE: 0.03076903590440185

initial size (80, 18)
without outliers (70, 18)
without uncorrelated features (70, 9)

Method: LRmodel Evaluation: 0.03222639640241043 Accuracy: 0.14285714285714285 MSE: 0.018203846171772793
Method: SVMmodel Evaluation: 0.013652678751817748 Accuracy: 0.07142857142857142 MSE: 0.01855321827459978
Method: RFmodel Evaluation: -0.15865480517472608 Accuracy: 0.07142857142857142 MSE: 0.021794326442857146
Method: XGBmodel Evaluation: -0.5962645310361918 Accuracy: 0.07142857142857142 MSE: 0.03002577196478008

initial size (80, 55)
without outliers (70, 55)
without uncorrelated features (70, 13)

Method: LRmodel Evaluation: 0.1854826039012566 Accuracy: 0.07142857142857142 MSE: 0.015321092999122373
Method: SVMmodel Evaluation: 0.25273673797785223 Accuracy: 0.0 MSE: 0.014056041021474919
Method: RFmodel Evaluation: -0.1975636633200104 Accuracy: 0.0 MSE: 0.022526203057142858
Method: XGBmodel Evaluation: -0.6006147761823821 Accuracy: 0.07142857142857142 MSE: 0.030107604468472413