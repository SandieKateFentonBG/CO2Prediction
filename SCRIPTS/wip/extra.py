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