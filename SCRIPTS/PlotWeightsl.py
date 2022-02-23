


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
        weightsDict[i] = j
    return weightsDict

def modelWeightsList(df, target, features, weights, intercept):
    weightsDict = modelWeightsDict(df, target, features, weights, intercept)
    ks = list(weightsDict.keys())
    ws = list(weightsDict.values())
    return ks, ws




