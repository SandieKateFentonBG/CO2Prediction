# plotScaleResDistribution(studies, displayParams)
#
# residualsMeanVar = plotResHistGauss(studies, displayParams, binwidth = 10, setxLim =(-300, 300))# (-150, 150)

def mergeList(list):

    return [j for i in list for j in i]

def assembleResid(studies):
    residualsSplit = dict()
    residualsMerge = dict()

    for i in range(len(studies[0])):
        residualsSplit[str(studies[0][i]['model'])] = []
        residualsMerge[str(studies[0][i]['model'])] = []
    for h in range(len(studies)):
        for i in range(len(studies[h])):

            residualsSplit[str(studies[h][i]['model'])].append((studies[h][i]['bModelResid']).reshape(1, -1).tolist()[0])
    for k in residualsMerge.keys():
        residualsMerge[k] = mergeList(residualsSplit[k])

    return residualsSplit, residualsMerge