import numpy as np

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