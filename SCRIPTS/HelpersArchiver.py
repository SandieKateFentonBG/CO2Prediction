import numpy as np
from HelpersVisualizer import *

def pickleDumpMe( DBpath, displayParams, obj, objFolder, objName):
    #objFolder = DATA; FILTER; WRAPPER; GS
    if displayParams['archive']:
        reference = displayParams['reference']
        import os
        import pickle
        outputFigPath = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)
        outputFigPath = f'{outputFigPath}/{objName}.pkl'

        with open(outputFigPath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('FILE has been saved here :', outputFigPath)


def pickleLoadMe(path, show = False):
    import pickle
    with open(path, 'rb') as handle:
        obj = pickle.load(handle)
    if show:
        print(obj)
    return obj


def saveStudy(DBpath, displayParams, obj, objFolder = 'report'):

    import inspect
    if displayParams['archive']:
        import os
        reference = displayParams['reference']
        outputPathStudy = DBpath + "RESULTS/" + reference + 'RECORDS/' + objFolder + '/'

        if not os.path.isdir(outputPathStudy):
            os.makedirs(outputPathStudy)

        with open(outputPathStudy  + "Archive.txt", 'a') as f:
            print('', file=f)
            test = inspect.getmembers(obj)
            for r in test:
                print(r, file=f)

        f.close()















