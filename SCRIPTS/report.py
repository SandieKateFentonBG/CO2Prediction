def pickleDumpMe( DBpath, displayParams, obj, name):

    if displayParams['archive']:
        reference = displayParams['reference']

        path, folder, subFolder = DBpath, "RESULTS/", reference + 'RECORDS/' + name
        import os
        import pickle
        outputFigPath = path + folder + subFolder
        if not os.path.isdir(outputFigPath):
            os.makedirs(outputFigPath)

        with open(outputFigPath, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('FILE has been saved here :', outputFigPath)


def pickleLoadMe(path, show = False):
    import pickle
    with open(path, 'rb') as handle:
        mydict = pickle.load(handle)
    if show:
        for k, v in mydict.items():
                print(' ', k, ' : ', v)
    return mydict