

def save_model(dataFrame, model_name, min_value, max_value): #todo : not used or checked yet
    import pickle
    file_name_string = './data/model_data_prediction/' + model_name + "_predictions"
    if min_value != None or max_value != None:
        file_name_string += "_from_" + str(min_value) + "_to_" + str(max_value)
    file_name_string += ".bin"
    print("this is the filename string")
    print("Saving model on " + file_name_string)

    dataFrame[min_value:max_value].to_pickle(file_name_string)

def execute(model):