#DASHBOARD IMPORT
from Dashboard_EUCB_FR_v2 import *

#LIBRARY IMPORTS
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge

#SCRIPT IMPORTS
from Model_Blending import *


def Run_Blending_NBest(modelList, displayParams, DBpath, import_reference, ConstructorKey = 'LR_RIDGE'):

    #CONSTRUCT
    LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
    LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}
    SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
    SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
    LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
    CONSTRUCTOR_DICT = {'LR': LR_CONSTRUCTOR, 'LR_RIDGE' : LR_RIDGE_CONSTRUCTOR,
                        'SVR_RBF': SVR_RBF_CONSTRUCTOR, 'SVR_LIN': SVR_LIN_CONSTRUCTOR,
                        'LR_ELAST': LR_ELAST_CONSTRUCTOR}

    CONSTRUCTOR = CONSTRUCTOR_DICT[ConstructorKey]

    # IMPORT MODELLIST



    # CONSTRUCT & REPORT
    print('RUNNING BLENDING')
    blendModel = Model_Blender(modelList, CONSTRUCTOR, Gridsearch = True, Type='NBest')
    blendModel.report_Blending_NBest(displayParams, DBpath)
    pickleDumpMe(DBpath, displayParams, blendModel, 'BLENDER', blendModel.GSName)

    # LOAD
    blendModel = import_Blender_NBest(import_reference)


    # PLOT
    print('PLOTTING BLENDER')
    blendModel.plotBlenderYellowResiduals(displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                       yLim=PROCESS_VALUES['residualsYLim'], xLim=PROCESS_VALUES['residualsXLim'],
                                       studyFolder='BLENDER/')

    return blendModel


def Run_Blending_CV(modelList, displayParams, DBpath, import_reference, ConstructorKey = 'LR_RIDGE'):

    #CONSTRUCT
    LR_CONSTRUCTOR = {'name': 'LR', 'modelPredictor': LinearRegression(), 'param_dict': dict()}
    LR_RIDGE_CONSTRUCTOR = {'name': 'LR_RIDGE', 'modelPredictor': Ridge(), 'param_dict': LR_param_grid}
    SVR_RBF_CONSTRUCTOR = {'name' : 'SVR_RBF',  'modelPredictor' : SVR(kernel ='rbf'),'param_dict' : SVR_param_grid}
    SVR_LIN_CONSTRUCTOR = {'name' : 'SVR_LIN',  'modelPredictor' : SVR(kernel ='linear'),'param_dict' : SVR_param_grid}
    LR_ELAST_CONSTRUCTOR = {'name' : 'LR_ELAST',  'modelPredictor' : ElasticNet(),'param_dict' : LR_param_grid}
    CONSTRUCTOR_DICT = {'LR': LR_CONSTRUCTOR, 'LR_RIDGE' : LR_RIDGE_CONSTRUCTOR,
                        'SVR_RBF': SVR_RBF_CONSTRUCTOR, 'SVR_LIN': SVR_LIN_CONSTRUCTOR,
                        'LR_ELAST': LR_ELAST_CONSTRUCTOR}

    CONSTRUCTOR = CONSTRUCTOR_DICT[ConstructorKey]

    # IMPORT MODELLIST



    # CONSTRUCT & REPORT
    print('RUNNING BLENDING')
    blendModel = Model_Blender(modelList, CONSTRUCTOR, Gridsearch = True, Type='NBest')
    blendModel.report_Blending_NBest(displayParams, DBpath)
    pickleDumpMe(DBpath, displayParams, blendModel, 'BLENDER', blendModel.GSName, combined=True)

    # LOAD
    blendModel = import_Blender_NBest(import_reference)


    # PLOT
    print('PLOTTING BLENDER')
    blendModel.plotBlenderYellowResiduals(displayParams=displayParams, DBpath=DB_Values['DBpath'],
                                       yLim=PROCESS_VALUES['residualsYLim'], xLim=PROCESS_VALUES['residualsXLim'],
                                       studyFolder='BLENDER/')

    return blendModel



def import_Blender_NBest(import_reference, label ='LR_RIDGE_Blender_NBest'):
    path = DB_Values['DBpath'] + 'RESULTS/' + import_reference + 'RECORDS/BLENDER/' + label + '.pkl'
    Blender = pickleLoadMe(path=path, show=False)

    return Blender