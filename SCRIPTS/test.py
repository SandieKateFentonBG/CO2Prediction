
from Dashboard_EUCB_FR_v2 import *
from Model_Blending_CV import *
from Main_FS_Steps import *

def get_minvalue(inputlist):
    min_value = min(inputlist)
    min_index = inputlist.index(min_value)
    return min_index
list1 = [23,56,32,89,21,44,51]

min_index = get_minvalue(list1)
print(min_index)

rd = 42

ref_prefix = DB_Values['acronym'] + '_' + studyParams['sets'][0][1]
ref_suffix_single = '_rd'
ref_suffix_combined = '_Combined/'
ref_single = ref_prefix + ref_suffix_single + str(PROCESS_VALUES['random_state']) + '/'

rdat, df, learningDf, baseFormatedDf, spearmanFilter, pearsonFilter, RFEs = import_Main_FS(ref_single, show = False)



XVal = baseFormatedDf.XVal.to_numpy()
XCheck = baseFormatedDf.XCheck.to_numpy()
yVal = baseFormatedDf.yVal.to_numpy().ravel()
yCheck = baseFormatedDf.yCheck.to_numpy().ravel()
XMeta = np.concatenate((XVal, XCheck), axis=0)
yMeta = np.concatenate((yVal, yCheck), axis=0)

print("check concatenate in good direction", XVal.shape, XCheck.shape, XMeta.shape)
print("check concatenate in good direction", yVal.shape, yCheck.shape, yMeta.shape)

XMeta = pd.DataFrame(XMeta)
# yMeta = pd.DataFrame(yMeta)

print('hi', type(XMeta), type(yMeta))

kf = split_blending_cv(XMeta, yMeta, k = 5)