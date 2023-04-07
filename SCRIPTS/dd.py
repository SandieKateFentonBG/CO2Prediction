import numpy as np
import pandas as pd
df1 = pd.DataFrame(dict(x=np.random.randn(10), y=np.random.randint(0, 5, 10), z=np.random.randint(-3, 2, 10)))
df2 = pd.DataFrame(dict(x=np.random.randn(10), y=np.random.randint(0, 2, 10), z=np.random.randint(-1, 2, 10)))

print(df1)
print(df2)
df = pd.concat([df1, df2]).groupby(level=0).mean()

print(df)


# The above exception was the direct cause of the following exception:
#
# Traceback (most recent call last):
#   File "C:/Users/sfenton/Code/Repositories/CO2Prediction/SCRIPTS/Main_Combine.py", line 68, in <module>
#     RUN_Combine_Report(All_CV, NBest_CV, Blenders_NBest_CV, randomvalues, displayParams)
#   File "C:\Users\sfenton\Code\Repositories\CO2Prediction\SCRIPTS\Main_Combine_Steps.py", line 41, in RUN_Combine_Report
#     report_BL_NBest_CV(CV_BlenderNBest, displayParams,  DB_Values['DBpath'], randomvalues)
#   File "C:\Users\sfenton\Code\Repositories\CO2Prediction\SCRIPTS\Model_Blending_CV.py", line 287, in report_BL_NBest_CV
#     ExtraDf.loc['BlenderAvg-BestModelAvg', :] = (Combined_Df.loc[BLE_VALUES['Regressor'] + "_Blender_NBest", :] - Combined_Df.iloc[0, :])
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexing.py", line 889, in __getitem__
#     return self._getitem_tuple(key)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexing.py", line 1060, in _getitem_tuple
#     return self._getitem_lowerdim(tup)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexing.py", line 807, in _getitem_lowerdim
#     section = self._getitem_axis(key, axis=i)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexing.py", line 1124, in _getitem_axis
#     return self._get_label(key, axis=axis)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexing.py", line 1073, in _get_label
#     return self.obj.xs(label, axis=axis)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\generic.py", line 3739, in xs
#     loc = index.get_loc(key)
#   File "C:\Users\sfenton\Anaconda3\envs\ml_labs\lib\site-packages\pandas\core\indexes\base.py", line 3082, in get_loc
#     raise KeyError(key) from err
# KeyError: 'SVR_RBF_Blender_NBest'
#
# Process finished with exit code 1
