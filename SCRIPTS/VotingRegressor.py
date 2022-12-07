# # ModelLs = []
# # ModelLsName = []
# # for GS_FS in GS_FSs:
# #     for learningDflabel in GS_FS.learningDfsList:
# #         GS = GS_FS.__getattribute__(learningDflabel)
# #         ModelLs.append(GS.bModel)
# #         ModelLsName.append(GS.GSName)
# #
# # ereg = VotingRegressor([(name, model) for (name,model) in zip(ModelLs, ModelLsName)])
# #
# # ereg.fit(X, y)
# #
# # # https://www.geeksforgeeks.org/ensemble-methods-in-python/
#
#
# # importing utility modules
# import pandas as pd
# from sklearn.metrics import mean_squared_error
#
# # importing machine learning models for prediction
# from sklearn.ensemble import RandomForestRegressor
# import xgboost as xgb
# from sklearn.linear_model import LinearRegression
#
# # importing train test split
# from sklearn.model_selection import train_test_split
#
# # loading train data set in dataframe from train_data.csv file
# df = pd.read_csv("train_data.csv")
#
# # getting target data from the dataframe
# target = df["target"]
#
# # getting train data from the dataframe
# train = df.drop("target")
#
# # Splitting between train data into training and validation dataset
# X_train, X_test, y_train, y_test = train_test_split(train, target, test_size=0.20)
#
# # performing the train test and validation split
# train_ratio = 0.70
# validation_ratio = 0.20
# test_ratio = 0.10
#
# # performing train test split
# x_train, x_test, y_train, y_test = train_test_split(
#     train, target, test_size=1 - train_ratio)
#
# # performing test validation split
# x_val, x_test, y_val, y_test = train_test_split(
#     x_test, y_test, test_size=test_ratio / (test_ratio + validation_ratio))
#
# # initializing all the base model objects with default parameters
# model_1 = LinearRegression()
# model_2 = xgb.XGBRegressor()
# model_3 = RandomForestRegressor()
#
# # training all the model on the train dataset
#
# # training first model
# model_1.fit(x_train, y_train)
# val_pred_1 = model_1.predict(x_val)
# test_pred_1 = model_1.predict(x_test)
#
# # converting to dataframe
# val_pred_1 = pd.DataFrame(val_pred_1)
# test_pred_1 = pd.DataFrame(test_pred_1)
#
# # training second model
# model_2.fit(x_train, y_train)
# val_pred_2 = model_2.predict(x_val)
# test_pred_2 = model_2.predict(x_test)
#
# # converting to dataframe
# val_pred_2 = pd.DataFrame(val_pred_2)
# test_pred_2 = pd.DataFrame(test_pred_2)
#
#
# # training third model
# model_3.fit(x_train, y_train)
# val_pred_3 = model_1.predict(x_val)
# test_pred_3 = model_1.predict(x_test)
#
# # converting to dataframe
# val_pred_3 = pd.DataFrame(val_pred_3)
# test_pred_3 = pd.DataFrame(test_pred_3)
#
#
#
# GS_FSs = import_Main_GS_FS(import_reference ='CSTB_rd43/')
# [LR, LR_RIDGE, LR_LASSO, LR_ELAST,  KRR_LIN, KRR_RBF,KRR_POL, SVR_LIN, SVR_RBF] = GS_FSs_43
#


# GS1 = LR_LASSO.RFE_GBR
# blend_train_1 = GS1.predict(raw_x_train) #dim 400*1
# blend_train_2 = GS2.predict(raw_x_train) #dim 400*1
# blend_df_train = pd.concat([blend_train_1, blend_train_2], axis=1) #dim 400*2

# blend_test_1 = GS1.predict(raw_x_test) #dim 20*1
# blend_test_2 = GS2.predict(raw_x_test) #dim 20*1
# blend_df_test = pd.concat([blend_test_1, blend_test_2], axis=1) #dim 20*2


# # making the final model using the meta features
# blend_model = LinearRegression()
# blend_model.fit(blend_df_train, y_train)
# final_pred = final_model.predict(blend_df_test)
# print(mean_squared_error(y_test, final_pred))
#
#
#
# # concatenating validation dataset along with all the predicted validation data (meta features)
# df_train = pd.concat([train_1, train_2], axis=1)


# df_test = pd.concat([x_test, test_pred_1, test_pred_2, test_pred_3], axis=1)
#
# # making the final model using the meta features
# final_model = LinearRegression()
# final_model.fit(df_val, y_val)
#
# # getting the final output
# final_pred = final_model.predict(df_test)
#
# # printing the mean squared error between real value and predicted value
# print(mean_squared_error(y_test, pred_final))