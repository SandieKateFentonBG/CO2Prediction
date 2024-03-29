# import numpy as np
#
#
#
# """
# ________________________________________________________________________________________________________________________
# RUN
# ________________________________________________________________________________________________________________________
# """
# #change when running a test
#
# displayParams = {"reference" : 'CSTB_residential_rd42/', 'showPlot': False, 'archive': True, 'showCorr' : False, 'plot_all': False}
#
# """
# ________________________________________________________________________________________________________________________
# DATABASE
# ________________________________________________________________________________________________________________________
# """
# #parameters specific to the database processed
#
# """
# EUCB-FR
# """
# DB_Values = {"DBpath" : "C:/Users/sfenton/Code/Repositories/CO2Prediction/",
#              "DBname" : "EU-ECB_dataset_feature_engineered_fr_residential_skf",
#              "DBdelimiter" : ';', "DBfirstLine" : 5 , 'acronym' : 'CSTB_residential_'} #"EU-ECB_dataset_feature_engineered_fr_skf",
#
# xQualLabels = [
# 'bldg_use_subtype',
# 'bldg_area_interval',
# 'bldg_struct_type',
# 'bldg_roof_type',
# 'bldg_energy_class_country',
# 'inv_mat_1_type',
# 'scope_parts',
# 'bldg_year_complete_interval']
#
# #add or remove : 'bldg_use_type''bldg_use_type',
#
# #incomplete : 'bldg_floors_ag','bldg_energy_class_general',
# #useless for this study ['admin_project_contact',  'bldg_project_status', 'site_country', 'bldg_QTO_type', 'lca_RSP', 'bldg_certification','lca_software', 'lca_database', 'scope_handling_D',]
#
# xQuantLabels = ['bldg_area_gfa', 'bldg_floors_bg']
# #incomplete : 'bldg_floors_ag','inv_energy_consumption','bldg_users_total',
# RemoveOutliersFrom = ['bldg_area_gfa']
#
# yLabels = ['GHG_sum_em_m2'] #'GHG_P2_sum_m2'
#
# #yLabels
# #'GHG_sum_em_m2' (max - 1799,718511)
# #'GHG_P2_sum_m2'
# #'GHG_A123_m2_harm_LCM' (max - 1226,516419)
# #'GHG_P1_sum_m2a'
#
# FORMAT_Values = {'yUnitFactor': 1, 'targetLabels': ['kgCO2e/m2'], 'TargetMinMaxVal': [0, 1500]}
#
# #'yUnitFactor' converts from yLabel unit to target Label unit:
# # ex : - if yLabel in kgCO2e : 1; if yLabel in tCO2e : 1000
#
# #todo : what to do with empty data? deletz line? replace by?
# # todo : remove outliers for number floors if count = 0 removes everything - what to do ?
#
# """
# ________________________________________________________________________________________________________________________
# PROCESSING
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for database processing
#
# PROCESS_VALUES = {'OutlierCutOffThreshhold' : 3, 'random_state' : 43, 'test_size' : 0.5, 'train_size': 0.8,
#                 'corrMethod1' : "spearman", 'corrMethod2' : "pearson", 'corrRounding' : 2, 'corrLowThreshhold' : 0.1,
#                      'corrHighThreshhold' : 0.65, 'corrHighThreshholdSpearman' : 0.75, 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]}
# #todo : check 'residualsYLim': [-500, 500], 'residualsXLim': [0, 800]
#
#
# """
# ________________________________________________________________________________________________________________________
# GRIDSEARCH
# ________________________________________________________________________________________________________________________
# """
# #parameters chosen for gridsearch opimization
#
# """
# ________________________________________________________________________________________________________________________
# FEATURE SELECTION
# ________________________________________________________________________________________________________________________
# """
#
# RFE_VALUES = {'RFE_n_features_to_select' : 15, 'RFE_featureCount' : 'list(np.arange(10, len(baseFormatedDf.XTrain)-10, 10))',
#               'RFE_process' : 'short', 'output_feature_count':'rfeCV'}
#
# #[5, 10, 15, 20, 25]['rfeHyp', 'rfeCV', int]
#
# """
# ________________________________________________________________________________________________________________________
# MODEL
# ________________________________________________________________________________________________________________________
# """
#
# GS_VALUES = {'coef0_range' : list(10.0 ** np.arange(-2, 2)),
#             'regul_range' : list(10.0 ** np.arange(-4, 4)),
#             'influence_range' : list(10.0 ** np.arange(-4, 4)),
#             'degree' : [2, 3, 4],
#             'margin_range' : list(10.0 ** np.arange(-4, 4)),
#             'kernel_list' : ['poly', 'linear', 'rbf']}
#
# """
# ________________________________________________________________________________________________________________________
# V1 - kernels  in params
# ________________________________________________________________________________________________________________________
# """
#
# # LR_param_grid={'alpha': GS_VALUES['regul_range']}
# # KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
# #                                                 'kernel' : GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}
# # SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
# #                 'epsilon': GS_VALUES['margin_range'], 'kernel': GS_VALUES['kernel_list'], 'coef0' : GS_VALUES['coef0_range']}
#
# """
# ________________________________________________________________________________________________________________________
# V2 - kernels not in params
# ________________________________________________________________________________________________________________________
# """
#
# LR_param_grid={'alpha': GS_VALUES['regul_range']}
# KRR_param_grid={'alpha': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                                                 'coef0' : GS_VALUES['coef0_range']}
# SVR_param_grid={'C': GS_VALUES['regul_range'], 'gamma': GS_VALUES['influence_range'], 'degree' : GS_VALUES['degree'],
#                 'epsilon': GS_VALUES['margin_range'],  'coef0' : GS_VALUES['coef0_range']}
#
# """
# ________________________________________________________________________________________________________________________
# HYPERPARAM
# ________________________________________________________________________________________________________________________
# """
#
# # # Example for Single Hyperparameter plot
# KRR_param_grid1={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['linear']}
# KRR_param_grid2={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['polynomial']}
# KRR_param_grid3={'gamma': list(10.0 ** np.arange(-3, 3)), 'kernel':['rbf']}
#
# """
# steps :
#
# ------------------------------------------------------------------------------------------------------------------------
# Clean data:
#
# Filter the CSTB dataframe using df_id_operation_valid, so only valid entries remain
# filtering away low quality cases as defined by CSTB
# #Remove cases where "type_structure_principale"
# Drop all rows which have a missing value
# #Remove cases where "type_structure_principale" is 5 (invalid entry)
# ------------------------------------------------------------------------------------------------------------------------
# Rename directly transferable data:
#
# "nb_occupant" to "bldg_users_total"
# "nb_niv_surface" to "bldg_floors_ag"
# "nb_niv_ssol" to "bldg_floors_bg"
# "sdp" to "bldg_area_hfa"
# "niveau_energie" to "bldg_energy_class_country"
# "cef" to "inv_energy_consumption"
# "periode_etude_reference" to "lca_RSP"
# "indicateur_1" to "GHG_sum_em"
#
# ------------------------------------------------------------------------------------------------------------------------
# "indicateur_1" : Potentiel de réchauffement climatique [GWP)[kg éq. CO2]
# remove operational energy and water emissions from total to get sum embodied:
# df_CSTB["GHG_sum_em"] = df_CSTB["GHG_sum_em"]-df_CSTB["IND1_ENE"]-df_CSTB["IND1_EAU"]-df_CSTB["IND1_LOT1"]-df_CSTB["IND1_LOT14"]
#
# ------------------------------------------------------------------------------------------------------------------------
# Translate and regroup/rename data
#
# "usage_principal" to "bldg_use_subtype"
#
# "type_toiture" to "bldg_roof_type"
# '3 pans et plus' to'Other'
# '2 pans','Gable or saddle roof'
# 'Terrasse','Flat roof'
# 'Monopente','Single pitched roof'
#
# "type_travaux" to "bldg_project_status"
# 'Bâtiments neufs','New Built'
# 'Extensions ou surélévations','Renovation'
#
# "date_etude_rsenv" to "bldg_year_complete_interval"
#
# ------------------------------------------------------------------------------------------------------------------------
# Transform and derive data through inference
#
# "type_structure_principale" and "materiau_principal" to "bldg_struct_type"
#
# 'Maçonnerie Terre cuite','massive brick'
# 'Voiles porteurs Béton','Voiles porteurs Béton haute performance'],'massive concrete'
# 'Maçonnerie Béton','massive brick'
# 'Maçonnerie Autre, à préciser','massive brick'
# 'Ossature Bois massif','frame wood'
# 'Poteaux/poutres Béton','Poteaux/poutres Béton haute performance'],'frame concrete'
# 'Maçonnerie Béton haute performance','massive brick'
# 'Poteaux/poutres Autre, à préciser','other'
# 'Poteaux/poutres Mixte: bois-béton','mix concrete wood'
# 'Ossature Bois massif reconstitué','frame wood'
# 'Ossature Mixte: bois-béton','frame concrete/wood'
# 'Poteaux/poutres Mixte: béton-acier','frame concrete/steel'
# 'Poteaux/poutres Bois massif','Poteaux/poutres Bois massif reconstitué'],'frame wood'
# 'Maçonnerie Bois massif reconstitué','Maçonnerie Mixte: bois-béton','Maçonnerie Bois massif'],'massive brick'
# 'Ossature Acier','frame steel'
# 'Voiles porteurs Béton cellulaire','massive concrete'
# 'Ossature Béton','frame concrete'
# 'Maçonnerie Pierre','massive brick'
# 'Voiles porteurs Terre cuite','massive brick'
# 'Maçonnerie Béton cellulaire','massive brick'
# 'Maçonnerie Bois massif','massive brick'
# 'Ossature Terre cuite','massive brick'
# 'Voiles porteurs Bois massif reconstitué','Voiles porteurs Bois massif'],'massive wood'
#
# "materiau_principal" to "inv_mat_1_type"
#
# 'Terre cuite','Ceramics (e.g., fired clay bricks)'
# 'Béton','Concrete w/o reinforcement'
# 'Pierre','Stone (granite, limestone, etc)'
# 'Autre, à préciser','Other'
# 'Bois massif','Timber, wood'
# 'Béton haute performance','Concrete w/o reinforcement'
# 'Mixte: bois-béton','Other'
# 'Bois massif reconstitué','Timber, wood'
# 'Mixte: béton-acier','Other'
# 'Acier','Metals (iron, steel)'
# 'Béton cellulaire','Other'
#
#
#
# #Infer bldg_use_type (Building type)
# #Copy "bldg_use_subtype"
# df_CSTB['bldg_use_type'] = df_CSTB['bldg_use_subtype']
#
# #Replace entries with infered type
#
# 'bldg_use_subtype' :
# 'Residential'
# 'Single family house' : 341
# 'Multi-family house' : 114
#
# 'Non-residential' :
# 'Office': 15
# 'School and Daycare': 14
# 'Hospital and Health' : 2
#
# df_CSTB['scope_handling_D'] = 'separately considered'
#
# #Infer GHG_sum_m2a (Sum)
# df_CSTB['GHG_sum_em_m2a'] = df_CSTB['GHG_sum_em']/df_CSTB['bldg_area_gfa']/df_CSTB['lca_RSP']
#
# #Infer GHG_sum_em_m2a (Sum Operational)
# df_CSTB['GHG_sum_op_m2a'] = df_CSTB['co2']
#
# ------------------------------------------------------------------------------------------------------------------------
# "Lots" decoding work
# #Use the "lots" to calculate the impact of individual building parts
#
# # "indicateur_1" to "GHG_sum_em"
#
# # "GHG_P1_sum_m2a" > Ground = Lot2: Foundations and infrastructure
# # 'GHG_P2_sum_m2' > Structure = Lot3: Structure and masonry + Lot4: Roof and cover
# # "GHG_P34_sum_m2a" > Envelope = Lot6: Exterior surfaces (facades), components (doors and windows) and joineries
# # "GHG_P4_sum_m2a" > Internal = Lot5: Interior partitions, suspended ceilings, components and joineries + Lot7: Interior coatings (floors, walls and ceilings)
# # "GHG_P56_sum_m2a" > Services =  Lot8: HVAC equipment + Lot10: Electrical equipment + Lot11: Special electrical equipment (systems, controls and communication) + Lot13: Local
# # "GHG_P78_sum_m2a" > Appliances = Lot9: Sanitary installations
#
#
# # a = 50 years > 'GHG_P2_sum_m2' = 50 * 'GHG_P2_sum_m2a'
# # https://github.com/mroeck/Embodied-Carbon-of-European-Buildings-Database/blob/develop/01_Preproc_CSTB.ipynb
#
# ------------------------------------------------------------------------------------------------------------------------
# Add empty data columns
# Rearrange columns (drop columns)
#
# Stages included in target
# We need to build a string based on:
# - Ground (1) (i.e. substructure, foundation, basement walls, etc.)
# - Load-bearing structure (2) (i.e. structural frame, walls, floors, roofs, etc.)
# - Envelope (3, 4) (i.e. openings, ext. finishes, etc.)
# - Internal (4) (i.e. partitions, int. finishes, etc.)
# - Services (5,6) (i.e. mechanical, electrical, renew. energy, etc.)
# - Appliances (7,8) (i.e. fixed facilities, mobile fittings, etc.)
#
#
# """
#
#
#
