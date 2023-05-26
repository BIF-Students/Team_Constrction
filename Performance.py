# This file contains the code used for creating the composite player role performance index along with the views of it

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helpers.helperFunctions import *
from helpers.student_bif_code import *

# Read data from CSV files
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv', sep=",", encoding='unicode_escape')
dat = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', sep=",", encoding='unicode_escape')

# Merge dataframes based on playerId and seasonId columns
df = pd.merge(df, dat, on=['playerId', 'seasonId'])

# Drop 'x' and 'y' columns from df dataframe
df = df.drop(['x', 'y'], axis=1)

# Create df_tendency dataframe by dropping columns with '_vaep' in their names from df dataframe
df_tendency = df.drop(df.filter(like='_vaep').columns, axis=1)

# Create df_vaep dataframe by dropping columns with '_tendency' in their names from df dataframe
df_vaep = df.drop(df.filter(like='_tendency').columns, axis=1)

# Extract specific columns from df_vaep dataframe and remove columns from df_tendency and df_vaep dataframes
clusters = df_vaep['ip_cluster']
ids = df_vaep[['playerId', 'seasonId', 'pos_group']]
df_input = df_tendency.drop(['ip_cluster', 'playerId', 'seasonId', 'pos_group'], axis=1)
df_input2 = df_vaep.drop(['ip_cluster', 'playerId', 'seasonId', 'pos_group'], axis=1)

# Call functions to get weight dictionaries
weight_dictsX = get_weight_dicts(df_input, clusters)
weight_dicts = get_weight_dicts2(df_input, clusters)

# Assign cluster name and create cluster_df dataframe based on weight dictionaries and cluster name
cluster_name = 'Cluster 6'
cluster_df = cluster_to_dataframe(weight_dicts, cluster_name)

# Plot sorted bar chart for feature weights using cluster_df dataframe
plot_sorted_bar_chart(cluster_df)

# Calculate weighted scores using df_vaep dataframe and weight dictionaries
dfp = calculate_weighted_scores2(df_vaep, weight_dicts)

# Load player data from database into players dataframe and filter dfp dataframe for specific seasonIds
players = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Players", db_name='Scouting')
players = players[['playerId', 'shortName', 'birthDate']]

# Perform performance analysis using dfp, players, and other parameters. Plot bar chart for selected players' performance
dfp = dfp[dfp['seasonId'].isin([187483, 187141, 187142])] # when measuring against the Superliga players
test = perf(dfp, players, mode='scaled', cluster=None, age=None) # use for presentation
plot_sorted_bar_chart_p(test, "M. Uhre")

# Assign a filtered subset of 'test' dataframe to 'test2' based on specific seasonId values and create boxplot
test2 = test[test.seasonId.isin([187534,187527,187526,187533,187530,187450,187523,187168,187475,187167,187570,187425,187452,187483,187507,187554,187479,187512,187491,187502,187471,187141,186215,187142,187401,187139,187145,187844,187276,187376,187618,187379,187696,187144,187065,187064,187504,187389,187448,187392,187393,187532,187482,187374,187459,187511,187555,187164,187405,187432,187434,187560,187624,187412,187443,187409,187575,187101,187420,187557,187927,187921,188267,187381])]
create_boxplot(test2)

# To be merged with transfer values from transfermarkt at a later point
tm = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Transfermarkt_Players", db_name='Scouting_Raw')

# Load Brøndby players' playerIds, merge with results, and export dataframe for validation with Brøndby staff
val = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/clusters_and_chem_AVG.csv', sep=";", encoding='unicode_escape')
val_input = test.copy()
# val_input = val_input[val_input['seasonId'] == 187483]
val = pd.merge(val, val_input, on='playerId', how='left')
val['top_cluster1_score'] = val.apply(lambda row: get_weighted_score(row, row['top_cluster1']), axis=1)
val['top_cluster2_score'] = val.apply(lambda row: get_weighted_score(row, row['top_cluster2']), axis=1)
val['top_cluster3_score'] = val.apply(lambda row: get_weighted_score(row, row['top_cluster3']), axis=1)
val = val.drop(columns=[f'Cluster {i} Weighted Score' for i in range(18)])
val = val.drop(['shortName_y', 'seasonId', 'playerId', 'pos_group_y', 'ip_cluster_y'], axis=1)
val.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/validation_set.csv', index=False)