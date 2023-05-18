import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helpers.helperFunctions import *
from helpers.student_bif_code import *

# loading
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv', sep=",", encoding='unicode_escape')
dat = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', sep=",", encoding='unicode_escape')
df = pd.merge(df, dat, on=['playerId', 'seasonId'])
df = df.drop(['x', 'y'], axis=1)

# creating tendency and vaep dataframes
df_tendency = df.drop(df.filter(like='_vaep').columns, axis=1)
df_vaep = df.drop(df.filter(like='_tendency').columns, axis=1)

# creating weight dictionaries as input
clusters = df_vaep['ip_cluster']
ids = df_vaep[['playerId', 'seasonId', 'pos_group']]
df_input = df_tendency.drop(['ip_cluster', 'playerId', 'seasonId', 'pos_group'], axis=1)
df_input2 = df_vaep.drop(['ip_cluster', 'playerId', 'seasonId', 'pos_group'], axis=1)
weight_dictsX = get_weight_dicts(df_input, clusters)
weight_dicts = get_weight_dicts2(df_input, clusters)

df_elo = load_db_to_pd(sql_query = "SELECT * FROM League_Factor", db_name='Scouting')
df_ssn = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Seasons", db_name='Scouting')
df_comp = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Competitions", db_name='Scouting')
df_elo = pd.merge(df_elo, df_ssn, on='seasonId', how='left')
df_elo = pd.merge(df_elo, df_comp, on='competitionId', how='left')
df_elo = df_elo[['name_y', 'leagueFactor', 'date']]
df_elo['date'] = pd.to_datetime(df_elo['date'])
df_elo = df_elo[df_elo['date'].dt.year == 2021]
df_elo.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/ELO.csv', index=False)

# testing
cluster_name = 'Cluster 6'
cluster_df = cluster_to_dataframe(weight_dicts, cluster_name)
plot_sorted_bar_chart(cluster_df)

# calculating scores and visualizing
dfp = calculate_weighted_scores2(df_vaep, weight_dicts)
players = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Players", db_name='Scouting')
players = players[['playerId', 'shortName', 'birthDate']]
# dfp = dfp[dfp['seasonId'].isin([187483, 187141, 187142])] # when measuring against the Superliga players
test = perf(dfp, players, mode='scaled', cluster=None, age=None) # use for presentation
create_boxplot(test)
plot_sorted_bar_chart_p(test, "S. Hedlund")

# to be merged with transfer values from transfermarkt at a later point
tm = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Transfermarkt_Players", db_name='Scouting_Raw')

# brøndby validation
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