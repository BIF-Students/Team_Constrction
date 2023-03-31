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
df_input = df_vaep.drop(['ip_cluster', 'playerId', 'seasonId', 'pos_group'], axis=1)
weight_dicts = get_weight_dicts(df_input, clusters)

# testing
cluster_name = 'Cluster 0'
cluster_df = cluster_to_dataframe(weight_dicts, cluster_name)
plot_sorted_bar_chart(cluster_df)

# applying weights
# ids = df_vaep[['playerId', 'seasonId', 'pos_group']]
# df_vaep = df_vaep.drop(['playerId', 'seasonId', 'pos_group', 'ip_cluster'], axis=1)

dfp = calculate_weighted_scores(df_vaep, weight_dicts)
dfp = dfp[dfp.filter(like='Weighted Score').columns]
# scale = MinMaxScaler(feature_range=(0,100))
# dfp[dfp.columns] = scale.fit_transform(dfp)
dfp = dfp.rank(pct = True) # use for presentation
dfp = round(dfp*100, 0) # same as line above

# validating
dfp = pd.concat([ids.reset_index(drop=True),dfp.reset_index(drop=True)], axis=1)
players = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Players", db_name='Scouting')
players = players[['playerId', 'shortName']]
dfp = pd.merge(players, dfp, on='playerId')

# merging with player value