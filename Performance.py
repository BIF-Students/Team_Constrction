import numpy as np
import pandas as pd
from helpers.helperFunctions import *

df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv',
                 sep=",", encoding='unicode_escape')

# creating weight dictionaries as input
clusters = df['ip_cluster']
df_input = df.drop(['ip_cluster', 'playerId', 'seasonId', 'position'], axis=1)
weight_dicts = get_weight_dicts(df_input, clusters)

# scaling weight dictionary from 0.5 to 2
weight_dicts = scale_weights(weight_dicts)

# testing/checking
cluster_name = 'Cluster 1'
cluster_df = cluster_to_dataframe(weight_dicts, cluster_name)
plot_sorted_bar_chart(cluster_df)

# applying weights
dfp = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv',
                 sep=",", encoding='unicode_escape')
dfp_id = dfp[['playerId', 'seasonId']]
dfp = dfp.drop(dfp.filter(like='_tendency').columns, axis=1)

dfp = calculate_weighted_scores(dfp, weight_dicts)
cols_to_rank = dfp.columns[dfp.columns.str.endswith('Weighted Score')]
dfp[cols_to_rank] = dfp[cols_to_rank].rank(pct=True)