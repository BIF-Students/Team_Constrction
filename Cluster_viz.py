# This file is used to visualize the clusters in spiderweb diagrams / radar charts to describe/name the clusters

import pandas as pd
import numpy as np
from helpers.student_bif_code import *
from helpers.cluster_visualizations import *
from helpers.metrics2 import *
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo

# Load relevant dataframes and merge
data = pd.read_csv("C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv", sep = ",", encoding='unicode_escape')
viz = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv', sep=",", encoding='unicode_escape')
data = pd.merge(data, viz, on=['playerId', 'seasonId'])

# Drop irrelevant features
data = data.drop(['x', 'y'], axis=1)
data = data.drop(data.filter(like='_vaep').columns, axis=1)

# Get percentile ranks per cluster
data = data[data['ip_cluster'] != -1]
ids = data.iloc[:, np.r_[0:4]]
test = data.iloc[:, np.r_[4:47]]
test = test.rank(pct = True)
test = pd.concat([ids.reset_index(drop=True),test.reset_index(drop=True)], axis=1)

# Plot spiderweb diagrams for all clusters withing the 6 feature groupings defined in metrics2.py
make_spider_web(test, finishing, "Finishing")
make_spider_web(test, takeon, "Take-on")
make_spider_web(test, passingA, "Passing (Attacking Play)")
make_spider_web(test, passingE, "Passing (Established Play)")
make_spider_web(test, defending, "Defending")
make_spider_web(test, zones, "Zones")

# Create Pareto chart for specific cluster(s) to gauge which features they have the highest volume/tendency off
pareto(test, 0)

# Get means for cluster X
check = stat_comp(test)
check2 = pos_dist(test, 0)

# Check player names in designated clusters to understand what types of players a cluster has
players = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Players", db_name='Scouting')
players = players[['playerId', 'shortName']]
dfp = pd.merge(players, test, on='playerId')
dfp = dfp.loc[(dfp['seasonId'] == 186810) | (dfp['seasonId'] == 187475)]
check3 = dfp[dfp.ip_cluster == 17]