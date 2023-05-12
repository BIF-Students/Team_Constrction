import pandas as pd
import plotly.graph_objects as go
import plotly.offline as pyo
# from helpers.metrics import *
from helpers.metrics2 import *
import plotly.express as px
import numpy as np
import helperFunctions
from helpers.helperFunctions import *

#Unused imports
'''
import numpy as np
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import seaborn as sns
import plotly.express as px
'''


# --------------------------------------------------------
# Define pitch with predeefiend rows

# Get individuals scores in subcategories
data = pd.read_csv("C:/ITU/ITU_Research_Project/clustered_data/clusters.csv", sep = ",", encoding='unicode_escape')
data = pd.read_csv("C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/clustersTEST.csv", sep = ",", encoding='unicode_escape')

#Function to produce averages for each cluster for specified values in arguemnt
def get_stat_values (data, metric):
    clusters = data.ip_cluster.unique()
    final_frame = pd.DataFrame()
    data = data[metric]
    for cluster in clusters:
        cluster_frame = data[data['ip_cluster'] == cluster]
        frame = cluster_frame.loc[:, cluster_frame.columns != 'ip_cluster']
        transposed = frame.T
        transposed['vals'] = transposed.mean(axis =1)
        transposed['ip_cluster'] = cluster
        vals_clusters = transposed[['vals', 'ip_cluster']]
        vals_clusters_labels = vals_clusters.rename_axis("labels").reset_index()
        vals_clusters_labels.columns.values[0] = "labels"
        final_frame = pd.concat([final_frame, vals_clusters_labels])
    return final_frame

#Sum scores per key in dict per player
def compute_sum_per_metric(data, dict):
    for key, val in dict.items():
        h = val
        h.remove('ip_cluster')
        data[key] = data[h].sum(axis=1)
    return data

# data = compute_sum_per_metric(data, dict_lists)



def get_avg(df):
    averages = pd.DataFrame(columns=['labels', 'vals'])
    labels = df.labels.unique()
    for label in labels:
        label_df = df[df['labels'] == label]
        avg_df = label_df['vals'].mean()
        data = [[label, avg_df]]
        df_done = pd.DataFrame(data, columns=['labels', 'vals'])
        averages = pd.concat([averages, df_done])
    return averages



#Function to create spiderweb plot for spcified stats
def make_spider_web(raw_data, stat, title_att):
  stat_vals = get_stat_values(raw_data, stat)
  averages_found = get_avg(stat_vals)
  clusters = stat_vals.ip_cluster.unique()
  avg_cluster = max(clusters) +1
  averages_found['ip_cluster'] = avg_cluster
  clusters.sort()

  fig = go.Figure(layout=go.Layout(
      title=go.layout.Title(text='Comparison clusters - ' + title_att ),
      polar={'radialaxis': {'visible': True}},
      showlegend=True
  ))

  fig.add_trace(
      go.Scatterpolar(r=averages_found.vals, fill='toself', opacity=0.4, theta=averages_found.labels, name="# AVG " + str(avg_cluster)),
  )

  for cluster in clusters:
      cluster_df = stat_vals[stat_vals['ip_cluster'] == cluster]
      frame = cluster_df.loc[:, cluster_df.columns != 'ip_cluster']
      fig.add_trace(
          go.Scatterpolar(r=frame.vals, fill='toself', opacity=0.4, theta=frame.labels, name="# Cluster " + str(cluster)),
      )

  pyo.plot(fig)

#Spider web with closed lines
def make_spider_web_v2(raw_data, stat, title_att):
      stat_vals = get_stat_values(raw_data, stat)
      df_visual = get_avg(stat_vals)
      # averages_found['ip_clusters'] =
      clusters = stat_vals.ip_cluster.unique()
      df_visual['cluster'] = 'AVG'
      clusters.sort()
      for cluster in clusters:
          cluster_df = stat_vals[stat_vals['ip_cluster'] == cluster]
          frame = cluster_df.loc[:, cluster_df.columns != 'ip_cluster']
          frame['cluster'] = cluster
          df_visual = pd.concat([df_visual, frame])
      fig = px.line_polar(df_visual, r="vals", theta="labels", color="cluster", line_close=True,
                          color_discrete_sequence=px.colors.sequential.Plasma_r,
                          template="plotly_dark", )
      fig.show()


def pareto(df, cluster):
    df = df.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)
    df = df.groupby(['ip_cluster'], as_index=False).mean()
    df = df[df.ip_cluster == cluster]
    df = df.drop(['ip_cluster'], axis=1)
    df = df.T
    df.rename(columns={df.columns[0]: "value"}, inplace=True)
    df = df.sort_values('value', ascending=False)

    fig = px.bar(df, x=df.index, y=df['value'])
    fig.add_hline(y=0.5)
    fig.show()

def stat_comp(df):
    df = df.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)
    df = df.groupby(['ip_cluster'], as_index=False).mean()
    return df

def pos_dist(df, cluster):
    df = df.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
    df = df[df.ip_cluster == cluster]
    df = df.drop(['ip_cluster'], axis=1)
    df = df.groupby(['map_group'], as_index=False).count()
    return df


# getting percentiles
ids = data.iloc[:, np.r_[0:5]]
test = data.iloc[:, np.r_[5:26]]
test = test.rank(pct = True)
test = pd.concat([ids.reset_index(drop=True),test.reset_index(drop=True)], axis=1)
test = compute_sum_per_metric(test, dict_lists)

test2 = test.iloc[:, np.r_[25:30]]
test2 = test2.rank(pct = True)
test2 = pd.concat([ids.reset_index(drop=True),test2.reset_index(drop=True)], axis=1)

#Plotting spiderwebs
make_spider_web(data, finishing, "Finishing")
make_spider_web(data, creating, "Creating")
make_spider_web(data, progression, "Progression")
make_spider_web(data, established, "Established")
make_spider_web(data, duels, "Duels")
make_spider_web(data, game_reading, "Game Reading")
make_spider_web(test2, categories, "Categories")

make_spider_web(test, attacking, "Attacking")
make_spider_web(test, possession, "Possession")
make_spider_web(test, defending, "Defending")
pareto(test, 0)
check = stat_comp(test)
check2 = pos_dist(test, 5)

players = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/Wyscout_Players.csv', sep=";", encoding='unicode_escape')
players.drop(columns=players.columns[0], axis=1, inplace=True)
dfp = pd.merge(players, test, on='playerId')
dfp = dfp[dfp.ip_cluster == 1]
check3 = dfp.iloc[:, np.r_[1, 15, 16:37]]

# validate clusters
xyz = names_clusters(test, 1)
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
xyz = data.iloc[:, np.r_[0, 5]]
val_df = pd.merge(val, xyz, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)

#-------------------------------------- Unused code atm ----------------------------------------------#

#Not used at, kept in case of potential future usage

'''
def get_cluster_scores(data):
   DEF_score = (data[(data.labels == 'ground_duels') | (data.labels == 'air_duels') | (data.labels == 'clearance') | (data.labels == 'game_reading') | (data.labels == 'disciplinary')]).groupby(['cluster']).sum()
   DEF_score['labels'] = "DEF"
   #DEF_score.index.name = 'cluster'
   #DEF_score.reset_index(inplace=True)

   POS_score = (data[ (data.labels == 'misc') | (data.labels == 'progression') | (data.labels == 'established')]).groupby(['cluster']).sum()
   POS_score['labels'] = "POS"
  # DEF_score.index.name = 'cluster'
   #DEF_score.reset_index(inplace=True)

   ATT_score = (data[(data.labels == 'finishing') | (data.labels == 'movement') | (data.labels == 'creating')]).groupby(['cluster']).sum()
   ATT_score['labels'] = "ATT"
#  DEF_score.index.name = 'cluster'
#  DEF_score.reset_index(inplace=True)

   list_of_frames = [DEF_score, POS_score, ATT_score]
   df_final = pd.concat(list_of_frames)

   df_final.index.name = 'cluster'
   df_final.reset_index(inplace=True)

   return df_final
def getdata():
    DEF = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/DEF.csv', sep=",", encoding='unicode_escape')
    MID = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/MID.csv', sep=",", encoding='unicode_escape')
    WIDE = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/WIDE.csv', sep=",", encoding='unicode_escape')
    ATT = pd.read_csv('C:/ITU/ITU_Research_Project/clustered_data/ATT.csv', sep=",", encoding='unicode_escape')
    raw = pd.read_csv("C:/ITU/ITU_Research_Project/preprocessed/events_CN.csv", sep=",", encoding='unicode_escape')

    DEF_v2 = pd.merge(DEF, raw, on=['playerId', 'seasonId', 'map_group', 'pos_group'])
    MID_v2 = pd.merge(MID, raw, on=['playerId', 'seasonId', 'map_group', 'pos_group'])
    WIDE_v2 = pd.merge(WIDE, raw, on=['playerId', 'seasonId', 'map_group', 'pos_group'])
    ATT_v2 = pd.merge(ATT, raw, on=['playerId', 'seasonId', 'map_group', 'pos_group'])

    DEF_v3 = DEF_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
    MID_v3 = MID_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
    ATT_v3 = ATT_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)
    WIDE_v3 = WIDE_v2.drop(['playerId', 'seasonId', 'map_group', 'pos_group', 'teamId'], axis=1)

    return DEF_v3, MID_v3, WIDE_v3, ATT_v3
def get_cluster_sub_scores(data):
    clusters = data.cluster.unique()
    final_frame = pd.DataFrame()
    for cluster in clusters:
        cluster_frame = data[data['cluster'] == cluster]
        vals = get_scores(cluster_frame)
        frame_made = pd.DataFrame((((pd.DataFrame.from_records(vals)).sum())))
        frame_made = frame_made.rename_axis("labels").reset_index()
        frame_made.columns.values[1] = "values"
        frame_made['cluster'] = cluster
        final_frame = pd.concat([final_frame, frame_made])

    return final_frame
def get_scores(cluster_frame):
    def_scores = {'ground_duels': cluster_frame[ground_duels].sum().sum(),
                  'air_duels': cluster_frame[air_duels].sum().sum(),
                  'clearance': cluster_frame[clearance].sum().sum(),
                  'game_reading': cluster_frame[game_reading].sum().sum(),
                  'disciplinary': cluster_frame[disciplinary].sum().sum(),
                  }

    pos_scores = {'established': cluster_frame[established].sum().sum(),
                  'progression': cluster_frame[progression].sum().sum(),
                  'misc': cluster_frame[misc].sum().sum(),
                  }

    att_scores = {'finishing': cluster_frame[finishing].sum().sum(),
                  'creating': cluster_frame[creating].sum().sum(),
                  'movement': cluster_frame[movement].sum().sum(),
                  }
    return def_scores, pos_scores, att_scores


# Extract data on each position group
DEF, MID, WIDE, ATT = getdata()

df = pd.DataFrame(dict(
    r=[1, 5, 2, 2, 3],
    theta=['processing cost','mechanical properties','chemical stability',
           'thermal stability', 'device integration']))

fig = px.line_polar(df, r='r', theta=finishing, line_close=True)
fig.show()



# Get summarized sub scores for each cluster divided by positional groups
def_sub_scores_per_cluster = get_cluster_sub_scores(DEF)

mid_sub_scores_per_cluster = get_cluster_sub_scores(MID)
wide_sub_scores_per_cluster = get_cluster_sub_scores(WIDE)
att_sub_scores_per_cluster = get_cluster_sub_scores(ATT)

def_scores = get_cluster_scores(def_sub_scores_per_cluster)
mid_scores = get_cluster_scores(mid_sub_scores_per_cluster)
wide_scores = get_cluster_scores(wide_sub_scores_per_cluster)
att_scores = get_cluster_scores(att_sub_scores_per_cluster)

sns.barplot(data=def_scores, x="labels", y="values")
plt.show()

sns.kdeplot(data=df1, x="values")
plt.show()


sns.barplot(data=def_scores, x="labels", y="values", hue='cluster')
plt.show()

def createPitchWithZones():
    data = np.array([
        [0, 0],
        [16, 0],  # x+16
        [0, 19],
        [16, 19],  # x+17
        [33, 0],
        [33, 19],
        [67, 0],  # x+34
        [67, 19],
        [84, 0],
        [84, 19],
        [100, 0],
        [100, 19],
        [0, 81],
        [0, 100],
        [16, 81],
        [16, 100],
        [33, 81],
        [33, 100],
        [67, 81],  # x+34
        [67, 100],
        [84, 81],
        [84, 100],
        [100, 81],
        [100, 100],
        [16, 63],
        [16, 63],
        [16, 63],
        [33, 63],
        [67, 63],
        [16, 37],
        [33, 37],
        [50, 19],
        [50, 0],
        [50, 37],
        [50, 63],
        [50, 81],
        [50, 100],
        [67, 37],
        [84, 37],
        [84, 63], ])
    pitch = Pitch(pitch_type='wyscout', axis=True,
                  positional=True,
                  tick=True,
                  label=True)
    fig, ax = pitch.draw()
    x, y = data.T
    plt.scatter(x, y)
    plt.show()
'''

def find_zone_chemistry(row):
    s = ""  # Initialize variable s to an empty string
    x = row['x']  # Extract the value of x from the input row
    y = row['y']  # Extract the value of y from the input row

    # Check which zone the (x, y) pair belongs to
    if x >= 0 and x < 50 and y >= 0 and y < 33:
        s = 1  # Assign s to 1 if the pair belongs to zone 1
    elif x >= 50 and x <= 100 and y >= 0 and y < 33:
        s = 4  # Assign s to 4 if the pair belongs to zone 4
    elif x >= 0 and x < 50 and y >= 33 and y < 66:
        s = 2  # Assign s to 2 if the pair belongs to zone 2
    elif x >= 50 and x <= 100 and y >= 33 and y < 66:
        s = 5  # Assign s to 5 if the pair belongs to zone 5
    elif x >= 0 and x < 50 and y >= 66 and y <= 100:
        s = 3  # Assign s to 3 if the pair belongs to zone 3
    elif x >= 50 and x <= 100 and y >= 66 and y <= 100:
        s = 6  # Assign s to 6 if the pair belongs to zone 6
    else:
        s = 0  # Assign s to 0 if the (x, y) pair doesn't belong to any zone

    return s


#Shows roc curve for each class and labels the class and teh auc score
def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest):
        # Creating visualization with the readable labels
        visualizer = ROCAUC(model)

        # Fitting to the training data first then scoring with the test data
        visualizer.fit(xtrain, ytrain)
        visualizer.score(xtest, ytest)
        visualizer.show()