import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from helpers.student_bif_code import *
from helpers.helperFunctions import *

# loading
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv',
                 sep=",", encoding='unicode_escape')
df = df.drop(['Zone 0 Actions', 'Zone 0 Actions_vaep'], axis=1)

# tjek positions_minutes (om der er mangler)
df_posmin = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Positions_Minutes", db_name='Scouting')
df_pos = df_posmin.drop(['matchId', 'teamId', 'time'], axis=1)
df_pos = df_pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
df = pd.merge(df, df_pos, on=['playerId', 'seasonId'])
df['pos_group'] = df.apply(lambda row: pos_group(row), axis=1)

# saving IDs
df_id = df[['playerId', 'seasonId', 'pos_group']]
df = df.drop(['playerId', 'seasonId', 'position', 'pos_group'], axis=1)
df = df.drop(df.filter(like='_vaep').columns, axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
test = df[['goal_tendency', 'head_shot_tendency', 'touch_in_box_tendency', 'shots_PA_tendency', 'Zone 6 Actions_tendency']]
dr = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(test)

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# optimal GMM model
opt_clus(dr)

# pca comparison
pca = PCA(n_components=0.8)
trans = pca.fit_transform(dr)
opt_clus(trans)

# clustering
clusters = 11
gmm = GaussianMixture(n_components=clusters, covariance_type='full', random_state=42).fit(dr)
probs = gmm.predict_proba(dr)
threshold = 0.7 # found via opt_clust
cluster_assignments = np.argmax(probs, axis=1)
cluster_assignments[probs.max(axis=1) < threshold] = -1
gmm_to_df(cluster_assignments, "ip").value_counts()

# visualization cluster
x = dr[:, 0]
y = dr[:, 1]

colors = plt.cm.viridis(np.linspace(0, 1, clusters)).tolist()
plt.scatter(x, y, c=[colors[l] if l != -1 else 'lightgray' for l in cluster_assignments], cmap='viridis')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.25, zorder=1)
plt.show()

# merging
df2 = pd.concat([df.reset_index(drop=True),gmm_to_df(cluster_assignments, "ip").reset_index(drop=True)], axis=1)
df2 = pd.concat([df_id.reset_index(drop=True),df2.reset_index(drop=True)], axis=1)

# export
df2.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv', index=False)