import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
from helpers.helperFunctions import *
from sklearn.mixture import GaussianMixture
from helpers.student_bif_code import *
from matplotlib.lines import Line2D

# loading
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv',
                 sep=",", encoding='unicode_escape')

df_posmin = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Positions_Minutes", db_name='Scouting')
df_pos = df_posmin.drop(['matchId', 'teamId', 'time'], axis=1)
df_pos = df_pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
df = pd.merge(df, df_pos, on=['playerId', 'seasonId'])

# saving IDs
df_id = df[['playerId', 'seasonId', 'position']]
df = df.drop(['playerId', 'seasonId', 'position'], axis=1)
df = df.drop(df.filter(like='_vaep').columns, axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(df)

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "position")
plt.show()

# optimal GMM model
opt_clus(dr)

# clustering
gmm = GaussianMixture(n_components=11, covariance_type='full', random_state=42).fit(dr)
probs = gmm.predict_proba(dr)
threshold = 0.7
cluster_assignments = np.argmax(probs, axis=1)
cluster_assignments[probs.max(axis=1) < threshold] = -1
gmm_to_df(cluster_assignments, "ip").value_counts()

# visualization cluster
x = dr[:, 0]
y = dr[:, 1]

colors = plt.cm.viridis(np.linspace(0, 1, 11)).tolist()
plt.scatter(x, y, c=[colors[l] if l != -1 else 'lightgray' for l in cluster_assignments], cmap='viridis')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.3, zorder=1)
plt.show()

# merging
df2 = pd.concat([df.reset_index(drop=True),gmm_to_df(cluster_assignments, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(cluster_assignments, "ip").reset_index(drop=True)], axis=1)
# df2_id = pd.concat([df2_id.reset_index(drop=True),df.reset_index(drop=True)], axis=1)

# creating DFs
c0 = df2[df2.ip_cluster == 0]
c1 = df2[df2.ip_cluster == 1]
c2 = df2[df2.ip_cluster == 2]
c3 = df2[df2.ip_cluster == 3]
c4 = df2[df2.ip_cluster == 4]

# dropping cluster
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)
c3 = c3.drop(['ip_cluster'], axis=1)
c4 = c4.drop(['ip_cluster'], axis=1)

# creating ID DFs
c0_id = df2_id[df2_id.ip_cluster == 0]
c1_id = df2_id[df2_id.ip_cluster == 1]
c2_id = df2_id[df2_id.ip_cluster == 2]
c3_id = df2_id[df2_id.ip_cluster == 3]
c4_id = df2_id[df2_id.ip_cluster == 4]

# dropping cluster
c0_id = c0_id.drop(['ip_cluster'], axis=1)
c1_id = c1_id.drop(['ip_cluster'], axis=1)
c2_id = c2_id.drop(['ip_cluster'], axis=1)
c3_id = c3_id.drop(['ip_cluster'], axis=1)
c4_id = c4_id.drop(['ip_cluster'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr0_ip = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(c0)
dr1_ip = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(c1)
dr2_ip = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(c2)
dr3_ip = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(c3)
dr4_ip = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=2, random_state=42).fit_transform(c4)

# when n_components=2
test = pd.DataFrame(dr0_ip, columns=["x", "y"])
test = c0_id.join(test)
plot = sns.scatterplot(data=test, x="x", y="y", hue = "pos_group")
plt.show()
test.groupby(['map_group'], as_index=False).count()

# creating clusters
opt_clus(dr0_ip) # running for 0-5
gmm0_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr0_ip).predict(dr0_ip)
gmm1_ip = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(dr1_ip).predict(dr1_ip)
gmm2_ip = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(dr2_ip).predict(dr2_ip)
gmm3_ip = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(dr3_ip).predict(dr3_ip)
gmm4_ip = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(dr4_ip).predict(dr4_ip)

# visualizing to check
plt.scatter(dr1_ip[:, 0], dr1_ip[:, 1], c=gmm1_ip, s=40, cmap='viridis')
plt.show()

# merging into df
c0_id = pd.concat([c0_id.reset_index(drop=True),gmm_to_df(gmm0_ip, "ip").reset_index(drop=True)], axis=1)
c1_id = pd.concat([c1_id.reset_index(drop=True),gmm_to_df(gmm1_ip, "ip").reset_index(drop=True)], axis=1)
c2_id = pd.concat([c2_id.reset_index(drop=True),gmm_to_df(gmm2_ip, "ip").reset_index(drop=True)], axis=1)
c3_id = pd.concat([c3_id.reset_index(drop=True),gmm_to_df(gmm3_ip, "ip").reset_index(drop=True)], axis=1)
c4_id = pd.concat([c4_id.reset_index(drop=True),gmm_to_df(gmm4_ip, "ip").reset_index(drop=True)], axis=1)

# adjusting clust no.
c1_id['ip_cluster'] = c1_id['ip_cluster'] + 3
c2_id['ip_cluster'] = c2_id['ip_cluster'] + 5
c3_id['ip_cluster'] = c3_id['ip_cluster'] + 7
c4_id['ip_cluster'] = c4_id['ip_cluster'] + 9

# val
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
frames = [c0_id, c1_id, c2_id, c3_id, c4_id]
val2 = pd.concat(frames)
val_df = pd.merge(val, val2, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)

# export
stats = [c0, c1, c2, c3, c4]
stats2 = pd.concat(stats)
test = pd.concat([val2.reset_index(drop=True),stats2.reset_index(drop=True)], axis=1)
test.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/clustersTEST.csv', index=False)