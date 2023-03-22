import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import umap
import plotly.express as px
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from helpers.student_bif_code import *
from helpers.helperFunctions import *

# loading
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv',
                 sep=",", encoding='unicode_escape')

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
dr = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df)

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = pd.concat([df_id.reset_index(drop=True),dr2.reset_index(drop=True)], axis=1)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# when n_components=3
dr2 = pd.DataFrame(dr, columns=["x", "y", "z"])
dr2 = pd.concat([df_id.reset_index(drop=True),dr2.reset_index(drop=True)], axis=1)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='pos_group')
fig.show()

# optimal GMM model
opt_clus(dr)

# pca comparison
pca = PCA(n_components=0.8)
trans = pca.fit_transform(df)
opt_clus(trans)

# clustering
clusters = 6
gmm = GaussianMixture(n_components=clusters, covariance_type='full', random_state=42).fit(dr)
probs = gmm.predict_proba(dr)
threshold = 0.6 # found via opt_clust
cluster_assignments = np.argmax(probs, axis=1)
cluster_assignments[probs.max(axis=1) < threshold] = -1
gmm_to_df(cluster_assignments, "ip").value_counts()

# visualization cluster
x = dr[:, 0]
y = dr[:, 1]
z = dr[:, 2]

# when n=2
colors = plt.cm.viridis(np.linspace(0, 1, clusters)).tolist()
plt.scatter(x, y, c=[colors[l] if l != -1 else 'lightgray' for l in cluster_assignments], cmap='viridis')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.25, zorder=1)
plt.show()

# when n=3
col = pd.DataFrame(cluster_assignments, columns=["c"])
dr2 = pd.concat([dr2.reset_index(drop=True),col.reset_index(drop=True)], axis=1)
color_map = {'-1': 'lightgrey'}
dr2['c'] = dr2['c'].astype(str)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='c', color_discrete_map=color_map)
fig.update_traces(marker=dict(line=dict(width=0.001, color='white')))
fig.show()

# merging
df2 = pd.concat([df.reset_index(drop=True),gmm_to_df(cluster_assignments, "ip").reset_index(drop=True)], axis=1)
df2 = pd.concat([df_id.reset_index(drop=True),df2.reset_index(drop=True)], axis=1)

df2.groupby('ip_cluster')['pos_group'].value_counts()

# -----------------------------------------------------------------------------------

# creating DFs
cX = df2[df2.ip_cluster == -1]
c0 = df2[df2.ip_cluster == 0]
c1 = df2[df2.ip_cluster == 1]
c2 = df2[df2.ip_cluster == 2]
c3 = df2[df2.ip_cluster == 3]
c4 = df2[df2.ip_cluster == 4]
c5 = df2[df2.ip_cluster == 5]

# dropping cluster
cX = cX.drop(['ip_cluster'], axis=1)
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)
c3 = c3.drop(['ip_cluster'], axis=1)
c4 = c4.drop(['ip_cluster'], axis=1)
c5 = c5.drop(['ip_cluster'], axis=1)

# dropping cluster
cX_id = cX[['playerId', 'seasonId', 'pos_group']]
cX = cX.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c0_id = c0[['playerId', 'seasonId', 'pos_group']]
c0 = c0.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c1_id = c1[['playerId', 'seasonId', 'pos_group']]
c1 = c1.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c2_id = c2[['playerId', 'seasonId', 'pos_group']]
c2 = c2.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c3_id = c3[['playerId', 'seasonId', 'pos_group']]
c3 = c3.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c4_id = c4[['playerId', 'seasonId', 'pos_group']]
c4 = c4.drop(['playerId', 'seasonId', 'pos_group'], axis=1)
c5_id = c5[['playerId', 'seasonId', 'pos_group']]
c5 = c5.drop(['playerId', 'seasonId', 'pos_group'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr0_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c0)
dr1_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c1)
dr2_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c2)
dr3_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c3)
dr4_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c4)
dr5_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=3, random_state=42).fit_transform(c5)

# creating clusters
# opt_clus(dr1_ip) # running for 0-5
gmm0_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr0_ip).predict_proba(dr0_ip)
ca0 = np.argmax(gmm0_ip, axis=1)
ca0[gmm0_ip.max(axis=1) < 0.6] = -1
gmm1_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr1_ip).predict_proba(dr1_ip)
ca1 = np.argmax(gmm1_ip, axis=1)
ca1[gmm1_ip.max(axis=1) < 0.6] = -1
gmm2_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr2_ip).predict_proba(dr2_ip)
ca2 = np.argmax(gmm2_ip, axis=1)
ca2[gmm2_ip.max(axis=1) < 0.6] = -1
gmm3_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr3_ip).predict_proba(dr3_ip)
ca3 = np.argmax(gmm3_ip, axis=1)
ca3[gmm3_ip.max(axis=1) < 0.6] = -1
gmm4_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr4_ip).predict_proba(dr4_ip)
ca4 = np.argmax(gmm4_ip, axis=1)
ca4[gmm4_ip.max(axis=1) < 0.6] = -1
gmm5_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr5_ip).predict_proba(dr5_ip)
ca5 = np.argmax(gmm5_ip, axis=1)
ca5[gmm5_ip.max(axis=1) < 0.6] = -1

# visualizing to check --- change all but dr3
dr3 = pd.DataFrame(dr0_ip, columns=["x", "y", "z"])
dr3 = pd.concat([c0_id.reset_index(drop=True),dr3.reset_index(drop=True)], axis=1)
dr3 = pd.concat([dr3.reset_index(drop=True),pd.DataFrame(ca0, columns=["c"]).reset_index(drop=True)], axis=1)
color_map = {'-1': 'lightgrey'}
dr3['c'] = dr3['c'].astype(str)
fig = px.scatter_3d(dr3, x='x', y='y', z='z', color='c', color_discrete_map=color_map)
fig.update_traces(marker=dict(line=dict(width=0.001, color='white')))
fig.show()

# merging into df
cX_id['ip_cluster'] = -1
c0_id = pd.concat([c0_id.reset_index(drop=True),gmm_to_df(ca0, "ip").reset_index(drop=True)], axis=1)
c1_id = pd.concat([c1_id.reset_index(drop=True),gmm_to_df(ca1, "ip").reset_index(drop=True)], axis=1)
c2_id = pd.concat([c2_id.reset_index(drop=True),gmm_to_df(ca2, "ip").reset_index(drop=True)], axis=1)
c3_id = pd.concat([c3_id.reset_index(drop=True),gmm_to_df(ca3, "ip").reset_index(drop=True)], axis=1)
c4_id = pd.concat([c4_id.reset_index(drop=True),gmm_to_df(ca4, "ip").reset_index(drop=True)], axis=1)
c5_id = pd.concat([c5_id.reset_index(drop=True),gmm_to_df(ca5, "ip").reset_index(drop=True)], axis=1)

# adjusting clust no.
c1_id.loc[c1_id['ip_cluster'] != -1, 'ip_cluster'] += 3
c2_id.loc[c2_id['ip_cluster'] != -1, 'ip_cluster'] += 6
c3_id.loc[c3_id['ip_cluster'] != -1, 'ip_cluster'] += 9
c4_id.loc[c4_id['ip_cluster'] != -1, 'ip_cluster'] += 12
c5_id.loc[c5_id['ip_cluster'] != -1, 'ip_cluster'] += 15

# merge
frames = [cX_id, c0_id, c1_id, c2_id, c3_id, c4_id, c5_id]
dfx = pd.concat(frames)
stats = [cX, c0, c1, c2, c3, c4, c5]
dfy = pd.concat(stats)
df2 = pd.concat([dfx.reset_index(drop=True),dfy.reset_index(drop=True)], axis=1)
df2 = df2.sort_values('playerId')
df2.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv', index=False)

# visualize - 3D
temp = df2[['playerId', 'seasonId', 'ip_cluster']]
dr2 = pd.merge(dr2, temp, on=['playerId', 'seasonId'])
color_map = {'-1': 'lightgrey'}
dr2['c2'] = dr2['ip_cluster'].astype(str)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='c2', color_discrete_map=color_map, color_discrete_sequence=px.colors.qualitative.Light24_r)
fig.update_traces(marker=dict(line=dict(width=0.001, color='white')))
fig.show()
