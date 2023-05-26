# This file contains the code for the clustering and dimensionality reduction part of the project related to the tactical player roles

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

# Load the dataframe from the Brøndby database
df = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/events_clean.csv',
                 sep=",", encoding='unicode_escape')

# Add positional groups to the initial dataset for visualization purposes
df_posmin = load_db_to_pd(sql_query = "SELECT * FROM Wyscout_Positions_Minutes", db_name='Scouting')
df_pos = df_posmin.drop(['matchId', 'teamId', 'time'], axis=1)
df_pos = df_pos.groupby(['playerId', 'seasonId'], as_index=False).agg(gmodeHelp)
df = pd.merge(df, df_pos, on=['playerId', 'seasonId'])
df['pos_group'] = df.apply(lambda row: pos_group(row), axis=1)

# Save IDs before reducing dimensionality
df_id = df[['playerId', 'seasonId', 'pos_group']]
df = df.drop(['playerId', 'seasonId', 'position', 'pos_group'], axis=1)
df = df.drop(df.filter(like='_vaep').columns, axis=1)

# Apply UMAP - remember to install pynndescent to make it run faster
dr = umap.UMAP(n_neighbors=90, min_dist=0.0, n_components=2, random_state=42, metric='manhattan').fit_transform(df)

# Visualization of dimensionality reduced data when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = pd.concat([df_id.reset_index(drop=True),dr2.reset_index(drop=True)], axis=1)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# Visualization of dimensionality reduced data when n_components=3
'''dr2 = pd.DataFrame(dr, columns=["x", "y", "z"])
dr2 = pd.concat([df_id.reset_index(drop=True),dr2.reset_index(drop=True)], axis=1)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='pos_group')
fig.show()'''

# Find optimal GMM model parameters (assignment threshold, number of components, covariance matrix)
opt_clus(dr)

# Test if PCA performs better than UMAP (use if that is the case)
pca = PCA(n_components=0.8)
trans = pca.fit_transform(df)
opt_clus(trans)

# Cluster based on input from opt_clust(dr)
clusters = 6
gmm = GaussianMixture(n_components=clusters, covariance_type='diag', random_state=42).fit(dr)
probs = gmm.predict_proba(dr)
threshold = 0.7 # found via opt_clust
cluster_assignments = np.argmax(probs, axis=1)
cluster_assignments[probs.max(axis=1) < threshold] = -1
gmm_to_df(cluster_assignments, "ip").value_counts()

# Extract UMAP x and y features for visualization (and z if 3D)
x = dr[:, 0]
y = dr[:, 1]
'''z = dr[:, 2]'''

# Visualization of initial clustering output when n=2
colors = plt.cm.viridis(np.linspace(0, 1, clusters)).tolist()
plt.scatter(x, y, c=[colors[l] if l != -1 else 'lightgray' for l in cluster_assignments], cmap='CMRmap')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.25, zorder=1)
plt.show()

# Visualization of initial clustering output when n=3
'''col = pd.DataFrame(cluster_assignments, columns=["c"])
dr2 = pd.concat([dr2.reset_index(drop=True),col.reset_index(drop=True)], axis=1)
color_map = {'-1': 'lightgrey'}
dr2['c'] = dr2['c'].astype(str)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='c', color_discrete_map=color_map)
fig.update_traces(marker=dict(line=dict(width=0.001, color='white')))
fig.show()'''

# Merge data with cluster labels and IDs
df2 = pd.concat([df.reset_index(drop=True),gmm_to_df(cluster_assignments, "ip").reset_index(drop=True)], axis=1)
df2 = pd.concat([df_id.reset_index(drop=True),df2.reset_index(drop=True)], axis=1)

# Check value counts per cluster for position groups
df2.groupby('ip_cluster')['pos_group'].value_counts()


# Visualization for the thesis report
'''df2 = pd.concat([df2.reset_index(drop=True),pd.DataFrame(dr, columns = ['x','y']).reset_index(drop=True)], axis=1)
palette = sns.color_palette("CMRmap", 6) + ['lightgrey']
hue_order = list(range(6)) + [-1]

sns.scatterplot(data=df2[df2['ip_cluster'] != -1], x='x', y='y', hue='ip_cluster', palette=palette, hue_order=hue_order)
sns.scatterplot(data=df2[df2['ip_cluster'] == -1], x='x', y='y', color='lightgrey', size=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.25, zorder=1)
plt.legend('', frameon=False)
# plt.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
plt.savefig('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/clustersIMG2.png')
plt.show()'''

# -----------------------------------------------------------------------------------

# Build new dataframe for reclustering
temp = df2[['playerId', 'seasonId', 'ip_cluster']]
df2 = pd.merge(dr2, temp, on=['playerId', 'seasonId'])

# Create dataframes for each overall clustering group
cX = df2[df2.ip_cluster == -1]
c0 = df2[df2.ip_cluster == 0]
c1 = df2[df2.ip_cluster == 1]
c2 = df2[df2.ip_cluster == 2]
c3 = df2[df2.ip_cluster == 3]
c4 = df2[df2.ip_cluster == 4]
c5 = df2[df2.ip_cluster == 5]

# Drop initial cluster label for each sub-dataframe
cX = cX.drop(['ip_cluster'], axis=1)
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)
c3 = c3.drop(['ip_cluster'], axis=1)
c4 = c4.drop(['ip_cluster'], axis=1)
c5 = c5.drop(['ip_cluster'], axis=1)

# Save IDs for each sub-dataframe
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

# Recluster each of the 6 initial overall cluster groups in 3 clusters each with a 0.6 assignment threshold and full covariance matrix
gmm0_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c0).predict_proba(c0)
ca0 = np.argmax(gmm0_ip, axis=1)
ca0[gmm0_ip.max(axis=1) < 0.6] = -1
gmm1_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c1).predict_proba(c1)
ca1 = np.argmax(gmm1_ip, axis=1)
ca1[gmm1_ip.max(axis=1) < 0.6] = -1
gmm2_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c2).predict_proba(c2)
ca2 = np.argmax(gmm2_ip, axis=1)
ca2[gmm2_ip.max(axis=1) < 0.6] = -1
gmm3_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c3).predict_proba(c3)
ca3 = np.argmax(gmm3_ip, axis=1)
ca3[gmm3_ip.max(axis=1) < 0.6] = -1
gmm4_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c4).predict_proba(c4)
ca4 = np.argmax(gmm4_ip, axis=1)
ca4[gmm4_ip.max(axis=1) < 0.6] = -1
gmm5_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(c5).predict_proba(c5)
ca5 = np.argmax(gmm5_ip, axis=1)
ca5[gmm5_ip.max(axis=1) < 0.6] = -1

# Visualize the reclustering for each sub-dataframe (switching the names in the code below) to check output
dr3 = pd.concat([c3.reset_index(drop=True),pd.DataFrame(ca3, columns=["c"]).reset_index(drop=True)], axis=1)
plot = sns.scatterplot(data=dr3, x="x", y="y", hue = "c")
plt.show()

# Merge reclustered dataframed with their initial IDs
cX_id['ip_cluster'] = -1
c0_id = pd.concat([c0_id.reset_index(drop=True),gmm_to_df(ca0, "ip").reset_index(drop=True)], axis=1)
c1_id = pd.concat([c1_id.reset_index(drop=True),gmm_to_df(ca1, "ip").reset_index(drop=True)], axis=1)
c2_id = pd.concat([c2_id.reset_index(drop=True),gmm_to_df(ca2, "ip").reset_index(drop=True)], axis=1)
c3_id = pd.concat([c3_id.reset_index(drop=True),gmm_to_df(ca3, "ip").reset_index(drop=True)], axis=1)
c4_id = pd.concat([c4_id.reset_index(drop=True),gmm_to_df(ca4, "ip").reset_index(drop=True)], axis=1)
c5_id = pd.concat([c5_id.reset_index(drop=True),gmm_to_df(ca5, "ip").reset_index(drop=True)], axis=1)

# Adjust the cluster numbering to avoid duplicates
c1_id.loc[c1_id['ip_cluster'] != -1, 'ip_cluster'] += 3
c2_id.loc[c2_id['ip_cluster'] != -1, 'ip_cluster'] += 6
c3_id.loc[c3_id['ip_cluster'] != -1, 'ip_cluster'] += 9
c4_id.loc[c4_id['ip_cluster'] != -1, 'ip_cluster'] += 12
c5_id.loc[c5_id['ip_cluster'] != -1, 'ip_cluster'] += 15

# Merge reclustered dataframe into a single dataframe
frames = [cX_id, c0_id, c1_id, c2_id, c3_id, c4_id, c5_id]
dfx = pd.concat(frames)
stats = [cX, c0, c1, c2, c3, c4, c5]
dfy = pd.concat(stats)
df3 = pd.concat([dfx.reset_index(drop=True),dfy.reset_index(drop=True)], axis=1)
df3 = df3.sort_values('playerId')
df3 = df3.reset_index(drop=True)

# Check ouput
test = df3[df3['ip_cluster'] == -1]
test['ip_cluster'].value_counts()

# Export dataframe
df3.to_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv', index=False)

# Plot output (if 2D)
palette = sns.color_palette("CMRmap", 18) + ['lightgrey']
hue_order = list(range(18)) + [-1]

sns.scatterplot(data=df3[df3['ip_cluster'] != -1], x='x', y='y', hue='ip_cluster', palette=palette, hue_order=hue_order)
sns.scatterplot(data=df3[df3['ip_cluster'] == -1], x='x', y='y', color='lightgrey', size=1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.title('Cluster Distribution', fontsize=14, fontweight='bold')
plt.xlabel('umap_x')
plt.ylabel('umap_y')
plt.grid(color='lightgray', alpha=0.25, zorder=1)
plt.legend('', frameon=False)
# plt.legend(loc="upper right", bbox_to_anchor=(1.12, 1))
plt.savefig('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/clustersIMG.png')
plt.show()

# Plot output (if 3D)
'''df3 = pd.read_csv('C:/Users/mll/OneDrive - Brøndbyernes IF Fodbold/Dokumenter/TC/Data/players_clusters.csv',
                 sep=",", encoding='unicode_escape')
temp = df3[['playerId', 'seasonId', 'ip_cluster']]
dr2 = pd.merge(dr2, temp, on=['playerId', 'seasonId'])
color_map = {'-1': 'lightgrey'}
dr2['c2'] = dr2['ip_cluster'].astype(str)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='c2', color_discrete_map=color_map, color_discrete_sequence=px.colors.qualitative.Light24_r)
fig.update_traces(marker=dict(line=dict(width=0.001, color='white')))
fig.show()'''
