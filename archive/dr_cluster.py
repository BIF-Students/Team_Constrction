import seaborn as sns
from archive.helpers.helperFunctions import *
import plotly.express as px
from sklearn.mixture import GaussianMixture
import umap.umap_ as umap

# loading
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv',
                 sep=",",
                 encoding='unicode_escape')

df = df.drop(['playerId', 'seasonId', 'map_group', 'pos_group'], axis=1)

# creating data set
df_id = df[['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group']]
df_def = df[df.pos_group == 'DEF']
df_mid = df[df.pos_group == 'MID']
df_wide = df[df.pos_group == 'WIDE']
df_att = df[df.pos_group == 'ATT']

# dropping the ids
df_def = df_def.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group'], axis=1)
df_mid = df_mid.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group'], axis=1)
df_wide = df_wide.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group'], axis=1)
df_att = df_att.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group'], axis=1)

# getting IDs for map visualization
df_id_def = df_id[df_id.pos_group == 'DEF']
df_id_mid = df_id[df_id.pos_group == 'MID']
df_id_wide = df_id[df_id.pos_group == 'WIDE']
df_id_att = df_id[df_id.pos_group == 'ATT']

# applying UMAP - remember to install pynndescent to make it run faster
dr_def = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df_def)

df = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df)

dr_mid = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df_mid)
dr_wide = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df_wide)
dr_att = umap.UMAP(n_neighbors=80, min_dist=0.0, n_components=3, random_state=42).fit_transform(df_att)

# when n_components=2
# dr2 = pd.DataFrame(dr_wide, columns=["x", "y"])
# dr2 = df_id_wide.join(dr2)
# plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "map_group")
# plt.show()

# when n_components=3
dr2 = pd.DataFrame(dr_mid, columns=["x", "y", "z"])
dr2 = pd.merge(df_id_mid, dr2, left_index=True, right_index=True)
fig = px.scatter_3d(dr2, x='x', y='y', z='z', color='map_group')
fig.show()

# optimal GMM model
opt_clus(dr_def)
opt_clus(dr_mid)
opt_clus(dr_wide)
opt_clus(dr_att)

# visualizing defenders
gmm_def = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit(dr_def).predict(dr_def)
# plt.scatter(dr_def[:, 0], dr_def[:, 1], c=gmm_def, s=40, cmap='viridis')
fig = px.scatter_3d(x=dr_def[:, 0], y=dr_def[:, 1], z=dr_def[:, 2], color=gmm_def)
fig.show()

# visualizing midfielders
gmm_mid = GaussianMixture(n_components=6, covariance_type='full', random_state=42).fit(dr_mid).predict(dr_mid)
# plt.scatter(dr_mid[:, 0], dr_mid[:, 1], c=gmm_mid, s=40, cmap='viridis')
fig = px.scatter_3d(x=dr_mid[:, 0], y=dr_mid[:, 1], z=dr_mid[:, 2], color=gmm_mid)
fig.show()

# visualizing wide players
gmm_wide = GaussianMixture(n_components=7, covariance_type='full', random_state=42).fit(dr_wide).predict(dr_wide)
# plt.scatter(dr_wide[:, 0], dr_wide[:, 1], c=gmm_wide, s=40, cmap='viridis')
fig = px.scatter_3d(x=dr_wide[:, 0], y=dr_wide[:, 1], z=dr_wide[:, 2], color=gmm_wide)
fig.show()

# visualizing attackers
gmm_att = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr_att).predict(dr_att)
# plt.scatter(dr_att[:, 0], dr_att[:, 1], c=gmm_att, s=40, cmap='viridis')
fig = px.scatter_3d(x=dr_att[:, 0], y=dr_att[:, 1], z=dr_att[:, 2], color=gmm_att)
fig.show()

# merging into df
df_id_def = pd.concat([df_id_def.reset_index(drop=True),gmm_to_df(gmm_def).reset_index(drop=True)], axis=1)
df_id_mid = pd.concat([df_id_mid.reset_index(drop=True),gmm_to_df(gmm_mid).reset_index(drop=True)], axis=1)
df_id_wide = pd.concat([df_id_wide.reset_index(drop=True),gmm_to_df(gmm_wide).reset_index(drop=True)], axis=1)
df_id_att = pd.concat([df_id_att.reset_index(drop=True),gmm_to_df(gmm_att).reset_index(drop=True)], axis=1)

# visualization to validate merge
test = pd.DataFrame(dr_mid, columns=["x", "y"])
test = pd.concat([df_id_mid.reset_index(drop=True),test.reset_index(drop=True)], axis=1)
plot = sns.scatterplot(data=test, x="x", y="y", hue = "cluster")
plt.show()

# exporting clusters
df_id_def.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/DEF.csv', index=False)
df_id_mid.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/MID.csv', index=False)
df_id_wide.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/WIDE.csv', index=False)
df_id_att.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/ATT.csv', index=False)

# val
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
frames = [df_id_def, df_id_mid, df_id_wide, df_id_att]
val2 = pd.concat(frames)
val_df = pd.merge(val, val2, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)