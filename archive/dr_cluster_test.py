import seaborn as sns
import umap
from archive.helpers.helperFunctions import *
from sklearn.mixture import GaussianMixture
from sklearn.feature_selection import VarianceThreshold

# ------------ PREP ------------

# loading
df = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv',
                 sep=",",
                 encoding='unicode_escape')

# saving IDs
df_id = df[['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group']]
df = df.drop(['playerId', 'seasonId', 'teamId', 'map_group', 'pos_group',], axis=1)

# variance thresholding
vt = VarianceThreshold(threshold=0.003)
_ = vt.fit(df)
mask = vt.get_support() # getting boolean mask
df = df.loc[:, mask] # subsetting the data

# possession and non-posession sets
df_ip = df.iloc[:, np.r_[0:3, 4:7, 9:15, 20:24, 26, 32:40, 42, 44, 47:52]]
df_op = df.iloc[:, np.r_[3, 7:9, 16:20, 24:26, 27:30, 40:42, 43, 45:47]]

df.columns.get_loc("Air duel")
df.columns[44]

# splitting op into two positional parts
df_op = df_op.join(df_id)
df_op.insert(22, 'OD', df_op.apply(lambda row: off_def(row), axis=1), allow_duplicates=True)
df_def = df[df_op.OD == 'def']
df_off = df[df_op.OD == 'off']
df_def_id = df_def.pop('playerId', 'seasonId', 'teamId', 'map_group', 'pos_group', 'OD')

test = df.describe()

outliers = find_outliers_IQR(df_op)
check = outliers.describe()

# ------------ IN POSSESSION ------------

# applying UMAP - remember to install pynndescent to make it run faster
dr_ip = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_ip)

# when n_components=2
dr2 = pd.DataFrame(dr_ip, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# optimal GMM model
opt_clus(dr_ip)

# clustering, in-possessions
gmm_ip = GaussianMixture(n_components=10, covariance_type='full', random_state=42).fit(dr_ip).predict(dr_ip)
plt.scatter(dr_ip[:, 0], dr_ip[:, 1], c=gmm_ip, s=40, cmap='viridis')
plt.show()

# merging
df2_ip = pd.concat([df_ip.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)
df2_ip.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/clusters.csv', index=False)

df2_id.groupby(['ip_cluster'], as_index=False).count()

# validation
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
val_df = pd.merge(val, df2_id, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)


# # ------------ OUT OF POSSESSION ------------

# applying UMAP - remember to install pynndescent to make it run faster
dr_op = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_op)

# when n_components=2
dr2 = pd.DataFrame(dr_op, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# optimal GMM model
opt_clus(dr_op)

# clustering, out-of-possessions
gmm_op = GaussianMixture(n_components=11, covariance_type='full', random_state=42).fit(dr_op).predict(dr_op)
plt.scatter(dr_op[:, 0], dr_op[:, 1], c=gmm_op, s=40, cmap='viridis')
plt.show()

# x
test = 'test'

# merging
df2_ip = pd.concat([df_ip.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm_ip, "ip").reset_index(drop=True)], axis=1)
df2_ip.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/clusters.csv', index=False)

df2_id.groupby(['ip_cluster'], as_index=False).count()

# validation
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
val_df = pd.merge(val, df2_id, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)





# ---------------------------------------------------------------------------------





# applying UMAP - remember to install pynndescent to make it run faster
dr = umap.UMAP(n_neighbors=100, min_dist=0.0, n_components=2, random_state=42).fit_transform(df_ip)

# when n_components=2
dr2 = pd.DataFrame(dr, columns=["x", "y"])
dr2 = df_id.join(dr2)
plot = sns.scatterplot(data=dr2, x="x", y="y", hue = "pos_group")
plt.show()

# optimal GMM model
opt_clus(dr)

# clustering, in-possessions
gmm = GaussianMixture(n_components=6, covariance_type='full', random_state=42).fit(dr).predict(dr)
plt.scatter(dr[:, 0], dr[:, 1], c=gmm, s=40, cmap='viridis')
plt.show()

# merging
df2 = pd.concat([df.reset_index(drop=True),gmm_to_df(gmm, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm, "ip").reset_index(drop=True)], axis=1)

# creating DFs
c0 = df2[df2.ip_cluster == 0]
c1 = df2[df2.ip_cluster == 1]
c2 = df2[df2.ip_cluster == 2]
c3 = df2[df2.ip_cluster == 3]
c4 = df2[df2.ip_cluster == 4]
c5 = df2[df2.ip_cluster == 5]

# dropping cluster
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)
c3 = c3.drop(['ip_cluster'], axis=1)
c4 = c4.drop(['ip_cluster'], axis=1)
c5 = c5.drop(['ip_cluster'], axis=1)

# creating ID DFs
c0_id = df2_id[df2_id.ip_cluster == 0]
c1_id = df2_id[df2_id.ip_cluster == 1]
c2_id = df2_id[df2_id.ip_cluster == 2]
c3_id = df2_id[df2_id.ip_cluster == 3]
c4_id = df2_id[df2_id.ip_cluster == 4]
c5_id = df2_id[df2_id.ip_cluster == 5]

# dropping cluster
c0_id = c0_id.drop(['ip_cluster'], axis=1)
c1_id = c1_id.drop(['ip_cluster'], axis=1)
c2_id = c2_id.drop(['ip_cluster'], axis=1)
c3_id = c3_id.drop(['ip_cluster'], axis=1)
c4_id = c4_id.drop(['ip_cluster'], axis=1)
c5_id = c5_id.drop(['ip_cluster'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
dr0_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c0)
dr1_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c1)
dr2_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c2)
dr3_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c3)
dr4_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c4)
dr5_ip = umap.UMAP(n_neighbors=10, min_dist=0.0, n_components=2, random_state=42).fit_transform(c5)

# when n_components=2
test = pd.DataFrame(dr5_ip, columns=["x", "y"])
test = c5_id.join(test)
plot = sns.scatterplot(data=test, x="x", y="y", hue = "pos_group")
plt.show()
test.groupby(['map_group'], as_index=False).count()

# creating clusters
opt_clus(dr0_ip) # running for 0-5
gmm0_ip = GaussianMixture(n_components=4, covariance_type='full', random_state=42).fit(dr0_ip).predict(dr0_ip)
gmm1_ip = GaussianMixture(n_components=2, covariance_type='full', random_state=42).fit(dr1_ip).predict(dr1_ip)
gmm2_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr2_ip).predict(dr2_ip)
gmm3_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr3_ip).predict(dr3_ip)
gmm4_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr4_ip).predict(dr4_ip)
gmm5_ip = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(dr5_ip).predict(dr5_ip)

# visualizing to check
plt.scatter(dr0_ip[:, 0], dr0_ip[:, 1], c=gmm0_ip, s=40, cmap='viridis')
plt.show()

# merging into df
c0_id = pd.concat([c0_id.reset_index(drop=True),gmm_to_df(gmm0_ip, "ip").reset_index(drop=True)], axis=1)
c1_id = pd.concat([c1_id.reset_index(drop=True),gmm_to_df(gmm1_ip, "ip").reset_index(drop=True)], axis=1)
c2_id = pd.concat([c2_id.reset_index(drop=True),gmm_to_df(gmm2_ip, "ip").reset_index(drop=True)], axis=1)
c3_id = pd.concat([c3_id.reset_index(drop=True),gmm_to_df(gmm3_ip, "ip").reset_index(drop=True)], axis=1)
c4_id = pd.concat([c4_id.reset_index(drop=True),gmm_to_df(gmm4_ip, "ip").reset_index(drop=True)], axis=1)
c5_id = pd.concat([c5_id.reset_index(drop=True),gmm_to_df(gmm5_ip, "ip").reset_index(drop=True)], axis=1)

# adjusting clust no.
c1_id['ip_cluster'] = c1_id['ip_cluster'] + 4
c2_id['ip_cluster'] = c2_id['ip_cluster'] + 6
c3_id['ip_cluster'] = c3_id['ip_cluster'] + 9
c4_id['ip_cluster'] = c4_id['ip_cluster'] + 12
c5_id['ip_cluster'] = c5_id['ip_cluster'] + 15

# val
val = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/val.csv', sep=";", encoding='unicode_escape')
frames = [c0_id, c1_id, c2_id, c3_id, c4_id, c5_id]
val2 = pd.concat(frames)
val_df = pd.merge(val, val2, on='playerId')
val_df.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/valDF.csv', index=False)

# export
stats = [c0, c1, c2, c3, c4, c5]
stats2 = pd.concat(stats)
test = pd.concat([val2.reset_index(drop=True),stats2.reset_index(drop=True)], axis=1)
test.to_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/clusters.csv', index=False)