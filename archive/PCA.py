import seaborn as sns
import umap
from sklearn.decomposition import PCA

from archive.helpers.helperFunctions import *

# Load dataset
raw = pd.read_csv("C:/ITU/ITU_Research_Project/clustered_data/events_CN_UMAP.csv", sep = ",", encoding='unicode_escape')
raw = pd.read_csv('C:/Users/mall/OneDrive - Implement/Documents/Andet/RP/Data/events_CN_UMAP.csv',
                 sep=",", encoding='unicode_escape')

# saving IDs
df_id = raw[['playerId', 'seasonId', 'map_group', 'pos_group']]
raw = raw.drop(['playerId', 'seasonId', 'map_group', 'pos_group', ], axis=1)

#Create pca instance
pca = PCA(n_components=0.8)

#Split data in train and test sets
trans = pca.fit_transform(raw)

# Compute an array of principal components
explained_variance = pca.explained_variance_ratio_

#extract number of principal components
print(len(explained_variance))
print(sum(explained_variance))
print(explained_variance)

plt.title("Line graph")
plt.plot(explained_variance, color ="green", marker='x')
plt.show()

# op no. clusters
opt_clus(trans)

# clustering
gmm3 = GaussianMixture(n_components=3, covariance_type='full', random_state=42).fit(trans).predict(trans)

# merging
df3 = pd.concat([raw.reset_index(drop=True),gmm_to_df(gmm3, "ip").reset_index(drop=True)], axis=1)
df3_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm3, "ip").reset_index(drop=True)], axis=1)

# creating DFs
c0 = df3[df3.ip_cluster == 0]
c1 = df3[df3.ip_cluster == 1]
c2 = df3[df3.ip_cluster == 2]

# dropping cluster
c0 = c0.drop(['ip_cluster'], axis=1)
c1 = c1.drop(['ip_cluster'], axis=1)
c2 = c2.drop(['ip_cluster'], axis=1)

# creating ID DFs
c0_id = df3_id[df3_id.ip_cluster == 0]
c1_id = df3_id[df3_id.ip_cluster == 1]
c2_id = df3_id[df3_id.ip_cluster == 2]

# dropping cluster
c0_id = c0_id.drop(['ip_cluster'], axis=1)
c1_id = c1_id.drop(['ip_cluster'], axis=1)
c2_id = c2_id.drop(['ip_cluster'], axis=1)

# applying UMAP - remember to install pynndescent to make it run faster
pca0 = PCA(n_components=0.8)
pca1 = PCA(n_components=0.8)
pca2 = PCA(n_components=0.8)
trans0 = pca0.fit_transform(c0)
trans1 = pca1.fit_transform(c1)
trans2 = pca2.fit_transform(c2)

print(len(pca.explained_variance_ratio_))

# finding/creating clusters
opt_clus(trans0) # running for 0-5

# ------------------------------------------------------------

# testing PCA + UMAP
dr = umap.UMAP(n_neighbors=50, min_dist=0.0, n_components=2, random_state=42).fit_transform(trans)
opt_clus(dr)

# clustering
gmm = GaussianMixture(n_components=5, covariance_type='full', random_state=42).fit(dr).predict(dr)
plt.scatter(dr[:, 0], dr[:, 1], c=gmm, s=40, cmap='viridis')
plt.show()

# merging
df2 = pd.concat([raw.reset_index(drop=True),gmm_to_df(gmm, "ip").reset_index(drop=True)], axis=1)
df2_id = pd.concat([df_id.reset_index(drop=True),gmm_to_df(gmm, "ip").reset_index(drop=True)], axis=1)

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

# finding/creating clusters
opt_clus(dr0_ip) # running for 0-5