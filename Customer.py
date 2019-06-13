#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
cust_df = pd.read_csv("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv")
cust_df.head()


# In[2]:


df = cust_df.drop('Address', axis=1)
df.head()


# In[3]:


import numpy as np
from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
clus_dataset = StandardScaler().fit_transform(X)


# In[4]:


clus_dataset


# In[5]:


from sklearn.cluster import KMeans
clusternum = 3
k_means = KMeans(init="k-means++", n_clusters=clusternum, n_init=12)
k_means.fit(clus_dataset)
lables = k_means.labels_
print(lables)


# In[6]:


df['Clus_km']=lables
df.head(5)


# In[7]:


df.groupby('Clus_km').mean()


# In[8]:


area = np.pi * (X[:,1])**2
plt.scatter(X[:,0],X[:,3], s=area, c=lables.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


# In[9]:


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8,6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
plt.cla()

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c=lables.astype(np.float))

