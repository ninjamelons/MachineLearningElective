#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
cust_df = pd.read_csv('data_1024.csv', sep='\t')
cust_df.head()


# In[3]:


cust_df.info()


# In[6]:


import numpy as np
from sklearn.preprocessing import StandardScaler
X = cust_df.values[:,1:]
X = np.nan_to_num(X)
clus_dataset = StandardScaler().fit_transform(X)
clus_dataset


# In[7]:


from sklearn.cluster import KMeans
clusternum = 4
k_means = KMeans(init="k-means++", n_clusters=clusternum, n_init=12)
k_means.fit(clus_dataset)
lables = k_means.labels_
print(lables)


# In[8]:


cust_df['Clus_km']=lables
cust_df.head(5)


# In[10]:


cust_df.groupby('Clus_km').mean()


# In[14]:


import matplotlib.pyplot as plt
area = np.pi * (X[:,1])**2
plt.scatter(X[:,0],X[:,1], s=area, c=lables.astype(np.float), alpha=0.5)
plt.xlabel('Distance_Feature', fontsize=18)
plt.ylabel('Speeding_Feature', fontsize=16)

plt.show()


# In[ ]:




