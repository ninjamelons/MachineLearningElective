#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import pyodbc

conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=92.43.176.119;DATABASE=Indusoft;UID=UCN-DDD05;PWD=rQdj7KmG8p')
sqlQ = "SELECT TOP (100000) * FROM [Indusoft].[dbo].[DataCollectionBaand2]"

m2df = pd.read_sql(sqlQ,conn)
m2df.head()


# In[2]:


m2df.info()


# In[5]:


m2df['Date'] = pd.to_datetime(m2df['Date'])
m2df.info()


# In[6]:


import seaborn as sns
sns.heatmap(m2df.corr(), annot = True)


# In[14]:


df = m2df[['Indgang 0401','Indgang 0403']]
df = df.set_index(m2df['Date'])


# In[15]:


df.head()


# In[18]:


df['Overlapping'] = df['Indgang 0401'] + df['Indgang 0403']
df.head()


# In[28]:


m2df.plot(x='Date',y='Indgang 0405')


# In[29]:


condition = m2df['Date'] < '2018-02-24'
m2df.where(cond=condition, inplace=True)


# In[30]:


m2df.plot(x='Date',y='Indgang 0405')


# In[31]:


condition = m2df['Date'] > '2018-02-20'
m2df.where(cond=condition, inplace=True)


# In[32]:


condition = m2df['Date'] < '2018-02-21'
m2df.where(cond=condition, inplace=True)


# In[33]:


m2df.plot(x='Date',y='Indgang 0405')


# In[35]:


condition = m2df['Date'] > '2018-02-20 06:00:00'
m2df.where(cond=condition, inplace=True)
condition = m2df['Date'] < '2018-02-20 21:00:00'
m2df.where(cond=condition, inplace=True)

m2df.plot(x='Date',y='Indgang 0405')


# In[ ]:




