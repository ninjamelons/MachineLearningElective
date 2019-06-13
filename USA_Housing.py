#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing librarires
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# loading csv data to dataframe 
USA_Housing = pd.read_csv('USA_Housing.csv')
# checking out the Data
USA_Housing.head()


# In[3]:


#checking columns and total records
USA_Housing.info()


# In[4]:


USA_Housing.describe()


# In[5]:


sns.pairplot(USA_Housing)


# In[6]:


sns.distplot(USA_Housing[['Price']],hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')


# In[7]:


#Displaying correlation among all the columns 
USA_Housing.corr()


# In[8]:


sns.heatmap(USA_Housing.corr(), annot = True)


# In[9]:


#Getting all Coulmn names
USA_Housing.columns


# In[10]:


# Columns as Features
X = USA_Housing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[11]:


# Price is my Target Variable, what we trying to predict
y = USA_Housing['Price']


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[13]:


#importing the Linear Regression Algorithm
from sklearn.linear_model import LinearRegression


# In[14]:


#creating LinearRegression Object
lm = LinearRegression()


# In[15]:


#Training the Data Model
lm.fit(X_train, y_train)


# In[16]:


#Displaying the Intercept
print(lm.intercept_)


# In[17]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[18]:


predictions = lm.predict(X_test)


# In[19]:


plt.scatter(y_test, predictions, edgecolor='black')


# In[20]:


sns.distplot((y_test - predictions), bins = 50, hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')


# In[21]:


from sklearn import metrics


# In[22]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




