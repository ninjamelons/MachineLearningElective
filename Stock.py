#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import DataFrame
from sklearn import linear_model


Stock_Market = {'Year': [2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2017,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016,2016],
                'Month': [12, 11,10,9,8,7,6,5,4,3,2,1,12,11,10,9,8,7,6,5,4,3,2,1],
                'InterestRate': [2.75,2.5,2.5,2.5,2.5,2.5,2.5,2.25,2.25,2.25,2,2,2,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75,1.75],
                'UnemploymentRate': [5.3,5.3,5.3,5.3,5.4,5.6,5.5,5.5,5.5,5.6,5.7,5.9,6,5.9,5.8,6.1,6.2,6.1,6.1,6.1,5.9,6.2,6.2,6.1],
                'StockIndexPrice': [1464,1394,1357,1293,1256,1254,1234,1195,1159,1167,1130,1075,1047,965,943,958,971,949,884,866,876,822,704,719]        
                }

df = DataFrame(Stock_Market,columns=['Year','Month','InterestRate','UnemploymentRate','StockIndexPrice'])

print (df)


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


sns.pairplot(df)


# In[6]:


sns.distplot(df['StockIndexPrice'], hist_kws=dict(edgecolor="black"))


# In[7]:


df.corr()


# In[8]:


sns.heatmap(df.corr(), annot=True)


# In[9]:


X = df[['InterestRate', 'UnemploymentRate']]
X2 = df[['InterestRate']]


# In[10]:


y = df['StockIndexPrice']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y, test_size = 0.1)


# In[12]:


#importing the Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
#creating LinearRegression Object
lm = LinearRegression()
lm2 = LinearRegression()
#Training the Data Model
lm.fit(X_train, y_train)
lm2.fit(X2_train, y2_train)


# In[13]:


import pandas as pd
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[14]:


coeff_df = pd.DataFrame(lm2.coef_, X2.columns, columns=['Coefficient'])
coeff_df


# In[15]:


predictions = lm.predict(X_test)
plt.scatter(y_test, predictions, edgecolor='black')


# In[16]:


predictions2 = lm2.predict(X2_test)
plt.scatter(y2_test, predictions2, edgecolor='black')


# In[17]:


sns.distplot((y_test - predictions))


# In[18]:


sns.distplot((y2_test - predictions2))


# In[19]:


from sklearn import metrics
import numpy as np
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[20]:


print('MAE:', metrics.mean_absolute_error(y2_test, predictions2))
print('MSE:', metrics.mean_squared_error(y2_test, predictions2))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y2_test, predictions2)))

