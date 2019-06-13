#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


FuelConsumption = pd.read_csv('FuelConsumptionCo2.csv')
FuelConsumption.head()


# In[3]:


#Useless - doesn't work cause of randomly assigning a number to the Letter
cons = FuelConsumption.replace("Z", 1, inplace = False)
cons = cons.replace("X", 2, inplace = False)
cons = cons.replace("D", 3, inplace = False)
cons = cons.replace("E", 4, inplace = False)
cons.head()


# In[4]:


cons.info()


# In[5]:


cons.describe()


# In[6]:


sns.pairplot(cons)


# In[7]:


sns.distplot(cons[['CO2EMISSIONS']],hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')


# In[8]:


cons.corr()


# In[9]:


sns.heatmap(cons.corr(), annot = True)


# In[10]:


cons.columns


# In[11]:


X = cons[['ENGINESIZE','CYLINDERS', 'FUELCONSUMPTION_CITY',
       'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB']]


# In[12]:


y = cons['CO2EMISSIONS']


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[14]:


#importing the Linear Regression Algorithm
from sklearn.linear_model import LinearRegression
#creating LinearRegression Object
lm = LinearRegression()
#Training the Data Model
lm.fit(X_train, y_train)


# In[15]:


#Displaying the Intercept
print(lm.intercept_)


# In[16]:


coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
coeff_df


# In[17]:


predictions = lm.predict(X_test)
plt.scatter(y_test, predictions, edgecolor='black')


# In[18]:


sns.distplot((y_test - predictions), bins = 50, hist_kws=dict(edgecolor="black", linewidth=1),color='Blue')


# In[19]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[ ]:




