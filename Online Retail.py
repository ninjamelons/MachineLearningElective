#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pyflux as pf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


# In[2]:


raw_df = pd.read_excel("Online Retail.xlsx")
raw_df.head()


# In[3]:


raw_df.info()


# Data Preparation. 
# Extract total value paid for an item. 
# Remove outliers. 
# Calculate the daily revenue per country. 
# Forecast the total daily revenue for the future period. 

# In[4]:


raw_df['Total_value_paid'] = raw_df['Quantity']*raw_df['UnitPrice']
raw_df


# In[5]:


raw_df.describe()


# In[6]:


plt.scatter(raw_df['Quantity'], raw_df['UnitPrice'])


# In[7]:


indexesToRemove = raw_df[raw_df['Quantity'] < 0].index.values


# In[8]:


filtered_data = raw_df.drop(indexesToRemove, axis=0)


# In[9]:


filtered_data.describe()


# In[10]:


unitPricesToRemove = raw_df[raw_df['UnitPrice'] < 0].index.values


# In[11]:


filt_data = filtered_data.drop(unitPricesToRemove, axis=0)


# In[12]:


filt_data.describe()


# In[13]:


from sklearn.preprocessing import MinMaxScaler


# In[14]:


scaler = MinMaxScaler(feature_range=[0,1])


# In[15]:


normalized_df = scaler.fit_transform(filt_data[['Quantity', 'UnitPrice']])


# In[16]:


normalized_df = pd.DataFrame(normalized_df, columns=['Quantity', 'UnitPrice'])


# In[17]:


normalized_df.index = filt_data.index.values


# In[18]:


normalized_df.head()


# Fine the outliers: z score analysis. Remove the features that are 3 score higher than the SD from the mean

# In[19]:


mean_q = normalized_df['Quantity'].mean()


# In[20]:


std_q = normalized_df['Quantity'].std()


# In[21]:


mean_q


# In[22]:


std_q


# In[23]:


outlier_index = normalized_df[normalized_df['Quantity'] >= mean_q + 3 * std_q].index


# In[24]:


outlier_index


# In[25]:


filt_data.drop(outlier_index, axis=0, inplace=True)


# In[26]:


mean_up = normalized_df['UnitPrice'].mean()
std_up = normalized_df['UnitPrice'].std()
outlier_index_price = normalized_df[normalized_df['UnitPrice'] >= mean_up + 3 * std_up].index
filt_data.drop(outlier_index_price, axis=0, inplace=True)


# In[27]:


filt_data.describe()


# In[28]:


filt_data['Country'].value_counts()


# In[29]:


df = filt_data[filt_data['Country'] == 'United Kingdom']
df.head()


# Calculate the daily revenue

# In[30]:


df['InvoiceDate']


# In[31]:


daily_revenue_df = df.groupby(df['InvoiceDate'].dt.date)[['Total_value_paid']].sum()


# In[32]:


daily_revenue_df.reset_index(inplace=True)


# In[33]:


daily_revenue_df.head()


# In[34]:


daily_revenue_df['InvoiceDate'] = pd.to_datetime(daily_revenue_df['InvoiceDate'])
full_df = staging_df.merge(daily_revenue_df, how='left', left_on'Date', right_on='InvoiceDate')


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(full_df['Date'], full_df['Total_value_paid'])
plt.title('Daily revenues')
plt.ylable('Total Value Paid')
plt.xlable('Date')


# In[ ]:


full_df['Total_value_paid'] = full_df['Total_value_paid'].fillna(method='ffill')
full_df.isnull().sum()


# In[ ]:


plt.figure(figsize=(10,5))
plt.plot(full_df['Date'], full_df['Total_value_paid'])
plt.title('Daily revenues')
plt.ylable('Total Value Paid')
plt.xlable('Date')


# In[ ]:


ts_df = full_df['', '']


# Time series forecasting. ARIMA. VAR. Facebook Prophet

# Divide data into training and testing set

# In[ ]:


working_data = ts_df[:-30] # everything except last 30 days
df_test = ts_df[-30:] #Only last 30 days


# In[ ]:


from statsmodels.tsa.stattools import adfuller
kpi_to_forecast = Total_value_paid
df_stationary_test = adfuller(working_data[kpi_to_forecast])
df_stationary_test


# In[ ]:


test_statistics = df_stationary_test[0]
test_statistics


# In[ ]:


critical_values = df_stationary_test[4]
critical_values


# In[ ]:


critical_value_to_analyze = critical_values['1%']
critical_value_to_analyze


# In[ ]:


def return_conclusion(test_statistics, critical_value_to_analyze):
    if test_statistics > critical_value_to_analyze:
        print("THe series is not stationary")
    else:
        print("the series is stationary")


# In[ ]:


return_conclusion(test_statistics, critical_value_to_analyze)


# In[ ]:


working_data.index = working_data['Date']
working_data.drop('Date', axis=1, inplace=True)
working_data.head()


# In[ ]:


working_data.shift()


# In[ ]:


data_diff = working_data-working_data.shift()
data_diff


# In[ ]:


plt.plot(data_diff.index, working_data[kpi_to_forecast])


# In[ ]:


data_diff.head()


# In[ ]:


data_diff.dropna(inplace=True)


# In[ ]:


df_stationary_test = adfuller(data_diff[kpi_to_forecast])


# In[ ]:


return_conclusion(df_stationary_test[0], critical_values)


# In[ ]:


model = pf.ARIMA(data=working_data, ar=7, ma=13, integ=0)
x=model.fit()
x.summary()


# In[ ]:


predicted_df = model.predict(h=39)
predicted_df


# In[ ]:


model.plot_fit()

