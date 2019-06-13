#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Learn about API authentication here: https://plot.ly/pandas/getting-started
# Find your api_key here: https://plot.ly/settings/api
import plotly
import plotly.plotly as py
import plotly.graph_objs as go


plotly.tools.set_credentials_file(username='ninjamelons', api_key='M01mLTvqIxmzqeDoXTqW')

import pandas as pd
#import numpy as np

#N = 500
#x = np.linspace(0, 1, N)
#y = np.random.randn(N)

import pyodbc

conn = pyodbc.connect('DRIVER={ODBC Driver 13 for SQL Server};SERVER=92.43.176.119;DATABASE=Indusoft;UID=UCN-DDD05;PWD=rQdj7KmG8p')
sql = "SELECT TOP 100000 [Date], [Indgang 0405], [Indgang 0407] FROM DataCollectionBaand2 WHERE [Indgang 0407] = 1 OR [Indgang 0405] = 1 ORDER BY CONVERT(VARCHAR, [Date]) ASC;"

df4745 = pd.read_sql(sql,conn)
df4745.columns = ['Date', 'Indgang45', 'Indgang47']
df4745.head()


# In[2]:


df4745['Date'] = pd.to_datetime(df4745['Date'])
df4745.info()


# In[3]:


import datetime
from datetime import timedelta


# In[4]:


def getDifference4745(df, retDf) :
    #Initialise 0407, 0405 rows and row to be added to new df
    row47 = {'Date':'1900-00-00 00:00:00', 'Indgang45': '0', 'Indgang47': '0'}
    row45 = {'Date':'1900-00-00 00:00:00', 'Indgang45': '0', 'Indgang47': '0'}
    retRow = {'Date':'1900-00-00 00:00:00', 'Indgang45': '0', 'Indgang47': '0'}
    ind47 = False
    ind45 = False
    
    #Iterate through input df to get the next 0407 with 1
    for row in df.itertuples() :
        if row.Indgang47 == 1 :
            ind47 = True
            row47 = row            
            break
            
    #Iterate through input df to get the next 0405 after 0407 with 1
    for row in df.itertuples() :
        if row47.Date != '1900-00-00 00:00:00' and row.Indgang45 == 1 and row.Date > row47.Date :
            ind45 = True
            row45 = row
            break
            
    #Assign values to be added to new df
    if hasattr(row45, 'Date') and hasattr(row47, 'Date') :
        retRow = row45.Date - row47.Date
        days, seconds = retRow.days, retRow.seconds
        hours = days * 24 + seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        retRow = timedelta(hours=hours, minutes=minutes, seconds=seconds)
    
        #Add new row to the final df
        retDf = retDf.append({'Date': row47.Date, 'DeltaTime':retRow}, ignore_index=True)
    
        #Delete rows up until this point from input df
        df.drop(df[df.Date < row45.Date].index, inplace=True)
        
    if len(df.index) > 20 :
        return getDifference4745(df,retDf)
    else :
        return retDf

finalDf = pd.DataFrame(columns=['Date', 'DeltaTime'])


# In[ ]:


nextDf = getDifference4745(df4745,finalDf)


# In[ ]:


nextDf


# In[ ]:


import plotly.plotly as py
import plotly.graph_objs as go

data = [
    go.Scatter(
        x=nextDf['Date'], # assign x as the dataframe column 'x'
        y=nextDf['DeltaTime']
    )
]

layout = go.Layout(
    title='Time taken for date',
    yaxis=dict(title='Time taken'),
    xaxis=dict(title='Date of start')
)

fig = go.Figure(data=data, layout=layout)

# IPython notebook
# py.iplot(fig, filename='pandas/line-plot-title')

url = py.plot(fig, filename='4745Data')


# In[ ]:




