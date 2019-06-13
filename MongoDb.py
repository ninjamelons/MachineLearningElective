#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pymongo
#https://api.mongodb.com/python/current/api/pymongo/collection.html
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
BigDb = mongoClient["BigData"]


# In[3]:


mongoClient.list_database_names()


# In[6]:


rests = BigDb["restaurants"]
neighbs = BigDb["neighborhoods"]
neighbs, rests


# In[8]:


rests.find_one()


# In[25]:


neighbs.find_one({ 'geometry': { '$geoIntersects': { '$geometry': { 'type': "Point", 'coordinates': [ -73.93414657, 40.82302903 ] } } } })


# In[49]:


neighborhood = neighbs.find_one({ 'geometry': { '$geoIntersects': { '$geometry': { 'type': "Point", 'coordinates': [ -73.93414657, 40.82302903 ] } } } })
neighborhood['geometry']


# In[60]:


restaurants = rests.find( { 'location': { '$geoWithin': { '$geometry': neighborhood['geometry'] } } } ).count()
restaurants


# In[61]:


restaurants


# In[ ]:




