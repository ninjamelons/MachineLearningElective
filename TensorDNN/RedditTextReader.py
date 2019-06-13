#!/usr/bin/env python
# coding: utf-8

# In[2]:


import praw
import pandas as pd
import datetime
reddit = praw.Reddit(client_id='YKHyzDsxpKV0mg',
                     client_secret='Ph_adJuGZAkrt2vx8n5MSNCnaWc',
                     user_agent='scripto by /u/ninjamelons',
                     username='ninjamelons')
print(reddit.info)


# In[3]:


import pymongo
mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
BigDb = mongoClient["BigData"]
redditDb = BigDb["reddit"]
redditDb


# Documents contain:
#     Title
#     text
#     subreddit
#     type of text
# *or*
#     Post title
#     comment text
#     subreddit
#     type of text

# In[4]:


posts = []

shortyStories = reddit.subreddit('ShortyStories').hot(limit=1000)
# dir(next(shortyStories)) #Check the variables that this object contains
for story in shortyStories:
    posts.append((datetime.datetime.utcfromtimestamp(story.created_utc),story.subreddit.display_name,story.title,story.selftext, 'shortstory'))

nosleepStories = reddit.subreddit('nosleep').hot(limit=1000)
for story in nosleepStories:
    posts.append((datetime.datetime.utcfromtimestamp(story.created_utc),story.subreddit.display_name,story.title,story.selftext, 'shortstory'))

sadStories = reddit.subreddit('ShortSadStories').hot(limit=1000)
for story in sadStories:
    posts.append((datetime.datetime.utcfromtimestamp(story.created_utc),story.subreddit.display_name,story.title,story.selftext, 'shortstory'))

hfyStories = reddit.subreddit('HFY').hot(limit=1000)
for story in hfyStories:
    posts.append((datetime.datetime.utcfromtimestamp(story.created_utc),story.subreddit.display_name,story.title,story.selftext, 'shortstory'))

wpromptStories = reddit.subreddit('writingprompts').hot(limit=1000)
# dir(next(writingPrompts).comments[0].submission) #Check the variables that this object contains
for submission in wpromptStories:
    submission.comments.replace_more(limit=0)
    for comment in submission.comments:
        posts.append((datetime.datetime.utcfromtimestamp(comment.created_utc),comment.subreddit.display_name,comment.submission.title,comment.body,'shortstory'))

promptstory = pd.DataFrame(posts, columns=['datetime','subreddit','title','body','type'])
promptstory.head(10), promptstory.tail(10)


# In[4]:


regex = "\*\*Welcome to the Prompt!\*\*"
filter = promptstory['body'].str.contains(regex)
filter.tail(10)


# In[5]:


botless_df = promptstory[~filter]
botless_df.head()


# In[6]:


import numpy as np
nandf = botless_df.replace('', np.nan, regex=False)
nandf = nandf.replace('[deleted]', np.nan, regex=False)
botless_df.head(), botless_df.tail(10)


# In[7]:


cleandf = nandf.dropna(inplace=False, axis=0)
cleandf['datetime'].apply(lambda x: x.strftime('%d%m%Y %H%M%S'))
cleandf.head()


# In[8]:


#redditDb is the mongoDb reddit collection instance
documentList = list()
for row in cleandf.itertuples():
    documentList.append({"datetime":row.datetime, "subreddit":row.subreddit, "title":row.title, "text":row.body, "type":row.type})


# In[9]:


cleandf.info()


# In[10]:


documentList


# In[11]:


result = redditDb.insert_many(documentList)
result

