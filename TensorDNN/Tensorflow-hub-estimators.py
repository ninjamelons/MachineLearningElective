#!/usr/bin/env python
# coding: utf-8

# https://medium.com/tensorflow/building-a-text-classification-model-with-tensorflow-hub-and-estimators-3169e7aa568

# In[1]:


import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

#from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder


# In[2]:


#import pymongo
#mongoClient = pymongo.MongoClient("mongodb://localhost:27017/")
#db = mongoClient["textrecognition"]
#cursor = db["ready"].find({}, {'_id': False})
#df = pd.DataFrame(list(cursor))

df = pd.read_json("singlelabel2.json", lines=True)


# In[3]:


df = df.sample(frac=1).reset_index(drop=True)


# In[4]:


df.head(10)


# In[5]:


text = df['text']
categories = df['categories']


# set1 = categories.apply(lambda x: x[0])
# set2 = categories.apply(lambda x: x[1])
# 
# set1 = set1.unique()
# set2 = set2.unique()
# 
# unique_categories = []
# 
# for s in set1:
#     unique_categories.append(s)
# for s in set2:
#     unique_categories.append(s)
# 
# unique_categories

# In[6]:


unique_categories = categories.unique()
unique_categories


# In[7]:


train_size = int(len(text) * .8)

train_text = text[:train_size]
train_categories = categories[:train_size]

test_text = text[train_size:]
test_categories = categories[train_size:]


# In[8]:


text_embeddings = hub.text_embedding_column(
  "pieces_of_text", 
  #module_spec="https://tfhub.dev/google/universal-sentence-encoder/2",
  module_spec="https://tfhub.dev/google/nnlm-en-dim128/1",
  trainable=True
)


# elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)
# text_embeddings = elmo(
#     "pieces_of_text",
#     signature="default",
#     as_dict=True)["elmo"]

# In[9]:


#encoder = MultiLabelBinarizer()
#encoder = LabelBinarizer()
encoder = LabelEncoder()
encoder.fit_transform(categories)
train_encoded = encoder.transform(train_categories)
test_encoded = encoder.transform(test_categories)
num_classes = len(encoder.classes_)


# In[10]:


encoder.classes_


# In[11]:


encoder.classes_.size


# In[12]:


#https://medium.com/octavian-ai/which-optimizer-and-learning-rate-should-i-use-for-deep-learning-5acb418f9b2

optimizer = []
optimizer.append(tf.train.AdadeltaOptimizer(learning_rate=1))


# In[13]:


optimizer


# multi_label_head = tf.contrib.estimator.multi_label_head(
#     num_classes,
#     loss_reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE
# )

# In[14]:


estimators = []

for op in optimizer:
    estimators.append(tf.estimator.DNNClassifier(
        feature_columns = [text_embeddings],
        hidden_units=[32, 16],
        n_classes=encoder.classes_.size,
        optimizer=op
    ))


# estimator = tf.estimator.DNNEstimator(
#     head=multi_label_head,
#     hidden_units=[64,16],
#     feature_columns=[text_embeddings]
# )

# In[15]:


# Format the data for the numpy_input_fn
features = {
  "pieces_of_text": np.array(train_text)
}
labels = np.array(train_encoded)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    features, 
    labels, 
    shuffle=False, 
    batch_size=32, 
    num_epochs=3
)


# In[16]:


for estimator in estimators:
    estimator.train(input_fn=train_input_fn)


# In[17]:


eval_input_fn = tf.estimator.inputs.numpy_input_fn({"pieces_of_text": np.array(test_text).astype(np.str)}, test_encoded.astype(np.int32), shuffle=False)


# In[18]:


estimators[0].evaluate(input_fn=eval_input_fn)


# In[20]:


estimators[0].evaluate(input_fn=eval_input_fn)


# estimators[2].evaluate(input_fn=eval_input_fn)

# estimators[3].evaluate(input_fn=eval_input_fn)

# In[45]:


from os import listdir
from os.path import isfile, join
mypath = 'C:/Users/Bruger/BigDataPython/TensorDNN/Samples'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

raw_data = []
for f in onlyfiles:
    print(f)
    with open(mypath + '/' + f, 'r', encoding="utf-8") as file:
        data = file.read().replace('\n', ' ')
        raw_data.append(data)


# In[46]:


predict_input_fn = tf.estimator.inputs.numpy_input_fn({"pieces_of_text": np.array(raw_data).astype(np.str)}, shuffle=False)

results = []
for estimator in estimators:
    results.append(estimator.predict(predict_input_fn))


# In[47]:


for result in results:
    for categories in result:
        top = categories['probabilities'].argsort()[-2:][::-1]
        print()
        for category in top:
            text_category = encoder.classes_[category]
            print(text_category + ': ' + str(round(categories['probabilities'][category] * 100, 2)) + '%', end=" | ")
    print()


# In[ ]:




