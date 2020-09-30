#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt  
from sklearn import svm
from sklearn import metrics
import joblib
from sklearn.decomposition import PCA
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[2]:


dataframe = pd.read_csv("Downloads\proper_dataset.csv")


# In[3]:


dataframe.head()


# In[4]:


dataframe.shape


# In[5]:


dataframe.describe()


# In[6]:


dataframe.info()


# In[7]:


sns.pairplot(dataframe[['Unirating','Feeling hungry','Trouble in breathing','Shortness in breath','High BP','Chest pain','Predict']],hue='Predict')


# In[8]:


train=dataframe.drop('Predict',axis=1)
train.head()


# In[9]:


Predict=dataframe.Predict
Predict.head()


# In[10]:


X_train, X_test, Y_train, Y_test = train_test_split(train,Predict, test_size=0.2, random_state=10)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[11]:


clf=svm.SVC(kernel='linear')
clf.fit(X_train,Y_train)
Y_pred=clf.predict(X_test)


# In[12]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred,average='micro'))


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[15]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred,average='micro'))


# In[16]:


import pickle


# In[17]:


with open('model3_pickle','wb') as f:
    pickle.dump(clf,f)


# In[18]:


with open('model3_pickle','rb') as f:
    mp = pickle.load(f)


# In[19]:


mp.predict(X_test)


# In[ ]:




