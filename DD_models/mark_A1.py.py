#!/usr/bin/env python
# coding: utf-8

# In[4]:


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


# In[5]:


dataframe = pd.read_csv("Downloads\Mark1.csv")


# In[6]:


dataframe.head()


# In[7]:


dataframe.shape


# In[8]:


dataframe.describe()


# In[9]:


sns.pairplot(dataframe[['stomach_pain','weight_loss','cough','chest_pain','continuous_feel_of_urine','outcome']],hue='outcome')


# In[10]:


train=dataframe.drop('outcome',axis=1)
train.head()


# In[11]:


outcome=dataframe.outcome
outcome.head()


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(train,outcome, test_size=0.2, random_state=0)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[14]:


clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)


# In[15]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[16]:


from sklearn.linear_model import LogisticRegression


# In[17]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[18]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[19]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[21]:


Y_pred = classifier.predict(X_test)


# In[22]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[23]:


import pickle


# In[24]:


with open('mark1_pickle','wb') as f:
    pickle.dump(clf,f)


# In[25]:


with open('mark1_pickle','rb') as f:
    mp = pickle.load(f)


# In[26]:


mp.predict(X_test)


# In[ ]:




