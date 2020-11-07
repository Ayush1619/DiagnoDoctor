#!/usr/bin/env python
# coding: utf-8

# In[23]:


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


# In[24]:


dataframe = pd.read_csv("Downloads\Mark1.csv")


# In[25]:


dataframe.head()


# In[26]:


dataframe.shape


# In[5]:


dataframe.describe()


# In[30]:


dataframe.corr()


# In[31]:


sns.pairplot(dataframe[['stomach_pain','weight_loss','cough','chest_pain','continuous_feel_of_urine','outcome']],hue='outcome')


# In[32]:


train=dataframe.drop('outcome',axis=1)
train.head()


# In[33]:


outcome=dataframe.outcome
outcome.head()


# In[34]:


X_train, X_test, Y_train, Y_test = train_test_split(train,outcome, test_size=0.2, random_state=0)
print("X_train size  ==>", X_train.shape)
print("X_test size  ==>", X_test.shape)
print("Y_train size  ==>", Y_train.shape)
print("Y_test size  ==>", Y_test.shape)


# In[35]:


clf1=svm.SVC(kernel='linear')
clf1.fit(X_train,Y_train)
Y_pred=clf1.predict(X_test)


# In[36]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[37]:


from sklearn.linear_model import LogisticRegression


# In[38]:


clf = LogisticRegression()
clf.fit(X_train, Y_train)
Y_pred=clf.predict(X_test)


# In[39]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[40]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[41]:


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, Y_train)


# In[42]:


Y_pred = classifier.predict(X_test)


# In[47]:


print("Accuracy:" ,metrics.accuracy_score(Y_test,Y_pred))
print("Precision:",metrics.precision_score(Y_test,Y_pred))


# In[48]:


import pickle


# In[49]:


with open('mark350_pickle','wb') as f:
    pickle.dump(clf1,f)


# In[50]:


with open('mark350_pickle','rb') as f:
    mp = pickle.load(f)


# In[51]:


mp.predict(X_test)


# In[52]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import  numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


# In[53]:


dataframe = pd.read_csv("Downloads\Mark1.csv")


# In[54]:


dataframe.head()


# In[56]:


labels = np.array(dataframe['outcome'])
labels


# In[57]:


feature = dataframe.drop('outcome', axis=1)
feature[:5]


# In[58]:


X_train, X_test, y_train, y_test = train_test_split(feature, labels, test_size=0.3, random_state=42)


# In[59]:


print("Training data", X_train.shape)
print("Testing data",X_test.shape)


# In[60]:


rf = RandomForestClassifier(n_estimators=1000, random_state=10)


# In[61]:


rf.fit(X_train, y_train)


# In[62]:


prediction = rf.predict(X_test)
prediction


# In[63]:


print("Predicted data: ", prediction)
print("Actual data: ", y_test)


# In[65]:


acc = accuracy_score(prediction, y_test)
acc


# In[ ]:




