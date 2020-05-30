#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


dataset = pd.read_excel("covid19-symptoms-dataset.xlsx")


# In[3]:


dataset.columns


# In[4]:


dataset.head()


# In[5]:


y = dataset['Infected with Covid19']


# In[6]:


y.value_counts()


# In[7]:


Y = pd.get_dummies(y,drop_first=True)


# In[8]:


Y


# In[9]:


X = dataset[['Dry Cough','High Fever','Sore Throat','Difficulty in breathing']]


# In[ ]:





# In[10]:


Y.shape


# In[11]:


X.shape


# In[12]:


X.info()


# In[13]:


from keras.optimizers import Adam


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# In[16]:


from keras.models import Sequential


# In[17]:


model = Sequential()


# In[18]:


from keras.layers import Dense


# In[19]:


model.add(Dense(units=3, input_dim=4, activation='relu' ))


# In[20]:


model.add(Dense(units=3, activation='relu'))


# In[21]:


model.add(Dense(units=1,  activation='sigmoid' ))


# In[ ]:





# In[22]:


model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'] )


# In[23]:


X_train.shape


# In[24]:


y_train.shape


# In[ ]:





# In[25]:


model.fit(X_train,y_train , epochs=500)


# In[26]:


df_loss = pd.DataFrame(model.history.history)


# In[27]:


df_loss.plot()


# In[28]:


df=pd.DataFrame(df_loss)


# In[29]:


d=df.to_numpy()


# In[30]:


type(d)


# In[31]:


accuracy = max(d[:, 1])


# In[32]:


accuracy


# In[33]:


file1 = open("result.txt", "w")
file1.write(str(accuracy*100))
file1.close()


# In[ ]:




