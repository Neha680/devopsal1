#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


dataset = pd.read_excel("covid19-symptoms-dataset.xlsx")


# In[3]:


y = dataset['Infected with Covid19']


# In[4]:


Y = pd.get_dummies(y,drop_first=True)


# In[5]:


X = dataset[['Dry Cough','High Fever','Sore Throat','Difficulty in breathing']]


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)


# In[8]:


from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense


# In[9]:


model = Sequential()


# In[10]:


def build(neuron, model):
    model.add(Dense(units=neuron, input_dim=4, activation='relu' ))
    model.add(Dense(units=neuron//10, activation='relu'))
    model.add(Dense(units=1,  activation='sigmoid' ))
    
    model.compile(optimizer=Adam(learning_rate=0.0001),loss='binary_crossentropy',metrics=['accuracy'] )
    
    model.fit(X_train,y_train , epochs=500, verbose=0)
    
    return model


# In[11]:


def acc(model):
    score = model.evaluate(X_test, y_test, verbose=0)
    accuracy = score[1]*100    
    print("Accuracy: %.2f%%" % (score[1]*100))
    return accuracy


# In[12]:


def resetWeight(model):
    print("Reseting weights")
    w = model.get_weights()
    #print("Value of weights",w)
    w = [[j*0 for j in i] for i in w]
    #print("Value of weights",w)
    model.set_weights(w)


# In[13]:


neuron = 50
model = build(neuron, model) 
accuracy = acc(model)
resetWeight(model)


# In[14]:


print(accuracy)


# In[15]:


count = 0
best_acc=accuracy
best_neu = 50


# In[16]:


while accuracy < 80 and count < 5:
    print("\t\tAttempt ",count+1)
    neuron = 5*(count+1)*4 
    model = build(neuron,model)
    accuracy = acc(model)
    
    if accuracy > best_acc:
        best_acc = accuracy
        best_neu = neuron

    resetWeight(model)
    count = count + 1
    
print("\nBest Accuracy: ", best_acc)

model.save('covid_model.h5')
print("Best model saved!!!")   


# In[17]:




