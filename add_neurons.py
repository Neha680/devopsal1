#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import tensorflow.compat.v1 as tf
# tf.disable_v2_behaviour()                            ...ye function k liye file bnari hu bss


# In[3]:


from keras.models import load_model
classifier = load_model('pretraine.h5')


# In[4]:


# Here we freeze the last 3 layers 
# Layers are set to trainable as True by default
for layer in classifier.layers:
    layer.trainable = False


# In[5]:


# Let's print our layers 
for (i,layer) in enumerate(classifier.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)


# In[ ]:


# Re-loads the classifier model without the top or FC layers
classifier = classifier(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))


# In[ ]:


#double slash kyu use krte te bhul gyi.. single \ specifier ki tarha treat karte h... like \n \t...ok but jb data set dete h tb toh single use krte h...shyd


# In[ ]:




