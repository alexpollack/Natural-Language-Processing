
# coding: utf-8

# ### import fasttext

# In[4]:

import fasttext


# ### import rest of needed modules

# In[2]:

import pandas as pd


# In[25]:

file = open('code-fall2019-a3/NLP_class/data5/training-data/training-data.txt', 'r', encoding='iso-8859-1')


# # Train the model on the full dataset for word embeddings with dimension 100

# In[17]:

# main model resulting in 0.65 score, dim = 100
model = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data.txt')


# #### the vectors have shape (100,)

# In[90]:

leng = str(len(model.words))
print('Number of words:', leng)
dim = str(model.get_word_vector("the").shape[0])
print('Dimension: ', dim)


# # Write the embeddings to a file in the needed format to be evalutated

# In[93]:

tofile = open("embeddings_dim.txt","w+")
tofile.write(str(leng + " " + dim + "\n")) 
for i in range(len(model.words)):
    embed = str(model.words[i]) + " " +" ".join(map(str,model.get_word_vector(model.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# ## Experiment with dimensions 

# #### dimension = 50

# In[4]:

# play with dimensions
model_dim50 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data.txt', dim=50)


# In[5]:

leng50 = str(len(model_dim50.words))
print('Number of words:', leng50)
dim50 = str(model_dim50.get_word_vector("the").shape[0])
print('Dimension: ', dim50)


# In[6]:

tofile = open("embeddings_dim50.txt","w+")
tofile.write(str(leng50 + " " + dim50 + "\n")) 
for i in range(len(model_dim50.words)):
    embed = str(model_dim50.words[i]) + " " +" ".join(map(str,model_dim50.get_word_vector(model_dim50.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# #### dimension = 25

# In[5]:

# play with dimensions
model_dim25 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data.txt', dim=25)


# In[7]:

leng25 = str(len(model_dim25.words))
print('Number of words:', leng25)
dim25 = str(model_dim25.get_word_vector("the").shape[0])
print('Dimension: ', dim25)


# In[8]:

tofile = open("embeddings_dim25.txt","w+")
tofile.write(str(leng25 + " " + dim25 + "\n")) 
for i in range(len(model_dim25.words)):
    embed = str(model_dim25.words[i]) + " " +" ".join(map(str,model_dim25.get_word_vector(model_dim25.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# ## Experiment with amount of data 
# full data is 26M words, half is 14M

# #### Half data, dimension = 25

# In[38]:

# play with dimensions
model_dim25 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_50perc.txt', dim=25)


# In[39]:

leng25 = str(len(model_dim25.words))
print('Number of words:', leng25)
dim25 = str(model_dim25.get_word_vector("the").shape[0])
print('Dimension: ', dim25)


# In[40]:

tofile = open("half_data_embeddings_dim25.txt","w+")
tofile.write(str(leng25 + " " + dim25 + "\n")) 
for i in range(len(model_dim25.words)):
    embed = str(model_dim25.words[i]) + " " +" ".join(map(str,model_dim25.get_word_vector(model_dim25.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# #### Half data, dimension = 50

# In[41]:

# play with dimensions
model_dim50 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_50perc.txt', dim=50)


# In[42]:

leng50 = str(len(model_dim50.words))
print('Number of words:', leng50)
dim50 = str(model_dim50.get_word_vector("the").shape[0])
print('Dimension: ', dim50)


# In[43]:

tofile = open("half_data_embeddings_dim50.txt","w+")
tofile.write(str(leng50 + " " + dim50 + "\n")) 
for i in range(len(model_dim50.words)):
    embed = str(model_dim50.words[i]) + " " +" ".join(map(str,model_dim50.get_word_vector(model_dim50.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# #### Half data, dimension = 100

# In[44]:

# main model resulting in 0.65 score, dim = 100
model = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_50perc.txt')


# In[45]:

leng = str(len(model.words))
print('Number of words:', leng)
dim = str(model.get_word_vector("the").shape[0])
print('Dimension: ', dim)


# In[46]:

tofile = open("half_data_embeddings_dim.txt","w+")
tofile.write(str(leng + " " + dim + "\n")) 
for i in range(len(model.words)):
    embed = str(model.words[i]) + " " +" ".join(map(str,model.get_word_vector(model.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# # quarter of data
# 6M words

# #### quarter of data, dimension = 25

# In[50]:

# play with dimensions
model_dim25 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_25perc.txt', dim=25)


# In[51]:

leng25 = str(len(model_dim25.words))
print('Number of words:', leng25)
dim25 = str(model_dim25.get_word_vector("the").shape[0])
print('Dimension: ', dim25)


# In[52]:

tofile = open("quarter_data_embeddings_dim25.txt","w+")
tofile.write(str(leng25 + " " + dim25 + "\n")) 
for i in range(len(model_dim25.words)):
    embed = str(model_dim25.words[i]) + " " +" ".join(map(str,model_dim25.get_word_vector(model_dim25.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# #### quarter of data, dimension = 50

# In[53]:

# play with dimensions
model_dim50 = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_25perc.txt', dim=50)


# In[54]:

leng50 = str(len(model_dim50.words))
print('Number of words:', leng50)
dim50 = str(model_dim50.get_word_vector("the").shape[0])
print('Dimension: ', dim50)


# In[55]:

tofile = open("quarter_data_embeddings_dim50.txt","w+")
tofile.write(str(leng50 + " " + dim50 + "\n")) 
for i in range(len(model_dim50.words)):
    embed = str(model_dim50.words[i]) + " " +" ".join(map(str,model_dim50.get_word_vector(model_dim50.words[i])))+"\n"
    tofile.write(embed)
tofile.close()


# #### Half data, dimension = 100

# In[56]:

# main model resulting in 0.65 score, dim = 100
model = fasttext.train_unsupervised('code-fall2019-a3/NLP_class/data5/training-data/training-data_25perc.txt')


# In[57]:

leng = str(len(model.words))
print('Number of words:', leng)
dim = str(model.get_word_vector("the").shape[0])
print('Dimension: ', dim)


# In[58]:

tofile = open("quarter_data_embeddings_dim.txt","w+")
tofile.write(str(leng + " " + dim + "\n")) 
for i in range(len(model.words)):
    embed = str(model.words[i]) + " " +" ".join(map(str,model.get_word_vector(model.words[i])))+"\n"
    tofile.write(embed)
tofile.close()

