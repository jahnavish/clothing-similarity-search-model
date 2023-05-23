#!/usr/bin/env python
# coding: utf-8

# # CLOTHING SIMILARITY SEARCH MODEL
# 
# This is an ***Clothing Similarity Search Model*** that receives text describing a clothing item and returns a ranked list of links to similar items from e-commerce websites. The data is initially **pre-processed**, then the **similarity is computed** before the model **returns a ranked list of similar clothing items**.

# In[1]:


# import the necessary libraries

import numpy as np
import pandas as pd
import json


# In[2]:


clothes = pd.read_csv('clothing_dataset.csv')


# In[3]:


clothes.isnull().sum() # to check for any null values


# In[4]:


clothes.duplicated().sum() # to check duplicates


# In[5]:


clothes_mod = clothes
clothes_mod['desc'] = clothes_mod['desc'].apply(lambda x:x.split()) # splits the words in the description and puts them in a list


# In[6]:


# remove any spaces and hyphens in the description, say Crew Neck or 
# Crew-Neck to CrewNeck to avoid errors in the search system

clothes_mod['desc'] = clothes_mod['desc'].apply(lambda x:[i.replace(" ", "") for i in x]) # spaces
clothes_mod['desc'] = clothes_mod['desc'].apply(lambda x:[i.replace("-", "") for i in x]) # hyphens


# In[7]:


# converting the list of description to a string

clothes_mod['desc'] = clothes_mod['desc'].apply(lambda x: " ".join(x))


# In[8]:


# convert the string into lowercase

clothes_mod['desc'] = clothes_mod['desc'].apply(lambda x:x.lower())


# In[9]:


clothes_mod['desc'].head()


# ### STEMMING
# 
# We will apply stemming on the data in order to normalize the text as there may be multiple variations of the same word.
# 
# nltk is a famous natural language processing library. Install nltk using **'pip install nltk'**.

# In[10]:


import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)


# In[11]:


clothes_mod['desc'] = clothes_mod['desc'].apply(stem)


# In[12]:


clothes_mod['desc']


# ### TEXT VECTORIZATION USING TF-IDF TECHNIQUE
# 
# Term Frequency-Inverse Document Frequency (TF-IDF) is used to score the relative importance of words in a document.
# 
# Term Frequency is the number of times a word appears in a document divded by the total number of words in the document. Every document has its own term frequency.
# 
# Inverse Document Frequency is the log of the number of documents divided by the number of documents that contain the word 'x'. Inverse data frequency determines the weight of rare words across all documents in the corpus.
# 
# TF-IDF is implemented through the tool provided by scikit-learn library, **'TfidfVectorizer'**, which will first need to be imported. It will be used to vectorize the input text given by the user.
# 
# ### TEXT VECTORIZATION USING BAG OF WORDS TECHNIQUE
# 
# In this approach, we look at the histogram of the words within the text, i.e. considering each word count as a feature. We will not be considering stop words (aka words that are used for sentence formation but add no value/contribution to the actual meaning of the sentence, like are, and, or, to, from, etc.)
# 
# Bag of Words is implemented through the tool provided by scikit-learn library, **'CountVectorizer'**, which will first need to be imported. It will be used to vectorize the items in the dataset.
# 
# ### COMPUTING THE COSINE SIMILARITY BETWEEN THE VECTORS
# 
# We will be calculating the Cosine Similarity of one vector with all the other vectors and repeat it for all the vectors
# 
# The smaller the angle is, the lesser the distance, therefore, the two vectors (clothing items) will be more similar. Cosine distance is inversely proportional to cosine similarity.
# 
# The cosime similarity can be performed manually as well but in this model, it will be computed using **'cosine_similarity'** which will need to be imported. The cosine_similarity method will compute the similarity between the input text and the items in the dataset.

# In[13]:


# import the necessary libraries

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[14]:


# the 'search_for_clothes' function takes the input text describing the clothing item
# and pre-processes it. it then uses TF-IDF technique to extract the key features from
# the text and uses Bag of Words technique to vectorize the items in the dataset.
# finally, it computes the similarity between the vectors and returns a ranked list of
# similar items

def search_for_clothes(text):
    input_text = text

    input_text = input_text.split() # pre-processing the input text
    input_text = [x.strip(" ") for x in input_text]
    input_text = [x.replace("-", "") for x in input_text]
    input_text = " ".join(input_text)

    input_text = input_text.lower()
    input_text = stem(input_text)

    vectorizer_text = TfidfVectorizer() # tf-idf to extract features
    vectors_text = vectorizer_text.fit_transform([input_text]).toarray()

    cv = CountVectorizer(max_features = 600, stop_words = 'english') # bag of words technique

    # there will be many 0 values in this. by default, CountVectorizer returns a SciPy sparse matrix
    # so we will convert it to a numpy array as we need it

    vectors_items = cv.fit_transform(clothes_mod['desc']).toarray()

    text_vector = np.zeros(vectors_items.shape)
    text_vector[:vectors_text.shape[0],:vectors_text.shape[1]] = vectors_text # making the vectors of equal length

    similarity = cosine_similarity(vectors_items, text_vector) # computing similarity
    similarity_score = sorted(list(enumerate(similarity)), reverse = True, key = lambda x: x[1][0]) # ranking it from most-to-least similar order

    data = []

    for i in similarity_score[0:9]:
        item = []
        temp_df = clothes_mod[clothes_mod['desc'] == clothes_mod.iloc[i[0]].desc] 
        item.extend(list(temp_df['link'].values))
        data.append(item)
    
    data = data[::-1]
    dict_data = {}
    
    for i in range(1, 9):
        dict_data[i] = data[i]
    
    dict_data = json.dumps(dict_data) # json responses of suggestions
    return dict_data # return ranked suggestions of similar items


# In[15]:


text = input("Enter the description of the clothing item: ")

search_for_clothes(text)