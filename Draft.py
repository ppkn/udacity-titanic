
# coding: utf-8

# In[22]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Investigating the Titanic Dataset

# ## Question Brainstorm

# * What factors made people more likely to survive?
# * Do the genders of different socio-economic classes vary? For example, are there more men in first class than women?
# * Is the age of a higher socio-economic class less likely to be estimated?
# * Are certain decks more likely to survive?
# * Was it more expensive to get on at different ports?

# In[13]:

titanic_data = pd.read_csv('titanic-data.csv')
titanic_data.head()


# In[8]:

def get_deck(cabin):
    if pd.isnull(cabin):
        return np.NaN
    return cabin[:1]


# In[15]:

titanic_data['Deck'] = titanic_data['Cabin'].apply(get_deck)


# In[18]:

titanic_data['Deck'].unique()


# In[20]:

titanic_data[titanic_data['Deck'] == 'T']


# In[57]:

deck_survived = titanic_data.groupby('Deck')['Survived'].sum()


# In[105]:

fig, axs = plt.subplots(1,2)
fig.set_figwidth(16)
deck_survived.plot(kind='bar', title='Survivor Counts', ax=axs[0]);
deck_counts.plot(kind='bar', title='Overall Counts', ax=axs[1]);


# In[69]:

deck_counts = titanic_data.groupby('Deck')['PassengerId'].count()


# In[103]:

deck_counts.plot(kind='bar');


# In[ ]:



