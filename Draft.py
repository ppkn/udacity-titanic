
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-pastel')


# # Investigating the Titanic Dataset

# ## Question Brainstorm

# * What factors made people more likely to survive?
# * Do the genders of different socio-economic classes vary? For example, are there more men in first class than women?
# * Is the age of a higher socio-economic class less likely to be estimated?
# * Are certain decks more likely to survive?
# * Was it more expensive to get on at different ports?

# In[2]:

titanic_data = pd.read_csv('titanic-data.csv')
titanic_data.head()


# In[3]:

def get_deck(cabin):
    if pd.isnull(cabin):
        return np.NaN
    return cabin[:1]


# In[4]:

titanic_data['Deck'] = titanic_data['Cabin'].apply(get_deck)


# In[5]:

titanic_data['Deck'].unique()


# In[6]:

titanic_data[titanic_data['Deck'] == 'T']


# In[7]:

deck_survived = titanic_data.groupby('Deck')['Survived'].sum()


# In[8]:

deck_counts = titanic_data.groupby('Deck')['PassengerId'].count()


# In[9]:

fig, axs = plt.subplots(1,2)
deck_survived.plot(kind='bar', title='Survivor Counts', ax=axs[0]);
deck_counts.plot(kind='bar', title='Overall Counts', ax=axs[1]);


# In[10]:

style.use('seaborn-pastel')
def label_survived(num):
    word_labels = {0: 'Died', 1: 'Survived'}
    return word_labels[num]
pd.crosstab(titanic_data['Deck'], titanic_data['Survived'].apply(label_survived)).plot.bar(stacked=True);
plt.legend(frameon=True);


# In[11]:

pd.crosstab(titanic_data['Pclass'], titanic_data['Survived'].apply(label_survived), normalize=0).plot.bar(stacked=True);
# plt.legend(frameon=True);


# In[12]:

pd.crosstab(titanic_data['Pclass'], titanic_data['Survived'].apply(label_survived), normalize=0)['Survived'].plot.bar();
# plt.legend(frameon=True);


# In[13]:

pd.crosstab(titanic_data['Sex'], titanic_data['Survived'], normalize=0, margins=True)


# In[14]:

pd.crosstab(titanic_data['Sex'], titanic_data['Survived'], normalize=1)


# In[15]:

pd.crosstab(titanic_data['Sex'], titanic_data['Survived'].apply(label_survived), normalize=0).plot.bar(stacked=True);


# In[16]:

# nifty trick found at: http://themrmax.github.io/2015/11/13/grouped-histograms-for-categorical-data-in-pandas.html
ag = titanic_data.groupby('Pclass')['Embarked'].value_counts().sort_index()
ag.unstack().plot(kind='bar', subplots=True)


# In[17]:

titanic_data.hist(by='Embarked', column='Fare');


# In[24]:

# binwidth trick found: http://stackoverflow.com/questions/6986986/bin-size-in-matplotlib-histogram
binwidth = 50
binned = np.arange(min(titanic_data['Fare']), max(titanic_data['Fare']), binwidth)
titanic_data.groupby('Embarked')['Fare'].plot.hist(normed=True, alpha=0.66, bins=binned);
plt.legend();


# In[19]:

by_survivors = titanic_data.groupby('Survived')
by_class = titanic_data.groupby('Pclass')
by_class['Fare'].mean().plot(kind='bar')


# In[20]:

titanic_data.plot.scatter(x='Age', y='Fare')


# In[21]:

def bin_width(width, column):
    min_val = titanic_data[column].min()
    max_val = titanic_data[column].max()
    return np.arange(min_val, max_val, width)
titanic_data.hist(by='Pclass', column='Age', bins=bin_width(5, 'Age'));
plt.xlabel('Age in Years');


# In[22]:

pd.crosstab(titanic_data['Pclass'], titanic_data['Sex'], normalize=0).plot.bar(stacked=True);


# In[23]:

import math
def is_estimated(age):
    if pd.isnull(age):
        return True
    else:
        return age == math.ceil(age)
def normalize(series):
    return 1 + (series - series.min()) / (series.max() - series.min())
titanic_data['Estimated'] = titanic_data['Age'].apply(is_estimated)
estimated_pass = titanic_data[titanic_data['Estimated']]
normalize(estimated_pass.groupby('Pclass').size()).plot(kind='bar');


# In[ ]:

from pandas.tools.plotting import scatter_matrix


# In[29]:

scatter_matrix(titanic_data, figsize=(8,8), diagonal='kde');

