
# coding: utf-8

# # Analysis of Kaggle's Titanic Dataset
# Author: Daniel Pipkin
# 
# Write something about the Titanic here. The dataset contains demographic and passenger information about passengers on the Titanic.

# In[1]:

# Make plots show up in the notebook
get_ipython().magic('matplotlib inline')

# Import needed libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

# Bring an artist's touch to matplotlib's default styles
style.use('seaborn-pastel')


# ## Posing Questions
# 
# The two questions we are mainly focusing on:
# * What factors made people more likely to survive? (Women, Children, Upper Class)
#     More Women survived. What about children? More first-class passengers survived. Is there an alternative explanation.
# * Was ticket price based on more than passenger class?
# 
# To answer these questions, I'm going to follow a simple loop:
# 1. Hypothesize an explanation for the question.
# 2. Gather items needed to investigate the hypothesis.
# 3. Investigate the relationship with one or more visualizations.
# 4. Draw a conclusion, then pose another hypothesis for further investigation.

# ## "I'll never let go, Jack!"
# 
# So what factors did contribute to the survival of some passengers? The one thing I remember from Titanic is that Jack dies and Rose lives. (That, and Leonardo Decaprio toasting like [he does in every movie](http://www.vulture.com/2013/06/gif-history-of-leo-dicaprio-raising-glasses.html)). Let's use this as a starting point.  
# **Hypothesis:** Women survived more than men  
# Now it's time to gather the needed information.

# In[2]:

# Import the titanic data
titanic_data = pd.read_csv('titanic-data.csv')


# This is what the first couple rows of data looks like.

# In[3]:

titanic_data.head()


# It seems like the `Sex` column uses '`male`' and '`female`' values. Let's make sure that looks right.

# In[4]:

titanic_data['Sex'].unique()


# Great! No surprises there. Now let's see how many of each group survived.

# In[59]:

# Kaggle's website talks about the meaning of the
# 'Survived' column here: https://www.kaggle.com/c/titanic/data
# -- 1 means they made it.
survivors_mf = titanic_data.loc[titanic_data['Survived'] == 1, 'Sex']
survivors_mf.value_counts()


# In[5]:

# Because 0 means no and 1 means yes, we can sum the `Survived` column to find total survivors
survivors_by_mf = titanic_data.groupby('Sex')['Survived'].sum()
survivors_by_mf


# Wow! More than twice as many women survived than men, but it's much easier to notice those types of relationships in a chart.

# In[43]:

survivors_by_mf.plot(kind='bar', rot=0)
plt.title('Num Survivors by Gender')
plt.savefig('figures/num_survivors_by_gender.png')


# By this might be deceptive. Maybe there were just more women on the titanic.

# In[47]:

passengers_by_mf = titanic_data['Sex']
passengers_by_mf.value_counts()                 .loc[['female', 'male']]                 .plot(kind='bar', rot=0)
plt.title('Num Passengers by Gender')
passengers_by_mf.value_counts()


# Normalize view so that it represents proportion of population

# In[84]:

def group_normalize(group, by='Survived', df=titanic_data):
    return pd.crosstab(df[group], df[by], normalize=0)

normalized_mf_survivors = group_normalize('Sex')                           .loc[:, 1]
normalized_mf_survivors.plot(kind='bar', rot='0')
plt.title('% Survivors by Gender')
plt.savefig('figures/per_survivors_by_gender.png')
normalized_mf_survivors


# It looks like an even higher percentage of females survived compared to males

# ## Women and Children First
# 
# Let's see how men, women, and children compare.

# In[90]:

def to_mwch(row):
    if row['Age'] < 14:
        return 'child'
    else:
        return row['Sex']

mwch = titanic_data.apply(to_mwch, axis=1)
with_mwch = titanic_data.assign(MWCh=mwch)

normalized_mwch_survivors = group_normalize('MWCh', df=with_mwch)                             .loc[:, 1]
normalized_mwch_survivors.plot(kind='bar', rot='0')
plt.title('% Survivors by Men, Women, and Children')
plt.savefig('figures/per_survivors_mwch.png')
normalized_mwch_survivors

