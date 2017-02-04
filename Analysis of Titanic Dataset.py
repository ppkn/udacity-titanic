
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

# In[5]:

# Kaggle's website talks about the meaning of the
# 'Survived' column here: https://www.kaggle.com/c/titanic/data
# -- 1 means they made it.
titanic_data.loc[titanic_data['Survived'] == 1, 'Sex']             .value_counts()


# Wow! More than twice as many women survived than men, but it's much easier to notice those types of relationships in a chart.

# In[6]:

titanic_data.loc[titanic_data['Survived'] == 1, 'Sex']             .value_counts()             .plot(kind='bar', rot=0)
plt.title('Num Survivors by Gender')
plt.savefig('figures/num_survivors_by_gender.png')


# But this might be deceptive. Maybe there were just more women on the titanic.

# In[19]:

def pipe_print(x):
    print(x)
    return x


# In[20]:

titanic_data['Sex'].value_counts()                    .sort_index()                    .pipe(pipe_print)                    .plot(kind='bar', rot=0)
plt.title('Num Passengers by Gender')
plt.savefig('figures/num_by_gender.png')


# Normalize view so that it represents proportion of population

# In[9]:

def percent_survived(group, by='Survived', df=titanic_data):
    return pd.crosstab(df[group], df[by], normalize=0)              .loc[:, 1]

normalized_mf_survivors = percent_survived('Sex')
normalized_mf_survivors.plot(kind='bar', rot='0')
plt.title('% Survivors by Gender')
plt.savefig('figures/per_survivors_by_gender.png')
normalized_mf_survivors


# It looks like an even higher percentage of females survived compared to males

# ## Women and Children First
# 
# Let's see how men, women, and children compare.

# In[10]:

def to_mwch(df):
    if df['Age'] < 14:
        return 'child'
    else:
        return df['Sex']

titanic_data.assign(MWCh=lambda x:                             x.apply(to_mwch, axis=1))             .pipe((percent_survived, 'df'), 'MWCh')             .pipe(pipe_print)             .plot(kind='bar', rot='0')

plt.title('% Survivors by Men, Women, and Children')
plt.savefig('figures/per_survivors_mwch.png')


# ## Stay classy 1912
# > Woman: I didn't know we had a king!  I thought we were autonomous collective.  
# > Dennis: You're fooling yourself!  We're living in a dictatorship!  A self-perpetuating autocracy in which the working **classes**--  
# > Woman: There you go, bringing **class** into it again...  
# 
# Women and children first, but maybe class had something to do with it too. The lower class may have some injuries and would have been helped.

# In[11]:

percent_survived('Pclass').plot(kind='bar', rot=0)


# ## Cabin
# Maybe their deck had something to do with it.

# In[12]:

def to_deck(cabin):
    if pd.isnull(cabin):
        return 'No Info'
    else:
        return cabin[0]


# In[13]:

# titanic_data['Cabin'].apply(to_deck)
titanic_data.assign(Deck=lambda x:                     x['Cabin'].apply(to_deck))             .pipe((percent_survived, 'df'), 'Deck')             .reindex(list('ABCDEFGT') + ['No Info'])             .pipe(pipe_print)             .plot(kind='bar')


# In[14]:

# https://www.encyclopedia-titanica.org/titanic-victim/stephen-weart-blackwell.html
titanic_data[titanic_data['Cabin'] == 'T']


# ## The Price is Right
# 
# I was wondering if some people had to pay different prices for different classes.

# In[15]:

titanic_data.pivot(columns='Pclass', values='Fare')             .plot(kind='box')
plt.title('Spread of Prices by Class (with outliers)')
plt.savefig('figures/class_price_spread_w_outliers.png')


# In[16]:

def remove_outliers(series):
    iqr = series.quantile(0.75) - series.quantile(0.25)
    median = series.quantile(0.5)
    bools = (series > median - 1.5 * iqr) & (series < median + 1.5 * iqr)
    print('Median of {}: {}'.format(series.name, median))
    return series[bools]
titanic_data.pivot(columns='Pclass', values='Fare')             .apply(remove_outliers)             .plot(kind='box')
plt.title('Spread of Prices by Class')
plt.savefig('figures/class_price_spread.png')


# Maybe something to do with the deck they are on?

# In[17]:

(titanic_data.assign(Deck=lambda x:
                    x['Cabin'].apply(to_deck))
            .pivot(columns='Deck', values='Fare')
            [list('ABCDEFG') + ['No Info']]  # Remove T because only one value
            .apply(remove_outliers)
            .plot(kind='box'))
print('Value  of T: {}'
      .format(titanic_data.loc[titanic_data['Cabin'] == 'T', 'Fare']
             .iloc[0]))
plt.title('Spread of Prices by Deck')
plt.savefig('figures/deck_price_spread.png')


# How about ports because I want to mix things up with a line graph

# In[18]:

(titanic_data.groupby(['Embarked', 'Pclass'])
            ['Fare'].median()
            .unstack(level=1)
            .reindex(list('SCQ'))
        .plot(subplots=True, figsize=(6, 6)))
plt.gcf().suptitle('Price from Ports: By Class')
plt.gcf().savefig('figures/price_from_ports.png')

