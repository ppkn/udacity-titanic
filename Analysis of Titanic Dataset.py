
# coding: utf-8

# # Analysis of Kaggle's Titanic Dataset
# Author: Daniel Pipkin

# Write something about the Titanic here. The dataset contains demographic and passenger information about passengers on the Titanic.

# ## Posing Questions

# The two questions we are mainly focusing on:
# * What factors made people more likely to survive? (Women, Children, Upper Class)
#     More Women survived. What about children? More first-class passengers survived. Is there an alternative explanation.
# * Was ticket price based on more than passenger class?

# In[3]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn-pastel')

