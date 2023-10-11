# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 00:12:38 2020

@author: Ankita
"""

#visualization

import matplotlib.pyplot as plt
import os
import h5py
import img2pdf
from sklearn.metrics import roc_auc_score
from matplotlib.pyplot import yticks, xticks, subplots, set_cmap
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing
import pickle
import gc
import pandas as pd
import numpy as np
#from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from matplotlib  import pyplot
import seaborn as sns
import scipy.stats as ss
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import img2pdf
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import calendar
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 10000)
from sklearn.neural_network import MLPClassifier
#from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import ensemble
import gc
import numpy as np
h = .02  # step size in the mesh
import logging

def plot_corr(data, size=11):
    corr = data.corr()
    fig, ax = subplots(figsize=(size, size))
    set_cmap("YlOrRd")
    ax.matshow(corr)
    xticks(range(len(corr.columns)), corr.columns)
    print(len(corr.columns))
    #ax.set_yticks(range(len(corr.columns)), corr.columns)
    fig.savefig("../out/corr.jpg")
    return ax 
    
def getCategoricalVariableWithTargetViz(data, categorical_vars):
    
    '''
    countplot for categorical variables vs target variable
    '''
    
    for i,cat in enumerate(categorical_vars):
        plt.figure(i)
        plt.figure(figsize=(12,8))
        sns.countplot(x=cat,hue='classVar',data=data)
        plt.xticks(rotation=90) 
        plt.savefig('../out/cat/{}.jpeg'.format(cat))
        plt.close()
def getCategoricalVariablePropWithTargetViz(data, categorical_vars):
    
    '''
    proportion for categorical variables vs target variable
    '''
    for i,cat in enumerate(categorical_vars):
        plt.figure(i)
        plt.figure(figsize=(12,8))
        props = data.groupby(cat)['classVar'].value_counts(normalize=True).unstack()
        props.plot(kind='bar', stacked='True') 
        plt.savefig('../out/cat/prop_{}.jpeg'.format(cat))
        plt.close()
def getNumericalVariableWithTargetViz(data, numerical_vars):
    
    '''
    distribution for numerical variables vs target variable
    '''
    
    churn_yes = data[data['classVar'] == 1]
    churn_no = data[data['classVar'] == 0]
    for i,cat in enumerate(numerical_vars):
        
        plt.figure(figsize=(12,8))
        plt.figure(i)
        sns.distplot(churn_no[cat],label='0')
        sns.distplot(churn_yes[cat],label='1')
        plt.legend(title='left',loc='best') 
        plt.savefig('../out/num_dist/{}.jpeg'.format(cat))
        plt.close()
def plotCorrelationMatrix(df_dat):
    '''
     Correlation states how the features are related to 
     each other or the target variable. 
     Correlation can be positive (increase in one value 
     of feature increases the value of the target variable) 
     or negative (increase in one value of feature decreases 
     the value of the target variable). 
     Heatmap makes it easy to identify which features 
     are most related to the target variable, we will 
     plot heatmap of correlated features using the seaborn library.
    '''
    corrmat = df_dat.corr().round(1)
    top_corr_features = corrmat.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(corrmat,annot=True,cmap="RdYlGn")
def CorrelationMatrixDiff(df_dat):
    '''
    check correlation matrix difference between
    churners and non-churners
    '''
    churner=df_dat[df_dat['classVar']==1]

    corrmat_churn = churner.corr().round(1)
    nonchurner=df_dat[df_dat['classVar']==0]
    
    corrmat_nonchurn = nonchurner.corr().round(1)
    corrmat_diff= corrmat_nonchurn- corrmat_churn
    
    top_corr_features = corrmat_churn.index
    plt.figure(figsize=(20,20))
    #plot heat map
    g=sns.heatmap(corrmat_diff,annot=True,cmap="RdYlGn")
   
def plotboxplotsfornumerical(df_dat,numerical_vars,flag):   
    
    churn_yes = df_dat[df_dat['classVar'] == 1]
    churn_no = df_dat[df_dat['classVar'] == 0]
    for i,cat in enumerate(numerical_vars):
        
        plt.figure(figsize=(12,8))
        plt.figure(i)
        fig, ax = plt.subplots()
        ax.set_title(cat)
        ax.boxplot([churn_yes[cat],churn_no[cat]])
        #ax.set_labels(['1','0'])
        if flag==0:
            plt.savefig('../out/box_plot_before/boxplot_{}.jpeg'.format(cat))
            plt.close()
        if flag==1:
            plt.savefig('../out/box_plot_after/boxplot_{}.jpeg'.format(cat))
            plt.close()
    
    
# function to save images in pdf
def img_to_pdf(filename):
# filename of pdf
    current_path = '../src/'
    
    with open(filename + ".pdf", "wb") as f:
        # set directory to factor plots
        os.chdir(r"../out")
        f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith(".jpeg")]))
    # change the directory back to current directory
    os.chdir(current_path)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------



plt.figure(figsize=(12,6))
p1=sns.kdeplot(train_set[train_set['sentiment']=='positive']['jaccard'],shade=True,color='r').set_title('Distributions')
p2=sns.kdeplot(train_set[train_set['sentiment']=='negative']['jaccard'],shade=True,color='b')
#------------------
p2=sns.distplot(train_set[train_set['sentiment']=='neutral']['jaccard'],kde=False)
#-----------------

top_words=Counter([i for j in positive['split_txt'] for i in j])
cd=pd.DataFrame(top_words.most_common(20))
cd.columns=['words','count']
cd.style.background_gradient(cmap='Greens')


import plotly.express as px
fig=px.treemap(ab,path=['Top_words'],values='count',title='Top_words')   #ab is the dataframe,
fig.show()
fig=px.bar(ab,x='count',y='Top_words',orientation='h',width=700,height=700,color='Top_words')
fig.show()


from palettable.colorbrewer.qualitative import Pastel1_7
plt.figure(figsize=(16,10))
my_circle=plt.Circle((0,0), 0.7, color='white')
plt.pie(Unique_Positive['count'], labels=Unique_Positive.words, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.title('DoNut Plot Of Unique Positive Words')
plt.show()




