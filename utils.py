# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:38:46 2020

@author: Ankita
"""

from sklearn.feature_selection import RFE
from sklearn.linear_model import  LogisticRegression
import matplotlib.pyplot as plt
import h5py
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
    


def getTopFeaturesChi2(df_dat, numericVariables, classVariable,k):
    
    '''
    Top 10 feature selection from the passed df_dat with respect to
    their association to the classVariable or target variable.
    selector = feature_selection.SelectKBest(score_func=  
               feature_selection.f_regression, k=10).fit(X,y)
    pvalue_selected_features = feature_names[selector.get_support()]
    ''' 
    bestfeatures = SelectKBest(score_func=chi2,k=k)
    bestfeature_fit = bestfeatures.fit(df_dat[numericVariables],df_dat[classVariable])
    dfscores = pd.DataFrame(bestfeature_fit.scores_)
    dfcolumns = pd.DataFrame(numericVariables)
    #concat two dataframes for better visualization 
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['variableName','Score']  #naming the dataframe columns
    print(featureScores.nlargest(k,'Score'))  #print 10 best features
    return featureScores.nlargest(k,'Score').variableName.tolist()

def getTopFeaturesByTreeClassifier(df_dat, numericVariables, classVariable, n):
    '''
     Inbuilt class feature_importances from a tree based classifiers
    '''
    from sklearn.ensemble import ExtraTreesClassifier
    import matplotlib.pyplot as plt
    model = ExtraTreesClassifier(random_state=0)
    model.fit(df_dat[numericVariables],df_dat[classVariable])
    #print(model.feature_importances_) 
    #plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=numericVariables)
    #feat_importances.nlargest(n).plot(kind='barh')
    #plt.show()
    
    feat_importances = pd.DataFrame(model.feature_importances_)
    dfcolumns = pd.DataFrame(numericVariables)

     #concat two dataframes  
    featureScores = pd.concat([dfcolumns,feat_importances],axis=1)
    featureScores.columns = ['TreeClassifier_Variable','TreeClassifier_Score']
    return featureScores.nlargest(n,'TreeClassifier_Score').reset_index(drop=True)

def getTopNFeaturesUsingRFELogitReg(df_dat, numericVariables, classVariable,n=10):
    '''
    Use recursive feature elimination for logistic regression
    to choose most important features
    '''
    model = LogisticRegression(random_state=0)
    rfe = RFE(model, n)
    rfeFit = rfe.fit(df_dat[numericVariables],df_dat[classVariable])
    TopFeatures=df_dat[numericVariables].columns[np.where(rfeFit.support_ == True)]
    #return pd.DataFrame(data=TopFeatures, index = "REF_logitReg_Variable")
    return pd.DataFrame(TopFeatures,columns=['REF_Variable'])

def calculateFeatureImportance(df_dat, numericVariables, classVariable,n):
    '''
    Runs three different feature selection and outputs the selected
    features as a dataframe along side importance measure if any.
    '''
    df_Chi2=getTopFeaturesChi2(df_dat,numericVariables,classVariable,n)
    df_TreeClass=getTopFeaturesByTreeClassifier(df_dat, numericVariables, classVariable, n)
    df_topFeaturs_RFE=getTopNFeaturesUsingRFELogitReg(df_dat, numericVariables, classVariable, n)
    return pd.concat([df_Chi2,df_TreeClass,df_topFeaturs_RFE], axis = 1)
    

    

def getClasswiseNullValuePercent(df_dat, classVariable):
    '''
    Try to comment here for what the function is doing
    '''
    classWiseNullCounts = df_dat.drop(classVariable, 1).isnull().groupby(df_dat[classVariable], sort=False).sum().reset_index()
    classWiseTotalCounts = df_dat.drop(classVariable, 1).groupby(df_dat[classVariable], sort=False).sum().reset_index()
    percentNull= ((classWiseNullCounts/classWiseTotalCounts)*100).T
    # Remove variables with zero missing values 
    percentNull=percentNull[percentNull>0]
    percentNull=percentNull[percentNull.notnull().all(1)]

    return percentNull

def shrikOutliers_topQTL(np_array, quartile):
    '''
    Return the passed np array where outliers are shrinked to 
    top quartile or third quartile as defied by nth quartile
    e.g., 25th = 25, 75th = 75, 99th = 99 etc.
    
    '''
    quartile_value = np.percentile(np_array, [quartile])
    return np.where((np_array > quartile_value))[0],quartile_value


def RFM_model(data, R,F,M):
    data=data.sort_values(by=R,ascending=False)
    data['RECENCY_Q']=pd.qcut(data[R].rank(method='first'),4,labels=[4,3,2,1])
    data=data.sort_values(by=F,ascending=False)
    data['FREQUENCY_Q']=pd.qcut(data[F].rank(method='first'),4,labels=[4,3,2,1])
    data=data.sort_values(by=M,ascending=False)
    data['MONETARY_Q']=pd.qcut(data[M].rank(method='first'),4,labels=[4,3,2,1])
    data.groupby('FREQUENCY_Q').mean()[[M,F,R]]
    data.groupby('MONETARY_Q').mean()[[M,F,R]]
    data.groupby('RECENCY_Q').mean()[[M,F,R]]

    data.groupby(['RECENCY_Q','FREQUENCY_Q','MONETARY_Q']).count()[['SUBS_ID']]
    data_1=data[data['churn']==1]
    print(data_1['churn'].isnull().sum())
    #data_2=data[data['churn']==0]
    res=data_1.groupby(['RECENCY_Q','FREQUENCY_Q','MONETARY_Q']).sum()[['churn','MRP']]
    #res_non_churn=data_2.groupby(['RECENCY_Q','FREQUENCY_Q','MONETARY_Q']).sum()[['churn','MRP']]
    
    res['avg']=res['MRP']/res['churn']
    res_sorted=res.sort_values(by=['churn'],ascending=False)
    print(res_sorted)
    
def cramers_v(x, y):
    '''
    returns Square root of Cramer's V to suggest
    intercorrelation of two discrete 
    variables x and y. Cramér's V is computed by taking 
    the square root of the chi-squared statistic divided 
    by the sample size and the minimum dimension minus 1

    Cramér's V varies from 0 (corresponding to no association 
    between the variables) to 1 (complete association) 
    and can reach 1 only when each variable is completely 
    determined by the other.
    '''
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    cal=np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
    return cal


def getClasswiseStats(df_dat,classVariable, columnsToGetStatsFor):
    '''
    Function to get the classwise (classVariable ) 
    descriptive stats for the columns (columnsToGetStatsFor)
    of the dataframe passed (df_dat).
    '''
    #df_dat_All_stats = df_dat.groupby(classVariable).describe(include = 'all').T
    
    df_stats = df_dat.groupby(classVariable)[columnsToGetStatsFor].describe().T
    return df_stats

def reduceNumOfCategoriesByMaxCategorySize(df_dat, categoryColName):
    '''
    Function to reduce the number of categories for the categoryColName
    in the dataframe df_dat. Number of categories to be combined into OTHERS
    are detected dynamically based on trend in the percent change with respect 
    to previous category when ordered in descending order of counts. 
    
    Categories with decreasing percent change with respect to previous 
    category are kept as is, and categories from 
    the increased percent change onwards are combined into others. 
    
    Returns the df_dat with shrinked categories 
    '''
    df_counts=pd.DataFrame(df_dat[categoryColName].value_counts())
    df_pct_change=df_counts[categoryColName].pct_change(fill_method='ffill').pct_change()
    cat_first_Positive_pct_change=df_pct_change[df_pct_change>0.0].index[0]
    cat_last= df_pct_change.index[-1]

    df_dat.replace(
            df_pct_change[cat_first_Positive_pct_change:cat_last]
            .index.values,'Others', inplace = True)
    return df_dat


    
def missing_count(data):
    l=[]
    for i in data.columns:
        l.append((i,data[i].isnull().sum()))
    return l

  

   

def fix_upper(col):
    '''
    FOR AON fix high values outliers to mean+-std. 
    This is an understimation for those on the 
    network for long.  
    Churn is more concerning for relatively recent
    subs, hence bringing AON for those who are with us for 
    long would have positive impact on the mode.
    '''
    mean_plus_std_AON=sum(col.describe().loc[['std','mean']])
    Upper_QTLR_AON=col.describe().loc['75%']
    col[col>Upper_QTLR_AON]=mean_plus_std_AON
    return col
def fix_outliers(col):
    '''
    FOR AON fix high values outliers to mean+-std. 
    This is an understimation for those on the 
    network for long.  
    Churn is more concerning for relatively recent
    subs, hence bringing AON for those who are with us for 
    long would have positive impact on the mode.
    '''
    elements = np.array(col)

    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)
    mean_plus_std_AON=sum(col.describe().loc[['std','mean']])
    Upper_QTLR_AON=col.describe().loc['75%']
    col[col>Upper_QTLR_AON]=mean_plus_std_AON
    Lower_QTLR_AON=col.describe().loc['25%']
    col[col<Lower_QTLR_AON]=mean-sd
    
    return col


def splitData(data,ratio):
    '''splitting dtaa in train and test with ratio being proportion
    of train data
    '''
    train_data=data.iloc[:int(data.shape[0]*ratio),]
    test_data=data.iloc[int(data.shape[0]*ratio) :,]
    return (train_data,test_data)


def convert_cat2float(X_train,X_test):
    
    '''Encoding Categorical variables'''
    d = defaultdict(LabelEncoder)
    convert_cols = X_train.select_dtypes(include=['object']).columns
    X_train[convert_cols] =X_train[convert_cols].apply(lambda x: d[x.name].fit_transform(x))
    # Inverse the encoded
    #fit.apply(lambda x: d[x.name].inverse_transform(x))
    # Using the dictionary to label future data
    X_test[convert_cols] = X_test[convert_cols].apply(lambda x: d[x.name].transform(x))
    np.save('dictionary.txt',d)
    file = open('dict.pkl', 'wb')
    pickle.dump(d,file)
    file.close()
    return X_train,X_test

#save d


def decile_report(X_test, probs, y_test, test_RES, segments):
    test_x_1 = pd.DataFrame(X_test)
    test_x_1.index = test_RES.index
    test_x_1['SUBS_ID']= test_RES[['SUBS_ID']]
    test_x_1['P_1'] = probs
    test_x_1['B'] = y_test['classVar'].values
    
    test_x_df = test_x_1.sort_values(by='P_1', ascending=False)
    test_x_df['sno'] = np.arange(len(test_x_df))
    # df.sort_values(by='sno\',ascending=False)
    test_x_df['Decile'] = pd.qcut(test_x_df['sno'], segments, labels=False)
    
    test_x_df['NonTarget'] = 1 - test_x_df.B
    
    grouped_val2 = test_x_df.groupby('Decile', as_index=False)
    
    agg1 = pd.DataFrame(grouped_val2.min().P_1, columns= ['min_prob'])
    agg1['min_prob'] = grouped_val2.min().P_1
    agg1['max_prob'] = grouped_val2.max().P_1
    agg1['Target'] = grouped_val2.sum().B
    agg1['Non_Target'] = grouped_val2.sum().NonTarget
    agg1['Total'] = agg1.Target + agg1.Non_Target
    agg2 = (agg1.sort_index(by = 'min_prob', ascending=False)).reset_index(drop = True)
    agg2['target_rate'] = (agg2['Target']/agg2.Total).apply('{0:.2%}'.format)
    agg2['capture_rate'] = (agg2.Target/test_x_df.B.sum()).apply('{0:.2%}'.format)
    agg2['cum_capture_rate'] = (agg2.Target/test_x_df.B.sum()).cumsum().apply('{0:.2%}'.format)
    #calculate KS statistics
    agg2['ks'] = np.round(((agg2.Target/test_x_df.B.sum()).cumsum()-(agg2.Non_Target/test_x_df.NonTarget.sum()).cumsum()),4)*100
    flag = lambda x: '<----' if x == agg2.ks.max() else ''
    agg2['max_ks'] = agg2.ks.apply(flag)
    return agg2,test_x_df
 
def roc_info(test_X_df):
    roc ={}
    for dec in test_X_df.Decile.unique():
        print(dec)
        try:
            g = test_X_df[test_X_df.Decile==dec].reset_index(drop=True)
            roc[dec] = roc_auc_score(list(g.B), list(g.P_1))
        except:
            roc[dec] = 'onecl'
            continue
    return roc

def getConfusionMatrixValues(y_true,y_pred):
    cm = pd.DataFrame(confusion_matrix(y_true,y_pred))
    TN,FP,FN,TP = cm[0][0],cm[1][0],cm[0][1],cm[1][1]
    TR,CR = 100*TP/(TP+FP),100*TP/(TP+FN)
    return TN,FP,FN,TP,TR,CR

def count_classwise_outliers(df,low_pct,upp_pct):
    Q1 = df.quantile(low_pct)
    Q3 = df.quantile(upp_pct)
    IQR = Q3 - Q1
    lower_out=(df<Q1).groupby(df['classVar']).sum().T
    lower_out['rat_churn_to_non_churn']=lower_out[1]/lower_out[0]
    lower_out.sort_values(by=['rat_churn_to_non_churn'], ascending=True,inplace=True)
    upper_out=(df>Q3).groupby(df['classVar']).sum().T
    upper_out['rat_churn_to_non_churn']=upper_out[1]/upper_out[0]
    upper_out.sort_values(by=['rat_churn_to_non_churn'], ascending=True,inplace=True)
    return lower_out, upper_out

def getClassificationReport(y_true,y_pred):
    
    return pd.DataFrame(classification_report(y_true, y_pred, output_dict=True))


def label_Days_to_expiry(row,var):
    if row[var] <2:
        return "0_<2"
    elif row[var] <5:
        return "1_<5"
    elif row[var] <10:
        return "2_<10"
    elif row[var] <20:
        return "3_<20"
    elif row[var] <30:
        return "4_<30"
    elif row[var] <60:
        return "5_<60"
    elif row[var] <90:
        return "6_<90"
    elif row[var] <180:
        return "7_<180"
    elif row[var] <365:
        return "8_<365"
    else:
        return "8_>365"
def label_Balance_bucket(row,var):
    if row[var] <0:
        return "0_low"
    elif row[var] <10:
        return "1_medium"
    else:
        return "2_high"

def label_data(row,var):
    if row[var] <5:
        return "0_<5"
    elif row[var] <100:
        return "1_<100"
    elif row[var] <1000:
        return "2_<1000"
    elif row[var] <5000:
        return "3_<5000"
    elif row[var] <10000:
        return "4_<10000"
    else:
        return "5_>10000"
# Label Encoding our target variable 

#One Hot Encoding of the Categorical features 
        '''
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
l=LabelEncoder() 
l.fit(data.Income) 

l.classes_ 
data.Income=Series(l.transform(data.Income))  #label encoding our target variable 
data.Income.value_counts() 

 

#One Hot Encoding of the Categorical features 
one_hot_workclass=pd.get_dummies(data.workclass) 
one_hot_education=pd.get_dummies(data.education) 
one_hot_marital_Status=pd.get_dummies(data.marital_Status) 
one_hot_occupation=pd.get_dummies(data.occupation)
one_hot_relationship=pd.get_dummies(data.relationship) 
one_hot_race=pd.get_dummies(data.race) 
one_hot_sex=pd.get_dummies(data.sex) 
one_hot_native_country=pd.get_dummies(data.native_country) 

#removing categorical features 
data.drop(['workclass','education','marital_Status','occupation','relationship','race','sex','native_country'],axis=1,inplace=True) 

 

#Merging one hot encoded features with our dataset 'data' 
data=pd.concat([data,one_hot_workclass,one_hot_education,one_hot_marital_Status,one_hot_occupation,one_hot_relationship,one_hot_race,one_hot_sex,one_hot_native_country],axis=1) 
'''
## for data
import pandas as pd
import numpy as np
## for plotting
import matplotlib.pyplot as plt
import seaborn as sns
## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
## for machine learning
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition
## for explainer
from lime import lime_tabular


dtf = dtf.set_index("Id")
dtf = dtf.rename(columns={"SalePrice":"Y"})




x = "Y"
fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False)
fig.suptitle(x, fontsize=20)
### distribution
ax[0].title.set_text('distribution')
variable = dtf[x].fillna(dtf[x].mean())
breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
variable = variable[ (variable > breaks[0]) & (variable < 
                    breaks[10]) ]
sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
des = dtf[x].describe()
ax[0].axvline(des["25%"], ls='--')
ax[0].axvline(des["mean"], ls='--')
ax[0].axvline(des["75%"], ls='--')
ax[0].grid(True)
des = round(des, 2).apply(lambda x: str(x))
box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", bbox=dict(boxstyle='round', facecolor='white', alpha=1))
### boxplot 
ax[1].title.set_text('outliers (log scale)')
tmp_dtf = pd.DataFrame(dtf[x])
tmp_dtf[x] = np.log(tmp_dtf[x])
tmp_dtf.boxplot(column=x, ax=ax[1])
plt.show()



x = "Y"
ax = dtf[x].value_counts().sort_values().plot(kind="barh")
totals= []
for i in ax.patches:
    totals.append(i.get_width())
total = sum(totals)
for i in ax.patches:
     ax.text(i.get_width()+.3, i.get_y()+.20, 
     str(round((i.get_width()/total)*100, 2))+'%', 
     fontsize=10, color='black')
ax.grid(axis="x")
plt.suptitle(x, fontsize=20)
plt.show()



cat, num = "FullBath", "Y"
fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False)
fig.suptitle(x+"   vs   "+y, fontsize=20)
            
### distribution
ax[0].title.set_text('density')
for i in dtf[cat].unique():
    sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
ax[0].grid(True)
### stacked
ax[1].title.set_text('bins')
breaks = np.quantile(dtf[num], q=np.linspace(0,1,11))
tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T
tmp = tmp[dtf[cat].unique()]
tmp["tot"] = tmp.sum(axis=1)
for col in tmp.drop("tot", axis=1).columns:
     tmp[col] = tmp[col] / tmp["tot"]
tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
### boxplot   
ax[2].title.set_text('outliers')
sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
ax[2].grid(True)
plt.show()