# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 13:21:52 2020

@author: Ankita
"""

import glob
import numpy as np
import pandas as pd
import seaborn as sns
from TextPreProcessing import Preprocessing_twitter
from sklearn.feature_selection import chi2
import nltk.corpus
nltk.download('stopwords')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
from IPython.display import display
import spacy
import pickle

from SentimentAnalysis import SentimentAnalysis
from Visualizations import Visualizations_twitter
from finalprocessing import finalprocessing

# nltk.download('wordnet')

#to supress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns',None)

# reading file
#df = pd.read_excel("data/twitter_feeds_no_PII.xlsx")

#to read binary file 1.Upgarde pandas 2. install pyxlsb
#df = pd.read_excel("data/Idea_April_SM.xlsb",engine='pyxlsb')
df1 = pd.read_excel("data/Vodafone_June_SM.xlsb",engine='pyxlsb')
df1['MONTH']='JUN'
df2= pd.read_excel("data/Vodafone_April_SM.xlsb",engine='pyxlsb')
df2['MONTH']='APR'
df3= pd.read_excel("data/Vodafone_May_SM.xlsb",engine='pyxlsb')
df3['MONTH']='MAY'
df=pd.concat([df1,df2,df3])
df=df.reset_index().drop(['index'],axis=1)

df1 = pd.read_excel("data/Idea_June_SM.xlsb",engine='pyxlsb')
df1['MONTH']='JUN'
df2= pd.read_excel("data/Idea_April_SM.xlsb",engine='pyxlsb')
df2['MONTH']='APR'
df3= pd.read_excel("data/Idea_May_SM.xlsb",engine='pyxlsb')
df3['MONTH']='MAY'
df=pd.concat([df1,df2,df3])
df=df.reset_index().drop(['index'],axis=1)

class twitter_text_analytics:
    nlp = spacy.load('en_core_web_sm')

    def __init__(self):
        self.Activation_Deactivation=['Activation/Deactivation']
        self.Others=['AGR','Blogs/Web','Brand','Campaign','DND','Comparison - Other Brands','News','Other Brand','Press Release','Social Media Activity','Sponsorship','Temporary Downtime']
        self.Balance_Payment=['Balance/Recharge','Billing/Payment']
        self.Services=['My Vodafone App','Services','VAS','']
        self.Migration=['Migration','Sales Lead','SIM']
        self.Network = ['Network']
        self.Data = ['Data']
        self.Generic=['Tag Line','Generic']
        self.Recharge=['Tariff/Offers']
        
        
#        self.Payment_Balance = ["Balance deduction","Balance Tranfer","Benefit not credited - Balance/Recharge","FRC","Information - Balance/Recharge","Recharge Unsuccessful","Refund - Balance/Recharge","Unable to recharge","Unspecified Balance/Recharge issue","Bill Not Received","Billing dispute","Charging","Ebill","Information - Billing/Payment","Overall Services - Billing/Payment","Payment not credited","Personal Info","Refund - Billing/Payment","Safe Custody","Security Deposit","Unable to pay","Unspecified Billing/payment Issue","Balance Transfer"]
#        self.Service_related = ["Calling sevices","Information - Activation/Deactivation","Overall Services - Activation/Deactivation","Pack/Plan","SMS services","App not working","App Related","Services","DND - Activation/Deactivation","Customer Care","Idea Store"]
#        self.SIM = ["Activation/Deactivation","Damaged SIM","Duplicate SIM","Information - SIM","PUK blocked","SIM Delivery","SIM Lost","SIM Registration","SIM Upgradation","Unspecified SIM Issue","MNP Lead","MNP Process","MNP Threat","Port In","Aadhar Linking","Activation - New connection","Circle Change","Postpaid to Prepaid"]
#        self.SPAM = ["Unwanted Calls","Unwanted Emails","Unwanted SMSs","Junk"]
#        self.VAS_Plans = ["Dialer Tones","Information - VAS","International Roaming","Missed Call Alerts","National Roaming","Other Value Added Services","USSD Codes","Benefit not credited - Tariff/Offers","Information - Tariff/Offers","Offers/Promotions","Overlapped/Lapsed","Plan/Pack Change","Unspecified Tariff/Offers Issue","VAS"]
#        self.Others = ["Call Details","Postpaid - Generic","Undisclosed Concern","Unspecified Generic Issue","Allegations","Generic Mention","Issue related other operater","Smartphone","Website"]

    def reducing_customer_category(self,value):
        '''converting categories into main categories only

        Arguments:
            value {[string]} -- [category name]
        '''
        if value in self.Network:
            return "Network"
        if value in self.Data:
            return "Data"
        if value in self.Generic:
            return "Generic"
        if value in self.Recharge:
            return "Recharge"
        if value in self.Migration:
            return "Migration"
        if value in self.Services:
            return "Services"
        if value in self.Balance_Payment:
            return "Balance_Payment"
        if value in self.Others:
            return "Others"
        if value in self.Activation_Deactivation:
            return "Activation_Deactivation"
    
    
    def nested_categories(self,df):
        if (df['Customer_Service_Category']=='Network' and df['Customer Service- Sub Category (Case)']=='Coverage'):
            return 'Coverage'
        elif (df['Customer_Service_Category']=='Network' and df['Customer Service- Sub Category (Case)']=='Connectivity'):
            return 'Network_Connectivity'
        elif (df['Customer_Service_Category']=='Network' and df['Customer Service- Sub Category (Case)']=='Call Drop'):
            return 'Call-drop'
        elif (df['Customer_Service_Category']=='Network'):
            return 'Other_network_related'
        
        elif (df['Customer_Service_Category']=='Others' and df['Customer Service Category (Case)']=='Blogs/Web'):
            return 'Blogs_web'
        elif (df['Customer_Service_Category']=='Others' and df['Customer Service Category (Case)']=='Press Release'):
            return 'Press_release'
        elif (df['Customer_Service_Category']=='Others' and df['Customer Service Category (Case)']=='DND'):
            return 'DND'
        elif (df['Customer_Service_Category']=='Others'):
            return 'Other_news_campaign'
        
        elif (df['Customer_Service_Category']=='Migration' and df['Customer Service- Sub Category (Case)']=='MNP Threat'):
            return 'MNP_Threat'
        elif (df['Customer_Service_Category']=='Migration' and df['Customer Service- Sub Category (Case)']=='Activation/Deactivation'):
            return 'Activation_Deactivation'
        elif (df['Customer_Service_Category']=='Migration' and df['Customer Service- Sub Category (Case)']=='Port In'):
            return 'Port_in'
        elif (df['Customer_Service_Category']=='Migration'):
            return 'Overall_services'

        elif (df['Customer_Service_Category']=='Balance_Payment' and df['Customer Service- Sub Category (Case)']=='Balance deduction'):
            return 'Balance_deduction'
        elif (df['Customer_Service_Category']=='Balance_Payment' and df['Customer Service- Sub Category (Case)']=='Billing dispute'):
            return 'Billing_dispute'
        elif (df['Customer_Service_Category']=='Balance_Payment'):
            return 'Recharge'
        
        elif (df['Customer_Service_Category']=='Data' and df['Customer Service- Sub Category (Case)']=='Speed'):
            return 'Speed'
        elif (df['Customer_Service_Category']=='Data' and df['Customer Service- Sub Category (Case)']=='Connectivity'):
            return 'Internet_connectivity'
        elif (df['Customer_Service_Category']=='Data' and df['Customer Service- Sub Category (Case)']=='Internet Services'):
            return 'Internet_services'

        elif (df['Customer_Service_Category']=='Services' and df['Customer Service- Sub Category (Case)']=='International Roaming'):
            return 'International_Roaming'
        elif (df['Customer_Service_Category']=='Services' and df['Customer Service- Sub Category (Case)']=='Generic Feedback'):
            return 'customer_feedback'
        elif (df['Customer_Service_Category']=='Services' and df['Customer Service- Sub Category (Case)']=='Customer Care Feedback'):
            return 'customer_feedback'
        else:
            return 'VAS'
    def tf_idf(self,dataframe):
        # function to create object of TFidfVectorizer and returns feature_matrix, labels and dictionary of feature name and their weight.
        stop_words = nltk.corpus.stopwords.words('english')

        my_stop_list=['yes','okay','ok','month','year','try','receive','kindly','subject','message',
                      'image','thank', 'regard','copy','virus','attachment','www','http www','vodafoneidea com',
                      'time','care','related']
        for i in my_stop_list:
            if i not in stop_words:
                stop_words.append(i)
        stop_words =set(stop_words)
        tfidf = TfidfVectorizer(sublinear_tf = True ,min_df=0.001, stop_words=stop_words, ngram_range=(1, 2))
        #tfidf = TfidfVectorizer(sublinear_tf = True ,min_df=0.001, stop_words='english', ngram_range=(1, 2))
        
        feature_matrix = tfidf.fit_transform(dataframe.final_text).toarray()

        labels = dataframe.Target

        print(feature_matrix.shape)
        feature_names = tfidf.get_feature_names()

        # writing feature names into text file
        text_file = open("feature_names.txt","w")
        for name in feature_names:
            text_file.write("{} \n".format(name.encode("utf-8")))


        # to get the feature_matrix weight
        idf = tfidf.idf_
        feature_weight = dict(zip(tfidf.get_feature_names(), idf))

        return tfidf, feature_matrix, labels, feature_weight

    def print_unigrams_bigrams(self, target_id, feature_matrix, labels):
        '''prints top unigrams and bigrams'''

        text_file = open("Unigrams_bigrams.txt","w")
        N = 5
        for Customer_Service_Category, target in sorted(target_id.items()):
            feature_matrix_chi2 = chi2(feature_matrix, labels == target)
            indices = np.argsort(feature_matrix_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
            text_file.write("# '{}':".format(Customer_Service_Category))
            text_file.write("\n  . Unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
            text_file.write("\n  . Bigrams:\n. {}\n\n".format('\n. '.join(bigrams[-N:])))


    def split_train_test(self, feature_matrix, labels, dataframe):
        '''split the dataset into training and test test
        
        Arguments:
            feature_matrix {[matrix]} -- generated by tfidf
            labels {[list]} -- list of true labels
            dataframe {[dataframe]} 
        
        Returns:
            [y_train, y_pred_train, y_test, y_pred_test, indices_train, indices_test, model]
        '''
        # model = RandomForestClassifier(max_feature_matrix='auto', n_estimators = 100, min_samples_leaf = 5, n_jobs = -1, criterion = 'gini')
        model = SVC(kernel = "rbf", verbose = False, gamma = 1, C = 1)
        # model = RidgeClassifier(alpha = 1.0, normalize = False, solver = 'auto')
        X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(feature_matrix, labels, dataframe.index, test_size=0.3, random_state=1)
        model.fit(X_train, y_train)
        #save model
        file='model.pkl'
        pickle.dump(model,open(file,'wb'))
        

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        return X_train, X_test, y_train, y_pred_train, y_test, y_pred_test, indices_train, indices_test, model


    def wrong_prediction(self, target_df, id_to_category, y_test, y_pred, indices_train, indices_test, conf_mat):
        '''prints which rows(with customer tweet) are predicted as other classes
        
        [description]
        
        Arguments:
            target_df {[dataframe]} -- contains two columns one for unique Customer_Service_Category and another for its label
            id_to_category {[dict]} -- keys are the labels and values are their true classes like (1 - Network and 2 - Data)
            y_test {[Series]} -- contains actual labels for test data
            y_pred {[Series]} -- contains predicted labels for test data
            indices_train {[Series]} -- contains index of each training dataset
            indices_test {[Series]} -- contains index for each test dataset
            conf_mat {[matrix]} -- confusion matrix for test dataset
        '''

        for predicted in target_df.Target:
            for actual in target_df.Target:
                if predicted != actual and conf_mat[actual, predicted] >= 10:
                    print("'{}' predicted as '{}' : {} examples.".format(id_to_category[actual], id_to_category[predicted], conf_mat[actual, predicted]))
                    display(df.loc[indices_test[(y_test == actual) & (y_pred == predicted)]][['Customer_Service_Category', 'Inbound Message']])
                    print(' ')


    def lemmatize(self, text_column, logging=True):
        """Lemmatizes the given texts.
       df:      data frame.
       text_column:    type: Series, column containing text
       logging: True/False. To keep track of how many documents(text feedback) have been processed."""
        # print(text_column)
        texts = []
        counter = 1
        for doc in text_column:
            if counter % 1000 == 0 and logging:
                print("Processed %d out of %d documents." % (counter, len(text_column)))
            counter += 1
            doc = twitter_text_analytics.nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = ' '.join(tokens)
            texts.append(tokens)
        # print(texts)
        return pd.Series(texts)

    def FP_FN(self, conf_matrix):
        '''calculates False_Positive and False_Negative
        
        Arguments:
            conf_matrix {[matrix]}
        
        Returns:
            lists - False_Positive and False_Negative
        '''
        FP = []
        FN = []
        for i in range(0,conf_matrix.shape[0]):
            count1 = 0
            count2 = 0
            for j in range(0,conf_matrix.shape[0]):
                if(i != j):
                    count1 = count1 + conf_matrix[i][j]
                    count2 = count2 + conf_matrix[j][i]
            FP.append(count1)
            FN.append(count2)
        return FP, FN

    def TN(self, conf_matrix):
        '''Return list containing True_Negative for each class
        
        Arguments:
            conf_matrix {[matrix]}
        
        Returns:
            [list] -- True_Negative list for each class
        '''
        TN = []
        for i in range(0,conf_matrix.shape[0]):
            count = 0
            for j in range(0,conf_matrix.shape[0]):
                for k in range(0,conf_matrix.shape[0]):
                    if(i != j and i != k):
                        count = count + conf_matrix[j][k]
            TN.append(count)
        return TN


    def getConfusionMatrixValues(self,y_true,y_pred):
        ''' Generates Confusion matrix and return TP,FP,FN,TN
        
        Arguments:
            y_true {[list]} -- [true labels]
            y_pred {[list]} -- [predicted labels]

        Returns:
            [lists] -- TP,FP,FN,TN

        '''
        cm = pd.DataFrame(confusion_matrix(y_true,y_pred))
        print("Confusion Matrix\n",cm,"\n")
        TP = []

        for i in range(0,cm.shape[0]):
            TP.append(cm[i][i])

        FP, FN = self.FP_FN(cm)
        TN = self.TN(cm)
        return (cm,TP,FP,FN,TN)


    def getClassificationReport(self, y_true, y_pred, df):
        '''returns classification report as a dataframe
        
        Arguments:
            y_true {[Series]} -- [true lables]
            y_pred {[Series]} -- [predicted labels]
            df {[Dataframe]} -- [dataframe after all the text preprocessing]
        
        Returns:
            [Dataframe] -- [classification report]
        ''' 
        
        return pd.DataFrame(metrics.classification_report(y_true, y_pred,target_names=df['Customer_Service_sub_Category'].unique(), output_dict=True))

    def detractor_overlap(self,df,path):
        '''Detractor Overlap analysis'''
        detractors_prepaid=glob.glob(path+ '\\' +'TNPS*'+'PREPAID' + '*VODA*'+ '*.csv')
        print('Detractor PREPAID FILES: ')
        print(detractors_prepaid)
        print()
        detractors_postpaid=glob.glob(path+ '\\' +'TNPS*'+'POSTPAID' + '*VODA*'+ '*.csv')
        print('Detractor POSTPAID FILES: ')
        print(detractors_postpaid)
        print()
            
        #combining files\
        big_list=[]
        print('Appending detractor Files in big_list....')
        for f in detractors_prepaid:
            det = pd.read_csv(f)
            det['LOB']='PREPAID'
            det['circle']=(f.split('_')[6]).split('.')[0]
            det.rename(columns={'MSISDNS':'MSISDN'},inplace=True)
            big_list.append(det)
        for f in detractors_postpaid:
            det = pd.read_csv(f)
            det['LOB']='POSTPAID'
            det['circle']=(f.split('_')[6]).split('.')[0]
            det.rename(columns={'MSISDNS':'MSISDN'},inplace=True)
            big_list.append(det)
        
        print('Combining into single dataframe.....')
        combined_detractor_base = pd.concat(big_list, axis=0, ignore_index =True)
        combined_detractor_base.rename(columns={'MSISDN':'CUSTOMER_MOBILE_NO'},inplace=True)
        print('Length of combined detractor base:',len(combined_detractor_base))
        combined_detractor_base.drop_duplicates(subset='CUSTOMER_MOBILE_NO',keep='first',inplace=True)
        print('Length of combined detractor base after dropping duplicates:',len(combined_detractor_base))
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
        print('unique MSISDNs in SM data:',len(df))
        #combined_detractor_base['CUSTOMER_MOBILE_NO']=combined_detractor_base['CUSTOMER_MOBILE_NO'].astype(object)
        overlapping_base=combined_detractor_base.merge(df,how='inner',on=['CUSTOMER_MOBILE_NO'])
        print('Length of overlapping detractor base:',len(overlapping_base))
        print('Saving File at location: output/')
        overlapping_base.to_csv('output/overlapping_Detractor_base.csv',index=False)
        print('Done!!')
        
        return combined_detractor_base,overlapping_base

    def fix_MSISDNs(self,df):
        df['Mobile number (Case)']=df['Mobile number (Case)'].astype(str).map(lambda x: ''.join([i for i in x if i.isdigit()]))
        df['len']=df['Mobile number (Case)'].astype(str).apply(lambda x : len(x))
        df['Mobile number (Case)']=df['Mobile number (Case)'].apply(lambda x:x[0:10] if len(x)>15 else x)
        df['Mobile number (Case)']=pd.to_numeric(df['Mobile number (Case)'], errors='coerce')
        return df
        
    def detractor_overlap_kpis(self,df,path):
        '''Detractor Overlap analysis'''
        detractors=glob.glob(path+ '\\' +'TNPS*'+ '*.csv')
        print('Detractor KPIs FILES: ')
        print(detractors)
        print()
            
        #combining files\
        print('Appending detractor Files in big_list....')
        for f in detractors:
            det = pd.read_csv(f)
            det.rename(columns={'MSISDNS':'MSISDN'},inplace=True)
            circle=det.loc[0,'CIRCLE_NAME']
            det.rename(columns={'MSISDN':'CUSTOMER_MOBILE_NO'},inplace=True)
            print('Length of combined detractor base:',len(det))
            det.drop_duplicates(subset='CUSTOMER_MOBILE_NO',keep='first',inplace=True)
            det['CUSTOMER_MOBILE_NO']=det['CUSTOMER_MOBILE_NO'].astype(str)
            print('Length of combined detractor base after dropping duplicates:',len(det))
            df.drop_duplicates(inplace=True)
            df.dropna(inplace=True)
            print('unique MSISDNs in SM data:',len(df))
            
            print("Circle:",circle)
            overlapping_base=det.merge(df,how='left',on=['CUSTOMER_MOBILE_NO'])
            print('Size of overlapping base in SM data:',len(overlapping_base))
            
            print('Saving File at location: output/')
            overlapping_base.to_csv('output/kpis_full_'+circle+'.csv',index=False)
            print('Done!!')
#            if f==detractors[0]:
#                data_f=overlapping_base
#            else:
#                data_f=pd.concat([data_f,overlapping_base])
#        return data_f
#        
    



if __name__ == '__main__':

    twitter_object = twitter_text_analytics()
    preprocess = Preprocessing_twitter()
    corpus, stop_words = preprocess.setting_stopwords_corpus()

    # choosing only twitter data
    #df = df[df['Social Network'] == "Twitter"]
    
    # converting different categories into main categories
    df['Customer_Service_Category'] = df['Customer Service Category (Case)'].apply(lambda x : twitter_object.reducing_customer_category(x))

    
    
    
    
    
    #removing stop words and punctuations
    df['Inbound_Message'] = preprocess.standardization(text_column= df['Inbound Message'], stopwords= stop_words, english_corpus= corpus)
    df['Inbound_Message'].fillna('Null',inplace= True)
    df = df.drop(df[df['Inbound_Message']=='Null'].index, axis=0).reset_index().drop(['index'],axis=1)
    print('Text_column Nulls:',df['Inbound_Message'].isnull().sum())
    print('Length of df: ', len(df))

    #Tokenizations
    df['tokens'] = preprocess.tokenization(text_column = df['Inbound_Message'])
    print('Length of df: ', len(df))

    #lemmatization
    df['final_text'] = twitter_object.lemmatize(text_column= df['Inbound_Message'])
    # df.to_csv('twitter.csv', index= False, header= True)

    # # Creating flag column for subsetting English Feedbacks
    df['flag'] = df['tokens'].apply(lambda line: preprocess.set_flag_if_english_text(text_row= line,total_df_rows = len(df), english_corpus = corpus))
    # selecting only english text
    df = df[df['flag'] == 1]
    df.to_csv('categories_sm_v1.csv',index=False)
    
    # sentiment Analyser
    sentiment_obj = SentimentAnalysis()
    df['senti_score'], df['pred_sentiment'] = sentiment_obj.sentiment_assignment(df.Inbound_Message)
    
    
    #---add MSISDNs from alternate number
    df_alt=df.loc[:, df.columns != 'Mobile number (Case)']
    df_alt.rename(columns={'Alternate Number (Case)':'Mobile number (Case)'},inplace=True)
    df_concat=pd.concat([df,df_alt])
    #fix MSISDNs
    df=twitter_object.fix_MSISDNs(df_concat)
    df.to_csv('output/Voda_SM_sentiments.csv', index= False, header= True)
    
    
    df['Customer_Service_Category']=np.where(df['Customer Service- Sub Category (Case)'].isin(['Unable to recharge','Unable to pay','Recharge Unsuccessful','Offers/Promotions']),'Recharge',df['Customer_Service_Category'])
    df=df[df['Customer_Service_Category']!='Generic']
    df=df[~((df['Customer_Service_Category']=='Others') & (df['Customer Service- Sub Category (Case)']=='Generic Mention'))]
    df=df[~((df['Customer_Service_Category']=='Others') & (df['Customer Service- Sub Category (Case)']=='NRI Customer'))]
    df['Customer_Service_Category']=np.where(df['Customer_Service_Category']=='Recharge','Balance_Payment',df['Customer_Service_Category'])
    df['Customer_Service_Category']=np.where(df['Customer_Service_Category']=='Activation_Deactivation','Migration',df['Customer_Service_Category'])
    
    df=df[~(df['Customer_Service_Category'].isnull())]
    df_services=df[df['Customer_Service_Category']=='Services']
    
    # plotting class distribution
    fig = plt.figure(figsize=(10,10))
    sns.countplot(x = "Customer_Service_Category", data = df)
    plt.savefig("Class_distribution.jpg")

    # converting categories into nested sub-categories
    df['Customer_Service_sub_Category'] =df.apply(twitter_object.nested_categories, axis=1)
    
    dict_level={}
    dict_level['Coverage']='Network'
    dict_level['Network_Connectivity']='Network'
    dict_level['Call-drop']='Network'
    dict_level['Other_network_related']='Network'
    
    dict_level['Blogs_web']='Others'
    dict_level['Press_release']='Others'
    dict_level['DND']='Others'
    dict_level['Other_news_campaign']='Others'
    
    dict_level['MNP_Threat']='Migration'
    dict_level['Activation_Deactivation']='Migration'
    dict_level['Port_in']='Migration'
    dict_level['Overall_services']='Migration'
    
    dict_level['Balance_deduction']='Balance_Payment'
    dict_level['Billing_dispute']='Balance_Payment'
    dict_level['Recharge']='Balance_Payment'
    
    dict_level['Speed']='Data'
    dict_level['Internet_connectivity']='Data'
    dict_level['Internet_services']='Data'
    
    dict_level['International_Roaming']='Services'
    dict_level['customer_feedback']='Services'
    dict_level['VAS']='Services'
    
    # plotting class distribution
    fig = plt.figure(figsize=(10,10))
    plt.xticks(rotation=90)
    sns.countplot(x = "Customer_Service_sub_Category", data = df)
    plt.savefig("Class_distribution_sub_category.jpg")

    # providing numerical value to categories
    df['Target'] = df['Customer_Service_sub_Category'].factorize()[0]
    
    df_save=df.copy()
#    unique_MSISDNs=df[['Mobile number (Case)']]
#    unique_MSISDNs.drop_duplicates(inplace=True)
#    unique_MSISDNs.dropna(inplace=True)
#    unique_MSISDNs.to_csv('output/unique_MSISDNs.csv',index=False)
#    #------------------------------------------------
    #--------------------------------------------------
    
    # Dictionaries for future use to calculate wrong predictions
    df=df[~(df['Customer_Service_sub_Category'].isnull())]
    target_df = df[['Customer_Service_sub_Category', 'Target']].drop_duplicates()
    # Dictionaries for future use to calculate wrong predictions
    target_id = dict(target_df.values)
    id_to_category = dict(target_df[['Target', 'Customer_Service_sub_Category']].values)
    target_id= {'Other_network_related': 0,
     'Coverage': 1,
     'Press_release': 2,
     'MNP_Threat': 3,
     'Recharge': 4,
     'Port_in': 5,
     'Speed': 6,
     'Call-drop': 7,
     'Internet_connectivity': 8,
     'Overall_services': 9,
     'Network_Connectivity': 10,
     'Activation_Deactivation': 11,
     'DND': 12,
     'VAS': 13,
     'Balance_deduction': 14,
     'Other_news_campaign': 15,
     'Internet_services': 16,
     'Billing_dispute': 17,
     'customer_feedback': 18,
     'Blogs_web': 19,
     'International_Roaming': 20}
    id_to_category= {0: 'Other_network_related',
         1: 'Coverage',
         2: 'Press_release',
         3: 'MNP_Threat',
         4: 'Recharge',
         5: 'Port_in',
         6: 'Speed',
         7: 'Call-drop',
         8: 'Internet_connectivity',
         9: 'Overall_services',
         10: 'Network_Connectivity',
         11: 'Activation_Deactivation',
         12: 'DND',
         13: 'VAS',
         14: 'Balance_deduction',
         15: 'Other_news_campaign',
         16: 'Internet_services',
         17: 'Billing_dispute',
         18: 'customer_feedback',
         19: 'Blogs_web',
         20: 'International_Roaming'}
            
    # getting feature_matrix and labels from tf_idf
    tfidf, feature_matrix, labels, feature_matrix_weight = twitter_object.tf_idf(df)
    feature_df=pd.DataFrame(feature_matrix,columns=tfidf.get_feature_names())
    list_features=pd.DataFrame(feature_df.sum(axis=0,skipna=True),columns=['sum']).reset_index()
    list_features.sort_values(by='sum',ascending=False,inplace=True)
    list_features.to_csv('top_words_in_SM.csv',index=False)
    
    file='tf_idf.pkl'
    pickle.dump(tfidf,open(file,'wb'))
        
    # print popular unigrams and bigrams
    twitter_object.print_unigrams_bigrams(target_id, feature_matrix, labels)

    # split into test and train
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(feature_matrix, labels, df.index, test_size=0.3, random_state=1)
    
    X_train, X_test, y_train, y_pred_train, y_test, y_pred_test, indices_train, indices_test, model = twitter_object.split_train_test(feature_matrix, labels, df)
    print("Classification Report on train set \n",metrics.classification_report(y_train, y_pred_train, target_names=df['Customer_Service_sub_Category'].unique()))

        

    print("Classification Report on test set \n",metrics.classification_report(y_test, y_pred_test, target_names=df['Customer_Service_sub_Category'].unique()))

    train_df = df.reindex(indices_train)
    train_df['target']=y_train
    #pred_train=pd.read_csv('train_df.csv')
    #pred_train=pred_train[['pred']]
    y_pred_train=train_df['predicted_class']
    train_df['predicted_class']=y_pred_train.tolist()
    train_df['predicted_cat']=train_df['predicted_class'].replace(id_to_category)
    train_df['predicted_cat_level1']=train_df['predicted_cat'].replace(dict_level)
    
    clf=pickle.load(open('model.pkl','rb'))
    y_pred_test = clf.predict(X_test)
        
    test_df = df.reindex(indices_test)
    test_df['target']=y_test
    test_df['predicted_class']=y_pred_test.tolist()
    test_df['predicted_cat']=test_df['predicted_class'].replace(id_to_category)
    test_df['predicted_cat_level1']=test_df['predicted_cat'].replace(dict_level)
    
    
    train_df.to_csv('train_df_new.csv',index=False)
    test_df.to_csv('test_df_new.csv',index=False)
    
    # getting classification report for level2
    order=df['Customer_Service_sub_Category'].unique()
    order.sort()
    train = pd.DataFrame(metrics.classification_report(train_df['Customer_Service_sub_Category'], train_df['predicted_cat'],target_names=order, output_dict=True))
    conf_mat_train,TP,FP,FN,TN = twitter_object.getConfusionMatrixValues(train_df['Customer_Service_sub_Category'],train_df['predicted_cat'])
    train_table = {'True_Positive' : TP, 'False_Positive' : FP, 'False_Negative' : FN, 'True_Negative' : TN}
    train_cm = pd.DataFrame(train_table, index = order)
    train_cm['Dataset'] = 'Train'
    train['Dataset'] = 'Train'


    test = pd.DataFrame(metrics.classification_report(test_df['Customer_Service_sub_Category'], test_df['predicted_cat'],target_names=order, output_dict=True))
    conf_mat_test,TP,FP,FN,TN = twitter_object.getConfusionMatrixValues(y_test,y_pred_test)
    test_table = {'True_Positive' : TP, 'False_Positive' : FP, 'False_Negative' : FN, 'True_Negative' : TN}
    test_cm = pd.DataFrame(test_table, index = order)
    test_cm['Dataset'] = 'Test'
    test['Dataset']= 'Test'

    # concating dataframes
    classification_report = pd.concat([train,test])
    df_confusion_matrix = pd.concat([train_cm,test_cm])

    # writing to the excel sheets
    with pd.ExcelWriter('output/Twitter_Classification_Report_level2_new.xlsx') as writer:
        classification_report.to_excel(writer, sheet_name = 'Classification_Report')
        df_confusion_matrix.to_excel(writer, sheet_name = 'Confusion_Matrix')
    
    
    
    # getting classification report for level1
    order_level1=df['Customer_Service_Category'].unique()
    order_level1.sort()
    train = pd.DataFrame(metrics.classification_report(train_df['Customer_Service_Category'], train_df['predicted_cat_level1'],target_names=order_level1, output_dict=True))
    conf_mat_train,TP,FP,FN,TN = twitter_object.getConfusionMatrixValues(train_df['Customer_Service_Category'],train_df['predicted_cat_level1'])
    train_table = {'True_Positive' : TP, 'False_Positive' : FP, 'False_Negative' : FN, 'True_Negative' : TN}
    train_cm = pd.DataFrame(train_table, index = order_level1)
    train_cm['Dataset'] = 'Train'
    train['Dataset'] = 'Train'


    test = pd.DataFrame(metrics.classification_report(test_df['Customer_Service_Category'], test_df['predicted_cat_level1'],target_names=order_level1, output_dict=True))
    conf_mat_test,TP,FP,FN,TN = twitter_object.getConfusionMatrixValues(test_df['Customer_Service_Category'], test_df['predicted_cat_level1'])
    test_table = {'True_Positive' : TP, 'False_Positive' : FP, 'False_Negative' : FN, 'True_Negative' : TN}
    test_cm = pd.DataFrame(test_table, index = order_level1)
    test_cm['Dataset'] = 'Test'
    test['Dataset']= 'Test'

    # concating dataframes
    classification_report = pd.concat([train,test])
    df_confusion_matrix = pd.concat([train_cm,test_cm])

    # writing to the excel sheets
    with pd.ExcelWriter('output/Twitter_Classification_Report_level1_new.xlsx') as writer:
        classification_report.to_excel(writer, sheet_name = 'Classification_Report')
        df_confusion_matrix.to_excel(writer, sheet_name = 'Confusion_Matrix')

    # printing wrong predictions
    twitter_object.wrong_prediction(target_df, id_to_category, y_test, y_pred_test, indices_train, indices_test, confusion_matrix(y_test,y_pred_test))

    # running model on unseen data
    # test_data = pd.read_excel("testing.xlsx")
    # fm = tfidf.transform(test_data.Customer).toarray()
    # test_data['Predicted'] = (model.predict(fm))
    # test_data.to_excel("testing.xlsx", index= False, header= True )


    
    #df2[df2.len>10] check
    df=pd.read_csv('output/Voda_SM_sentiments.csv')
    df['Mobile number (Case)']=df['Mobile number (Case)'].astype(str).apply(lambda x: x.split('.')[0])
    df1=pd.read_csv('train_df_new.csv')
    df2=pd.read_csv('test_df_new.csv')
    
    df_cat=pd.concat([df1,df2])
    df_cat=df_cat[['Case Id','Mobile number (Case)','predicted_cat','predicted_cat_level1']]
    

    df_category_sentiment=df.merge(df_cat,on=['Mobile number (Case)','Case Id'],how='inner')
    df_category_sentiment.rename(columns={'predicted_cat':'predicted_cat_level2'},inplace=True)
    
    df_category_sentiment.rename(columns={'Mobile number (Case)':'CUSTOMER_MOBILE_NO'},inplace=True)
    df_category_sentiment=df_category_sentiment[['CUSTOMER_MOBILE_NO','senti_score','pred_sentiment','predicted_cat_level2','predicted_cat_level1']]
    df_category_sentiment.to_csv('output/SM_category_sentiment.csv',index=False)
    detractor_kpis=twitter_object.detractor_overlap_kpis(df_category_sentiment,path=r"..\..\TextMining_pipeline\TNPS_NUMBERS_KPIs")
    
    
    
    
    
    # detractor overlap analysis
    df.rename(columns={'Mobile number (Case)':'CUSTOMER_MOBILE_NO'},inplace=True)
    detractor_base, overlapping_base=twitter_object.detractor_overlap(df[['CUSTOMER_MOBILE_NO']],path=r"..\..\TextMining_pipeline\Detractor_base")
    #circle wise analysis
    overlap_base_circle_wise=pd.DataFrame(overlapping_base.groupby(['circle','LOB']).size(),columns=['Text_analytics_overlap_base'])
    detractor_base_circle_wise=pd.DataFrame(detractor_base.groupby(['circle','LOB']).size(),columns=['Detractor_base'])
    overlap_circle_wise=detractor_base_circle_wise.merge(overlap_base_circle_wise,left_index=True,right_index=True)
    overlap_circle_wise.to_csv('output/overlap_base_circle_wise.csv',index=True)
#---------------------------------------------------------------------------------------------------------------------   
#--------------------------------------sentiment analysis of overlapping base
    senti_sm=overlapping_base.merge(df,how='left',on=['CUSTOMER_MOBILE_NO'])
    senti_sm.to_csv('output/Voda_SM_sentiments_overlap_base.csv', index= False, header= True)
    senti_sm=pd.read_csv('output/Voda_SM_sentiments_overlap_base.csv')
    
    #sentiment_metrics
    small_to_caps={'neutral':'Neutral','negative':'Negative','positive':'Positive'}
    senti_sm['pred_sentiment']=senti_sm['pred_sentiment'].replace(small_to_caps)
    order=senti_sm['Sentiment'].unique()
    order.sort()
    train = pd.DataFrame(metrics.classification_report(senti_sm['Sentiment'], senti_sm['pred_sentiment'], target_names=order,output_dict=True))
    conf_mat_train,TP,FP,FN,TN = twitter_object.getConfusionMatrixValues(senti_sm['Sentiment'],senti_sm['pred_sentiment'])
    train_table = {'True_Positive' : TP, 'False_Positive' : FP, 'False_Negative' : FN, 'True_Negative' : TN}
    train_cm = pd.DataFrame(train_table, index = order)
        
    with pd.ExcelWriter('output/Twitter_Classification_Report_sentiments.xlsx') as writer:
            train.to_excel(writer, sheet_name = 'Classification_Report')
            train_cm.to_excel(writer, sheet_name = 'Confusion_Matrix')

#---------------------------------------------------------------------------------------------------------------------
#------------------------------------wordcloud and unigram bigrams
    senti_sm.rename(columns={'MON':'MONTH','Final_text':'final_text'},inplace=True)
    viz = Visualizations_twitter()
    viz.create_wordcloud(df= senti_sm, stopwords= stop_words, file_type_str = 'SM')
    
    #unigrams bigrams
    final_processing_obj=finalprocessing()
    #changed min_df
    final_processing_obj.create_n_grams(df= senti_sm, file_type_str= 'SM', stopwords =stop_words, bonus_score=100, min_df=0.01)
