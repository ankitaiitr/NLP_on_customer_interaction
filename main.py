# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:49 2020

@author: Ankita
"""

import gensim
# import en_core_web_sm
# import spacy
# from nltk.corpus import stopwords
# from nltk.corpus import wordnet
import nltk.corpus
import numpy as np
import re 
import pandas as pd
import string
import time
import os
import glob

from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
pd.set_option('Display.max_columns',None)

# import dask.dataframe as dd



class main:

    # calling constructor
    def __init__(self, selected_months):
        # creating objects of the classes
        # self.LoadData_object = LoadData()
        # self.TNPS_Dict2DF_object = TNPS_Dict2DF()
        # self.Mine_object = Mine()
        self.dict_all = dict()

        self.TNPS_channels = {'contact_centre': "contact_centre_JAN'19", 
                        'digital':        "General_report_digital_JAN'19", 
                        'retail':          "General_report_retail_JAN'19", 
                        'network':        "General_report_network_JAN'19" }  

        self.count = 0 #for keeping Track of no of documents processed in def eng_flag() function

        self.months = {'JFM' : ["JAN","FEB","MAR"],
                        'AMJ' : ["APR","MAY","JUN"],
                        'JAS' : ['JUL','AUG','SEP'],
                        'OND' : ['OCT','NOV','DEC']
        }

        self.months_NPS = self.months[selected_months]
        self.months_TNPS = self.months[selected_months]


        #----------TNPS COLUMNS ---------------------------
        self.TNPS_cols_to_inc = ['CUSTOMER_MOBILE_NO', 'CUSTOMER_LANGUAGE', 'BRAND', 'CIRCLE',  'LOB','LTR', 'SENTIMENT', 
                            'CUSTOMER_CATEGORY','MONTH', 'GLOBAL CHANNEL']

        self.TNPS_text_col = ['CUSTOMER RESPONSE', 'CUSTOMER_COMMENT', 'RCA', 'RCA SUB CATEGORY LEVEL3', 'SECONDARY RCA', 
                         'SECONDARY RCA CATEGORY','SECONDARY RCA SUB CATEGORY', 'SECONDARY RCA LEVEL3 SUB CATEGORY', 
                         'CALLBACK CUSTOMER TEXT', 'COMMENT 1', 'COMMENT 2', 'COMMENT 3', 'REASON_FOR_CALL', 
                         'REASON_FOR_CALL_SUBTYPE', 'REASON_FOR_CALL_DFF1' ]

        self.TNPS_cat_col =  ['CUSTOMER_CATEGORY', 'NEW NORMAL CATEGORY']
        self.TNPS_cols_to_inc.extend(self.TNPS_text_col)



        #----------- NPS COLUMNS -------------------------
        self.NPS_cols_to_inc = ['Circle', 'Main Service Provider', 'Type of Connection', 'VOC_TYPE',
                           'NPS Segments', 'Consumer Segments', 'MI User / Non User', 'MONTH',
                           'Internet Type', 'VOC', 'Areas of Improvement']

        self.NPS_text_col = ['VOC', 'Areas of Improvement']

        self.NPS_categories = ['RURAL', 'URBAN']


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
            det['circle']=(f.split('_')[5]).split('.')[0]
            det.rename(columns={'MSISDNS':'MSISDN'},inplace=True)
            big_list.append(det)
        for f in detractors_postpaid:
            det = pd.read_csv(f)
            det['LOB']='POSTPAID'
            det['circle']=(f.split('_')[5]).split('.')[0]
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
        print('unique MSISDNs in TNPS data:',len(df))
        overlapping_base=combined_detractor_base.merge(df,how='inner',on=['CUSTOMER_MOBILE_NO'])
        print('Length of overlapping detractor base:',len(overlapping_base))
        print('Saving File at location: ../Output/')
        overlapping_base.to_csv('../Output/overlapping_Detractor_base_dropDup.csv',index=False)
        print('Done!!')
        
        return combined_detractor_base,overlapping_base
    
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
            
            print('unique MSISDNs in SM data:',len(df))
            
            print("Circle:",circle)
            overlapping_base=det.merge(df,how='left',on=['CUSTOMER_MOBILE_NO'])
            print('Size of overlapping base in SM data:',len(overlapping_base))
            MYDIR='../Output/customer_profiling/circle_wise_kpis'
            CHECK_FOLDER = os.path.isdir(MYDIR)
    
            # If folder doesn't exist, then create it.
            if not CHECK_FOLDER:
                os.makedirs(MYDIR)
                print("created folder : ", MYDIR)
            
            print('Saving File at location: ../Output/')
            overlapping_base.to_csv('../Output/customer_profiling/circle_wise_kpis/kpis_full_'+circle+'.csv',index=False)
            print('Done!!')
#            if f==detractors[0]:
#                data_f=overlapping_base
#            else:
#                data_f=pd.concat([data_f,overlapping_base])
#        return data_f
#        
    


if __name__ == '__main__':
    # importing other classes 
    from load_data import LoadData
    from tnps_dict2df import TNPS_Dict2DF
    from mine import Mine
    from TextPreProcessing import Preprocessing
    from Visualizations import Visualizations
    from finalprocessing import finalprocessing
    from SentimentAnalysis import SentimentAnalysis

    main_object = main("JFM")
    Mine_object = Mine("JFM")
    df, file_type_str = Mine_object.MineVOCs("TNPS",'..\TEXT_P_DATA\TNPS\dict_to_df.pkl')


    
    preprocess = Preprocessing('JFM')
    #Corpus and stopwords
    corpus, stop_words = preprocess.setting_stopwords_corpus()

    # file_type_str = 'TNPS'
    # df = pd.read_csv(r'../TEXT_P_DATA/TNPS/Merged_TNPS_data.csv'))
    #Standardization
    df['text_col_std'] = preprocess.standardization(text_column= df['text_col_std'], stopwords= stop_words, english_corpus= corpus)
    df['text_col_std'].fillna('Null',inplace= True)
    df = df.drop(df[df['text_col_std']=='Null'].index, axis=0).reset_index().drop(['index'],axis=1)
    print('Text_column Nulls:',df['text_col_std'].isnull().sum())
    # print('Length of df: ', len(df))
    # #Tokenization
    df['tokens'] = preprocess.tokenization(text_column = df['text_col_std'])
    print('Length of df: ', len(df))
    

    # Creating flag column for subsetting English Feedbacks
    df['flag'] = df['tokens'].apply(lambda line: preprocess.set_flag_if_english_text(text_row= line,total_df_rows = len(df), english_corpus = corpus))
    # writing in Excel
    df = preprocess.English_text_to_excel_or_csv(df_text= df, file_type_str= file_type_str)  
  
      #Creating Wordcloud
    viz = Visualizations('JFM')
    viz.create_wordcloud(df= df, stopwords= stop_words, file_type_str = file_type_str)
    # Converting wordcloud images to pdf
    viz.img_to_pdf(images_folder_path= os.path.join(r'../Output/Word_Cloud', file_type_str), pdf_name= 'WordClouds')

    #Final processing 
    final_processing_obj = finalprocessing('JFM')
    
    # Lemmatization
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace =True)
    df['final_text'] = final_processing_obj.lemmatize(text_column= df['text_col_std'])
    # print(df['final_text'])

    #Assigning Sentiments
    SentimentAnalysis_obj = SentimentAnalysis()
    df['senti_score'], df['pred_sentiment'] = SentimentAnalysis_obj.sentiment_assignment(text_column_for_senti = df['text_col_std'])
    #Saving dataframe with sentiments.
    if file_type_str == 'NPS':
    	df.to_excel(excel_writer=os.path.join('../Output/Sentiment_Analysis/', file_type_str,'Feedbacks_with_sentiments.xlsx'), header=True,index=False)
    elif file_type_str == 'TNPS':
    	df.to_csv( os.path.join('../Output/Sentiment_Analysis/', file_type_str,'Feedbacks_with_sentiments.csv'), header=True,index=False)
    	 #Metrics for sentiment assignment comparison for TNPS
    	# SentimentAnalysis_obj.sentiment_comparison(default_senti_column_name='SENTIMENT', df=df, pred_senti_column_name = 'senti_score',file_type_str= file_type_str, months_start_end_initials= 'JFM')
    print('File is saved at location: ', os.path.join('../Output/Sentiment_Analysis/', file_type_str) )

    # print(df)

    #N-grams creation
    final_processing_obj.create_n_grams(df= df, file_type_str= file_type_str, stopwords =stop_words, bonus_score=100, min_df=0.01)
    
    
    #-------------------------input data with category and sentiments
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

    df_category_sentiment=pd.read_csv('../Output/Feedbacks_with_sentiments_predicted.txt')
    df_category_sentiment['predicted_cat']=df_category_sentiment['predicted_class'].replace(id_to_category)
    df_category_sentiment['predicted_cat_level1']=df_category_sentiment['predicted_cat'].replace(dict_level)
    df_category_sentiment.rename(columns={'predicted_cat':'predicted_cat_level2'},inplace=True)
    df_category_sentiment['CUSTOMER_MOBILE_NO']=df_category_sentiment['CUSTOMER_MOBILE_NO'].astype(str).apply(lambda x: x.split('.')[0])
    
    df_category_sentiment=df_category_sentiment[['CUSTOMER_MOBILE_NO','senti_score','pred_sentiment','predicted_cat_level2','predicted_cat_level1']]
    
    detractor_kpis=main_object.detractor_overlap_kpis(df_category_sentiment,path=r"..\..\TNPS_NUMBERS_KPIs")
    
    
    #Detractor analysis
    if file_type_str=='TNPS':
        df=pd.read_csv('../Output/Sentiment_Analysis/TNPS/Feedbacks_with_sentiments.csv')
  
        cols_to_keep=['CUSTOMER_MOBILE_NO','MONTH','GLOBAL CHANNEL',
                                     'LTR','REASON_FOR_CALL','REASON_FOR_CALL_SUBTYPE',
                                     'REASON_FOR_CALL_DFF1','pred_sentiment','senti_score']
        TNPS_Senti_df=df[cols_to_keep]
        detractor_base, overlapping_base=main_object.detractor_overlap(TNPS_Senti_df[['CUSTOMER_MOBILE_NO']],path=r"..\..\Detractor_base")
        #circle wise analysis
        overlap_base_circle_wise=pd.DataFrame(overlapping_base.groupby(['circle','LOB']).size(),columns=['Text_analytics_base'])
        detractor_base_circle_wise=pd.DataFrame(detractor_base.groupby(['circle','LOB']).size(),columns=['Detractor_base'])
        overlap_circle_wise=detractor_base_circle_wise.merge(overlap_base_circle_wise,left_index=True,right_index=True)
        overlap_circle_wise.to_csv('../Output/overlap_base_circle_wise_dropDup.csv',index=True)
        
          #Creating Wordcloud
        viz = Visualizations('JFM')
        viz.create_wordcloud(df= df, stopwords= stop_words, file_type_str = 'TNPS')
     