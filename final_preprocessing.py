# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:52 2020

@author: Ankita
"""
import spacy
import en_core_web_sm
import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from Visualizations import Visualizations

class finalprocessing(Visualizations):

    nlp = spacy.load('en_core_web_sm')
    telco_dictn = {
               'voc'        : 10,
               'tnps'       : 10,
              'telecom'     : 10, 
              'prepaid'     : 10,
              'postpaid'    : 10,
              'connectivity': 10,
              'sim'         : 10,
             'offer'        : 10,
              'voucher'     : 10,
              'balance'     : 10,
              'detractor'   : 10,
              'coverage'    : 10,
              'tariff'      : 10,
              'validity'    : 10,
              'flytxt'      : 10,
              '4g'          : 10,
              '3g'          : 10,
              '2g'          : 10,
              'app'         : 10,
              'billing'     : 10,
              'plan'        : 10,
              'deduction'   : 10,
              'calls'       : 10,
              'roaming'     : 10,
              'arpu'        : 10,
              # 'handset'     : 10,
              'signal'      : 10,
              'activate'    : 10,
              'deactivate'  : 10,
              'port'        : 10,
              'disconnect'  : 10,
              'charges'     : 10,
              'dnd'         : 10,
              'vas'         : 10,
              'sms'         : 10,
              'voice'       : 10,
              'retention'   : 10}

        

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
            doc = finalprocessing.nlp(doc, disable=['parser', 'ner'])
            tokens = [tok.lemma_.strip() for tok in doc if tok.lemma_ != '-PRON-']
            tokens = ' '.join(tokens)
            texts.append(tokens)
        # print(texts)
        return pd.Series(texts)


    def ngram_sum(self,df):
        """Creating a new column for arranging them in descending order."""

        df= df.sum(axis=0).T


        #Sorting dataframes according to frequency
        df.sort_values(ascending =False,inplace=True)
        sum_df = pd.DataFrame(df,columns =['sum'])
        sum_df.reset_index(inplace= True)
        sum_df.rename(columns= {'index': 'terms'},inplace= True)
        return sum_df 


    def create_n_grams(self,df, file_type_str, stopwords,bonus_score =10,min_df=1):
        """Creates Unigrams/bigrams channel wise.
    df             : Text data frame from which we wish to extract unigrams/bigrams.
                     Flag either to extract unigrams or bigrams.
    file_type      : TNPS/NPS.
    file_type_list : list of different channels based on file type 
                     if TNPS --> ['contact_centre', 'digital', 'network', 'retail']
                     IF NPS  --> ['RURAL', 'URBAN'].
    stopwords      : user defined stopwords 
    bonus_score    : Increment frequency of encountered domain specific words by this score, by default =10.
                     
    min_df         : value --> int/float, default =0.01
                     Minimum no. of occurence criteria to qualify as unigram/bigram.
    Return Type: Unigram, Bigrams dataframes along with excel workbook for unigram and bigram."""
        
        path = os.path.join('../Output/Popular_N_grams', file_type_str)
        writer = pd.ExcelWriter(os.path.join(path, 'N-grams.xlsx'))
        # Visualizations_obj = Visualizations("JFM") #for plotting frequency histograms.
        if file_type_str == 'TNPS':
            for channel in list(self.TNPS_channels.keys()):
                #----------- UNIGRAMS ---------------------
                print('Creating unigrams using CountVectorizer for channel -->' + str(channel) )
                cvu = CountVectorizer(stop_words = stopwords,ngram_range=(1,1), min_df=min_df)
                eng_cvu = cvu.fit_transform(df[df['GLOBAL CHANNEL']== str(channel)]['final_text'].values)
                print('Unigrams Shape:',eng_cvu.shape)
                # print('Creating Dataframe.....')
                eng_dfu = pd.DataFrame(eng_cvu.toarray(), columns=cvu.get_feature_names())
                # print('Summing up for arranging them in descending order......')
                cvu_df = self.ngram_sum(eng_dfu)

                #------------ BIGRAMS ------------
                print('Creating bigrams using CountVectorizer for channel -->' + str(channel) )
                cvb = CountVectorizer(stop_words = stopwords, ngram_range=(2,2), min_df= min_df)
                eng_cvb = cvb.fit_transform(df[df['GLOBAL CHANNEL']== str(channel)]['final_text'].values) 
    #             print(eng_cvb)
                eng_dfb = pd.DataFrame(eng_cvb.toarray(), columns=cvb.get_feature_names())
    #             print(eng_dfb)
                print('Bigrams Shape: ', eng_dfb.shape)

                # print('Creating Dataframe.....')
                cvb_df = self.ngram_sum(eng_dfb)
                #------- Fixing Palindromic words to avoid duplicating(good network, network good etc)
                """ process: 1. Spliting the bigrams on space which creates list of words corresponding each bigrams, 
                                 2. Sorting the words in alpphabetical order then unlisiting them with space as separator.
                                 3. Groupby words so that identical bigrams combine and add up their frequencies(sum)
                                 4. Arrange in descending order of the frequencies(sum)"""

                cvb_df['terms'] = cvb_df['terms'].apply(lambda x: ' '.join(sorted(x.split()))) #1
                cvb_df =  cvb_df.groupby('terms').sum().reset_index().sort_values(by=['sum'], ascending=False) #2,3,4
                #----------------------- Increasing weightage of domain specific words----------------------------------
                #For unigrams
                cvu_df.loc[(cvu_df['terms'].isin(finalprocessing.telco_dictn.keys())), 'sum'] += int(bonus_score)
                #For bigrams 
                cvb_df[['terms1','terms2']]  = cvb_df['terms'].str.split(expand=True)
                cvb_df.drop(cvb_df.loc[cvb_df['terms1']==cvb_df['terms2']].index, axis=0 ,inplace=True) #Dropping repeated bigrams(e.g. 'network network')

                cvb_df.loc[ (cvb_df['terms1'].isin(finalprocessing.telco_dictn.keys()) | cvb_df['terms2'].isin(finalprocessing.telco_dictn.keys()) ) , 'sum'] += int(bonus_score)
                cvb_df.drop(['terms1', 'terms2'], axis=1, inplace= True)

                #---------------------------- Plotting Frequency histograms.---------------------------------------------
                #Unigrams Plot
                cvu_df.sort_values(by=['sum'], ascending= False, inplace= True)
                self.plot_words_frequency_histogram(x_axis_data= cvu_df['terms'],  y_axis_data= cvu_df['sum'], plot_title= str(channel) + ' UNIGRAMS', file_type_str= file_type_str, no_of_words_to_plot= 20)
                #Bigrams Plot
                cvb_df.sort_values(by=['sum'], ascending= False, inplace= True)
                self.plot_words_frequency_histogram(x_axis_data= cvb_df['terms'],  y_axis_data= cvb_df['sum'], plot_title= str(channel)+ ' BIGRAMS', file_type_str= file_type_str, no_of_words_to_plot= 20)
                #Converting Images to PDF
                self.img_to_pdf(images_folder_path= os.path.join('../Output/Popular_N_grams/', file_type_str), pdf_name= file_type_str+'_Frequency_histogram')
                #--------------------------- Writing in Excel-----------------------------------------
                #Unigrams
                print()
                print('Writing Unigrams In Excel.....')
                cvu_df[['terms','sum']].to_excel(writer, sheet_name = str(channel)+'_unigrams_', index= False, header= True )
                print('Done')

                #Bigrams
                print()
                print('Writing Bigrams In Excel.....')
                cvb_df[['terms','sum']].to_excel(writer,  sheet_name = str(channel)+'_bigrams_', index= False, header= True)
            writer.save()
            writer.close()

        elif file_type_str == 'NPS':
            for channel in list(self.NPS_categories):
                #------------  COUNTVECTORIZER UNIGRAM ------------
                print('Creating unigrams using CountVectorizer for channel ---> '+ str(channel))
                cvu = CountVectorizer(stop_words = stopwords, ngram_range=(1,1), min_df=min_df)
                eng_cvu = cvu.fit_transform(df[df['VOC_TYPE']== str(channel)]['final_text'].values)
                print('Unigrams Shape:',eng_cvu.shape)

                # print('Creating Dataframe.....')
                eng_dfu = pd.DataFrame(eng_cvu.toarray(), columns=cvu.get_feature_names())

                #summing up
                # print('Summing up for arranging them in descending order......')
                cvu_df = self.ngram_sum(eng_dfu)


                #------------  COUNTVECTORIZER BIGRAM------------
                print('Creating bigrams using CountVectorizer for channel -->' + str(channel) )
                cvb = CountVectorizer(stop_words = stopwords, ngram_range=(2,2), min_df= min_df)
                eng_cvb = cvb.fit_transform(list(df[df['VOC_TYPE']== str(channel)]['final_text'])) 
                eng_dfb = pd.DataFrame(eng_cvb.toarray(), columns=cvb.get_feature_names())
                print('Bigrams Shape: ', eng_dfb.shape)

                # print('Creating Dataframe.....')
                cvb_df = self.ngram_sum(eng_dfb)
                
                #------- Fixing Palindromic words to avoid duplicating(good network, network good etc)
                """ process: 1. Spliting the bigrams on space which creates list of words corresponding each bigrams, 
                             2. Sorting the words in alpphabetical order then unlisiting them with space as separator.
                             3. Groupby words so that identical bigrams combine and add up their frequencies(sum)
                             4. Arrange in descending order of the frequencies(sum)"""

                cvb_df['terms'] = cvb_df['terms'].apply(lambda x: ' '.join(sorted(x.split()))) #1
                cvb_df =  cvb_df.groupby('terms').sum().reset_index().sort_values(by=['sum'], ascending=False) #2,3,4



                #---------------------  Increasing weightage of domain specific words----------------------
                #For unigrams
                cvu_df.loc[(cvu_df['terms'].isin(finalprocessing.telco_dictn.keys())), 'sum'] += int(bonus_score)
                #For bigrams
                cvb_df[['terms1','terms2']]  = cvb_df['terms'].str.split(expand=True)
                cvb_df.drop(cvb_df.loc[cvb_df['terms1']==cvb_df['terms2']].index, axis=0 ,inplace=True) #Dropping repeated bigrams(e.g. 'network network')
                cvb_df.loc[ (cvb_df['terms1'].isin(finalprocessing.telco_dictn.keys()) | cvb_df['terms2'].isin(finalprocessing.telco_dictn.keys()) ) , 'sum'] += int(bonus_score)
                cvb_df.drop(['terms1', 'terms2'], axis=1, inplace= True) 
                

                #--------------------- Plotting Frequency histograms.-----------------------------------
                #Unigrams Plot
                cvu_df.sort_values(by= ['sum'], ascending= False,inplace=True)
                self.plot_words_frequency_histogram(x_axis_data= cvu_df['terms'],  y_axis_data= cvu_df['sum'], plot_title= str(channel)+' UNIGRAMS', file_type_str= file_type_str, no_of_words_to_plot= 10)
                #Bigrams Plot
                cvb_df.sort_values(by= ['sum'], ascending= False,inplace=True)
                self.plot_words_frequency_histogram(x_axis_data= cvb_df['terms'],  y_axis_data= cvb_df['sum'], plot_title= str(channel) + ' BIGRAMS', file_type_str= file_type_str, no_of_words_to_plot= 10)

                #Converting Images to PDF
                self.img_to_pdf(images_folder_path= os.path.join('../Output/Popular_N_grams/', file_type_str), pdf_name= file_type_str+'_Frequency_histogram')
                #-------------------- Writing in Excel -------------------------------------------
                #Unigrams
                print()
                print('Writing Unigrams In Excel.....')
                cvu_df[['terms','sum']].to_excel(writer, sheet_name = str(channel)+'_unigrams_', index=False, header= True )
                print('Done')

                #Bigrams
                print()
                print('Writing Bigrams In Excel.....')
                cvb_df[['terms','sum']].to_excel(writer,  sheet_name = str(channel)+'_bigrams_', index=False, header=True)
            writer.save()
            writer.close()



        print('Done')
        print('N-grams Excel file is saved at location: ', path)
        return cvu_df, cvb_df

