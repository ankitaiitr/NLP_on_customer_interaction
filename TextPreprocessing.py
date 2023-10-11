# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:19 2020

@author: Ankita
"""

#-----Text_preprocessing

import numpy as np
import pandas as pd
import string
import nltk.corpus
import os
from main import main
nltk.download('words')
nltk.download('stopwords')
nltk.download('punkt')
class Preprocessing(main):
    count = 0 # for keeping track of how many rows/documents has passed through set_flag_if_english_text function.
    text_correction_dict = {     
                'coz'              : ' because ',
                'bcoz'             : ' because ',
                'bcz'              : ' because ',
                ' cust '           : ' customer ',
                "(^|\s)cust($|\s)" : ' customer ',     # Start of the string or space followed by 'cust' end of the string or space.
                'problm'           : ' problem ',
                'probelum'         : ' problem ',
                'probulem'         : ' problem ',
                ' pblm '           :  ' problem ',
                'broblem'          :  ' problem ',
                "rpoblem"          : "problem",
                "porblem"          : "problem ",
                'vodaphone'        : ' vodafone ',
                ' voda '           :' vodafone ',
                 " no\. "           : ' number ',
                "nos\."             : ' numbers ',
                " nos "            : ' numbers ',
                " chet "           :  " chat ",
                " amt "            : ' amount ',
                "benifit"          : ' benefit ',
                " nyc "            : ' nice ',
                " nic "            :  ' nice ',
                " internate "      : ' internet ',
                " intrnet "        : ' internet ',
                " intrnet "        : " internet ",
                " internat "       : " internet ",
                " net "            : " internet ",
                "(^|\s)net($|\s)"  : " internet ",   # Start of the string or space followed by 'net' end of the string or space.
                "netwrk "          :  "network"  ,
                "metwork"          :  "network"  ,
                "solvef"           : " solved " ,
                "vodafonefone"     : " voda ",
                "nicee"            : " nice ",
                "thank"            :" thanks ",
                " ur "             : " your ",
                " ok "             : " okay ",
                "gud"              : " good", 
                "customeromer"     : " customer ",
                " r "              : " are ",
                " im "             : " i am ",
                "i m"              : " i am "  ,
                "oll"              : "all"   ,
                " godo "           :  " good " ,
                "conenctivity" : "connectivity",
                "disconnectn"  : "disconnection",
                " its "        : "it is", 
                " dro "        : " drop ",
                " faciilities ": " facilities ", 
                " cist "       : " cost "      ,
                " servce "     : " service ",
                "n\'t"         : " not",
                "\'re"         : " are",
                "\'ll"         : " will",
                "\'ve"         : " have",
                "\'s"          : " is",
                "\'m"          : " am",
                "\'ll've"      : " will have",
                "n\'t\'ve"     : " not have",
                "\'d"          :" had"
                
        
            }

    def setting_stopwords_corpus(self):

        #------------------- Setting Up corpus ---------------
        c = 0
        words_list = [] 
        # corpus_path = r'C:\Users\mum435022\Desktop\TextMining_pipeline\Code'     #change to path where you have saved english_corpus.txt)
        f =  open('english_corpus.txt', 'r')
        for i in f:
            i = i.replace('\n', '')

            words_list.append(i)
            c +=1

        #Set of English words from nltk
        corp1 = set(w.lower() for w in nltk.corpus.words.words())
        corp2 = set(words_list)


        #-------------- Creating corpus--------------
        words_corpus = corp1.union(corp2)
        to_add_in_corpus = ['voc','tnps', 'telecom',' internet', 'email', 'postpaid', 'prepaid', 'connect', 'vodafone','distruptions','sim',
                              'executive','internet']

        for word in to_add_in_corpus:
            if word not in words_corpus:
        #         print(word)
                words_corpus.add(word)


        #------------------------- Stopwords -------------------------------------
        stop_to_add =['get', 'said', 'say' ,'face', 'facing','given', 'spoke', 'feedback', 'taken', 'regarding', 'agreed', 'customer', 'tnps','particular','probe','apology', 'share', 'voc', 'ibcc', 'also','hence'] 
        stop_to_keep= ["no", "don't", "not", "shouldn't", "mustn't", "won't", "wouldn't", "isn't", "hasn't", "hadn't", 'down']
        stop_words = nltk.corpus.stopwords.words('english')

        #removing Negation words from stop_words
        # print("hadn't" in stop_words)
        stop_words = [ stop for stop in stop_words if stop not in stop_to_keep]

        #Removal verification
        # for i in stop_to_keep:
        #     if i in stop_words:
        #         print(' Removed but was present as stopwords: ',i)
        #         pass

        #Adding words in stop_words       
        for i in stop_to_add:
            if i not in stop_words:
                stop_words.append(i)
       

        #Addition verification
        # for i in stop_to_add:
        #     if i not in stop_words:
        #         print('Added but not present as stopwords: ',i)
        #         pass

        stop_words =set(stop_words)

        
        return words_corpus, stop_words






     
    def standardization(self, text_column, stopwords, english_corpus):
        """ Performs: 1. Stopwords Removal.
                      2. Punctuation Removal.
                      3. Spell correction. 

            text_column: Pass the column as Series e.g.(df['text_column'])"""

        #Spell correction
        print('Correcting Spellings.....')
        text_column = text_column.replace(Preprocessing.text_correction_dict, regex=True)


        #Removing Punctuations
        print('Removing Punctuations.....')
        #   # Refer NOTES to understand the logic ***
        text_column = text_column.apply(lambda x: (str(x).lower()).translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation))).replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2, ' ').strip()) 


        #Removing Stopwords
        print('Removing Stopwords.......')
        text_column = text_column.str.split().apply(lambda x: ' '.join([word for word in x  if word not in (stopwords)]))

        #Removing single letter words         ** Refer at the end of code for logic
        print('Remove single letter words.....')
        text_column = text_column.str.replace(r'[0,1,5-9]+', '').str.replace(r'\b\w\b', '')
        text_column.drop(text_column[text_column == ''].index, axis=0, inplace= True)

        #Remove blank rows after processing
        # text_column = text_column.reset_index().drop(['index'],axis=1)
        # print(text_column)
        return text_column

    def tokenization(self, text_column):
        """text_column: Pass the column as Series e.g.(df['text_column'])"""
        print('Tokenizing......')
        text_column = text_column.apply(lambda line: nltk.word_tokenize(str(line)) )
        text_column = text_column[~(text_column.str.len() ==1 )]
        # text_column = text_column.reset_index()
        # text_column.drop(['index'], axis=1, inplace=True)
        print('Done tokenizing')
        print()
        return text_column

    def set_flag_if_english_text(self, text_row, total_df_rows, english_corpus ):
        """This function analyzes if 60 % of the words of total words in the sentence is english then return value as 1 else 0.
           text_row:   Pass the column as Series e.g.(df['text_column']).
           total_df_rows: Total no of rows in dataframe.
           english_corpus:  English corpus file."""
        
        #------------- Setting Flag ----------------------
        flag_val = 0
    
        count_words= []
        text_row= str(text_row)
        for word in text_row:
            if word in english_corpus:
                count_words.append(word)
        if round((len(count_words)/len(text_row)),1) >= 0.6:
            flag_val = 1
        Preprocessing.count +=1
        if Preprocessing.count%1000 == 0 :
            print('Processed '+ str(Preprocessing.count) +' documents.')
            print('Remaining Rows: ', total_df_rows- Preprocessing.count)
            print()
            print()

        return pd.Series(flag_val)


    def English_text_to_excel_or_csv(self,df_text, file_type_str):
        """Save the subsetted flaged as English customer comments in Excel."""
        
        df =df_text
        df_text = df_text[df_text['flag'] ==1]
        # Dropping unecessary columns
        df_text.drop(['flag'], axis=1, inplace=True)
        df.drop(['text_col', 'flag'], axis=1, inplace=True)
        print('English Feedback data: ', df_text.shape)
        print('Total Feeback data: ', df.shape)
        print('Percentage of english feedback data: '+ str(round((len(df_text)/len(df))*100,2)) + '%')
        print()

        # Writing in Excel/Csv
        checkpoint1 = os.getcwd()
        path = os.path.join(r'../TEXT_P_DATA',str(file_type_str))
        
        if file_type_str == 'NPS':
            os.chdir(path)
            print('Writing data in Excel')
            with pd.ExcelWriter('Feedbacks.xlsx', engine='xlsxwriter') as writer:
                    df.to_excel(writer, sheet_name='all_feed',index= False, header=True)
                    df_text.to_excel(writer, sheet_name = 'english_feed', index=False, header=True)
                    writer.save()
                    writer.close()

        elif file_type_str == 'TNPS':
            os.chdir(path)
            print('Writing all feedbacks to csv....')
            df.to_csv('Feedbacks.csv', index= False, header= True)
            print('Writing English Feedbacks to csv....')
            df_text.to_csv('English_Feedbacks.csv', index= False, header= True)
            print('Done!!')
            print()



        print('File is saved at location: ', os.getcwd())
        os.chdir(checkpoint1)
        return df_text
        


                    
            
            
            
if __name__ == '__main__':
    obj = Preprocessing()

    
   

# ------------------------------- IMP NOTES--------------------------

#Notes:*** 
#1. str.maketrans(string.punctuation, ' '*len(string.punctuation): Replaces no. of  punctuation in the sentence with that much no of spaces. 
#                                                                  e.g. s= '!@#karan' ---> str.maketrans(string.punctuation, ' '*len(string.punctuation) will produce '   karan' (NOTICE 3 SPACES) 

#2. replace(' '*4, ' ').replace(' '*3, ' ').replace(' '*2,' ').strip():  Removes the extra spaces which were created after replacing No. of
#                                                                        punctuations with that much no of spaces.



# 3. Removing single letter word:
        # https://stackoverflow.com/questions/6664151/difference-between-b-and-b-in-regex
        # https://stackoverflow.com/questions/41736038/remove-single-letters-from-strings-in-pandas-dataframe
        
        
        
        #------------------------------------------------------------------
        #-----------------------------------------------------------------
