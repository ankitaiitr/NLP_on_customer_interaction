# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:41 2020

@author: Ankita
"""


import pandas as pd
import os
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer#If this does not work uncomment below line
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import csv


class SentimentAnalysis():

    def sentiment_assignment(self, text_column_for_senti):


            vs = SentimentIntensityAnalyzer()
            #Reading telecom specific word-polarity csv for scores.
            new_words= {}
            reader = csv.reader(open(r'Telecom_words_for_Vader.csv', "rt", encoding='utf-8'))
            for i, rows in enumerate(reader):
                        for rows in reader:
                            k = rows[0]
                            v = rows[1]
                            new_words[k] = float(v)
                
            # print(new_words)  #To see what words and their corresponding scores are updated in vader lexicon dictionary

            #updating new words in analyzer
            vs.lexicon.update(new_words)
            senti_score = text_column_for_senti.apply(lambda index: round(vs.polarity_scores(index)['compound'],2) ) #Since vaderSentiment provides a dictionary accesing 'compound' key gave values
            # print(senti_score)
            senti_score = pd.Series(senti_score)
            pred_senti  = senti_score.apply(lambda s : 'positive' if(s >0.09) else('neutral' if (s==0) else('negative')) )

            return senti_score, pd.Series(pred_senti)



    def sentiment_comparison(self, df,default_senti_column_name, pred_senti_column_name, file_type_str, months_start_end_initials):
         """Compares the sentiment Assigned by the algorithm v/s sentiment column present Monthwise in data 
             df                            : dataframe subset with only default sentiment column and predicted sentiment(created by assigning sentiments to text) column.
                                              e.g. df[['senti_column', 'pred_senti_column_name']]
             default_senti_column_name_name     : sentiment column already present in the data.
             pred_senti_column_name_name        : Column containing sentiment assigned by the algorithm.
             file_type                     : 'TNPS'/'NPS'.
             months_start_end_initials     : Please enter required months initials together in a string. 
                                            {'JFM' : ["JAN","FEB","MAR"],
                                              'AMJ' : ["APR","MAY","JUN"],
                                              'JAS' : ['JUL','AUG','SEP'],
                                              'OND' : ['OCT','NOV','DEC']}

             Return Type: .txt file."""

         months = {'JFM' : ["JAN","FEB","MAR"],
                    'AMJ' : ["APR","MAY","JUN"],
                    'JAS' : ['JUL','AUG','SEP'],
                    'OND' : ['OCT','NOV','DEC']
                  }
         path = os.path.join('../Output/Sentiment_Analysis', str(file_type_str),'VOCs_metrics.txt')
         with open(path, 'a+') as f:
            f.truncate(0)  #For clearing the text file if it contains any previous data
            f.close()

            for month in months[str(months_start_end_initials)]:
                print('Month: ', month)
                print()
                #Drop records which contains values other than 'positve, negative, neutral.
                # -------------------- Accuracy -------------------------

                x = df[(df[str(default_senti_column_name)] == 'positive') & (df[str(pred_senti_column_name)] <0) ].count()
                y = df[(df[str(default_senti_column_name)] == 'negative') & (df[str(pred_senti_column_name)] > 0)].count()
                z = df[(df[str(default_senti_column_name)] == 'neutral') &  (df[str(pred_senti_column_name)] !=0)].count()

                pos_acc = 1- ( x/df[df[str(default_senti_column_name)]== 'positive'].count()) 
                neg_acc = 1- (y/ df[df[str(default_senti_column_name)]== 'negative'].count()) 
                neu_acc = 1- (z /df[df[str(default_senti_column_name)] == 'neutral'].count()) 

                print('Positive Accuracy: ', pos_acc[0])
                print('Negative Accuracy: ', neg_acc[0])
                print('Neutral Accuracy: ',  neu_acc[0])
                print()

                #Class Positive
                print('positive ')
                P_TP = df[(df[str(default_senti_column_name)] == 'positive') & (df[str(pred_senti_column_name)] >  0)  ].count()[0]
                print('TP:', P_TP)
                P_TN = df[(df[str(default_senti_column_name)] != 'positive') & (df[str(pred_senti_column_name)] <= 0)  ].count()[0]
                print('TN: ', P_TN)
                P_FP = df[(df[str(default_senti_column_name)] != 'positive') & (df[str(pred_senti_column_name)] >  0)  ].count()[0]
                print('FP: ', P_FP)
                P_FN = df[(df[str(default_senti_column_name)] == 'positive') & (df[str(pred_senti_column_name)] <= 0)  ].count()[0]
                print('FN: ', P_FN)
                print()

                #Class Negative
                print('Negative')
                Ng_TP = df[(df[str(default_senti_column_name)] == 'negative') & (df[str(pred_senti_column_name)] <  0)  ].count()[0]
                print('TP: ', Ng_TP)
                Ng_TN = df[(df[str(default_senti_column_name)] != 'negative') & (df[str(pred_senti_column_name)] >= 0)  ].count()[0]
                print('TN: ', Ng_TN)
                Ng_FP = df[(df[str(default_senti_column_name)] != 'negative') & (df[str(pred_senti_column_name)] <  0)  ].count()[0]
                print('FP: ', Ng_FP)
                Ng_FN = df[(df[str(default_senti_column_name)] == 'negative') & (df[str(pred_senti_column_name)] >= 0)  ].count()[0]
                print('FN: ', Ng_FN)
                print()


                #Class Neutral
                print('Neutral')
                Ne_TP = df[(df[str(default_senti_column_name)] == 'neutral')  & (df[str(pred_senti_column_name)] ==  0)  ].count()[0]
                print('TP: ', Ne_TP)
                Ne_TN = df[(df[str(default_senti_column_name)] != 'neutral')  & (df[str(pred_senti_column_name)] !=  0)  ].count()[0]
                print('TN: ', Ne_TN)
                Ne_FP = df[(df[str(default_senti_column_name)] != 'neutral')  & (df[str(pred_senti_column_name)] ==  0)  ].count()[0]
                print('FP: ', Ne_FP)
                Ne_FN = df[(df[str(default_senti_column_name)] == 'neutral') &  (df[str(pred_senti_column_name)] !=  0)  ].count()[0]
                print('FN: ', Ne_FN)

                # ------------------- Precision-Recall ---------------------
                #Positive
                Pos_P = P_TP/(P_TP + P_FP )
                print('Positive Class Precision: ', Pos_P)
                Pos_R = P_TP/(P_TP + P_FN)
                print('Positive Class Recall: ', Pos_R)

                #Negative
                Ng_P = Ng_TP/(Ng_TP + Ng_FP )
                print('Negative Class Precision: ', Ng_P)
                Ng_R = Ng_TP/(Ng_TP + Ng_FN)
                print('Negative Class Recall: ', Ng_R)

                #Neutral
                Ne_P = Ne_TP/(Ne_TP + Ne_FP )
                print('Neutral  Class Precision: ', Ne_P)
                Ne_R = Ne_TP/(Ne_TP + Ne_FN)
                print('Neutral Class Recall: ', Ne_R)
                print()


                #----- Writing in text file
                print('Writing In Text file...')
                with open(path, 'a+') as f:
                    f.write('MONTH: '+ str(month)+ '\n')
                    f.write('-         ACCURACY     '+'\n')
                    f.write('Positive Accuracy: ' + str(round(pos_acc[0],2)) + '\n')
                    f.write('Negative Accuracy: ' + str(round(neg_acc[0],2)) + '\n')
                    f.write('Neutral Accuracy: '  + str(round(neu_acc[0],2)) + '\n')
                    f.write('\n')

                    #----------- Precision - Recall-------
                    f.write('    PRECISION - RECALL      '+'\n')
                    f.write('Positive Class Precision: '+str(round(Pos_P,2))+'\n')
                    f.write('Positive Class Recall: '   +str(round(Pos_R,2))+'\n')
                    f.write('Negative Class Precision: '+str(round(Ng_P,2))+'\n')
                    f.write('Negative Class Recall: '   +str(round(Ng_R,2))+'\n')
                    f.write('Neutral Class Precision: ' +str(round(Ne_P,2))+'\n')
                    f.write('Neutral Class Recall: '    +str(round(Ne_R,2))+'\n')
                    f.write('-----------------------------------------------'+ '\n')

         print('Done!!')
         print('Text File is saved at: '+path)
 #------------------------------------------------------
 #--------------------------------------------------------
