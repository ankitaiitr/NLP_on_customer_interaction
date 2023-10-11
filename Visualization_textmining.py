# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:54 2020

@author: Ankita
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS,ImageColorGenerator
import PIL.Image
import os
import img2pdf

from TextPreProcessing import Preprocessing


class Visualizations(Preprocessing):


    def img_to_pdf(self,images_folder_path, pdf_name):
        current_path = os.getcwd() #store the path of the current directory
        
        with open(os.path.join(images_folder_path,str(pdf_name) + ".pdf"), "wb") as f:
            os.chdir(images_folder_path)  
                #next line will convert all the images with extension as '.jpg' in the directory to pdf
            f.write(img2pdf.convert([i for i in os.listdir(os.getcwd()) if i.endswith('.jpg')]) )  
        print('PDF file is saved at location: ', images_folder_path)
        os.chdir(current_path)


    def create_wordcloud(self, df, stopwords, file_type_str):
    #     """Creating wordcloud 
    #     text_data: text data for wordcloud.
    #     file_type_str: NPS/TNPS.
    #     """

        #mask
        mask = np.array(PIL.Image.open('vodafone_icon.png'))
        path =os.path.join(r'../Output/Word_Cloud',str(file_type_str) )
        

#        fig = plt.figure()
    # create_wordcloud(self, text_data, file_type_str)
        if file_type_str == 'TNPS':
            for i in range(len(self.TNPS_channels.keys())):
                # ax = fig.add_subplot(2,2,i+1)
                for j in self.months_TNPS:
                    words = df[(df['GLOBAL CHANNEL']== str(list(self.TNPS_channels.keys())[i])) & (df['MONTH']==j)]['text_col_std']
                    
                    print('Channel: ', str(list(self.TNPS_channels.keys())[i]))
                    print('MONTH:',j)
                    print(words)
                    words = ' '.join([word for word in words])
                    # print(words)
                    # print('Creating wordcloud for'+ file_type_str + ' ---> '+ str(list(self.TNPS_channels.keys())[i]))
                    # #wordcloud           
                    wc = WordCloud(width =800, height=800, background_color="white", mask=mask, contour_width=2, contour_color='red', stopwords = stopwords)
                    # # generate word cloud
                    wc.generate(words)
                    plt.figure(figsize =(12,12))
                    plt.imshow(wc, interpolation='bilinear')
                    # print('title: ', list(self.TNPS_channels.keys())[i] + ' Wordcloud')
                    plt.title(list(self.TNPS_channels.keys())[i] + ' Wordcloud', loc = 'center')
                    plt.axis("off")
                    plt.savefig(os.path.join(path, list(self.TNPS_channels.keys())[i] +j+ ' Wordcloud.jpg'), dpi=200)
                    # plt.show()

        elif file_type_str == 'NPS':
            for i in range(len(self.NPS_categories)):
                # ax = fig.add_subplot(2,2,i+1)
                text_data = df[df['VOC_TYPE']== (self.NPS_categories[i])]['text_col_std']
                words = ' '.join([word for word in text_data])
                print('Creating wordcloud for '+ file_type_str + ' ---> '+ (self.NPS_categories[i]))
                #wordcloud           
                wc = WordCloud(width =800, height=800, background_color="white", mask=mask, contour_width=2, contour_color='red', stopwords = stopwords)
                # generate word cloud
                wc.generate(words)
                plt.figure(figsize =(12,12))
                plt.imshow(wc, interpolation='bilinear')
                plt.title(self.NPS_categories[i]+ ' Wordcloud', loc = 'center')
                plt.axis("off")
                plt.savefig(os.path.join(path,(self.NPS_categories[i])+ ' Wordcloud.jpg'), dpi=200)
                # plt.show()
        
        print('Wordclouds are saved at: ', path)


    def plot_words_frequency_histogram(self, x_axis_data, y_axis_data, plot_title, file_type_str, no_of_words_to_plot=10):
        """Creates word frequency histograms from the given N-grams(unigram/bigram) data."""
        plot_title = 'TOP '+ str(no_of_words_to_plot) +' ' + str(plot_title)
        print('Plotting frequency histogram for '+ str(plot_title)+'.....')
        path = os.path.join('../Output/Popular_N_grams',file_type_str)
        sns.barplot(x= x_axis_data[:int(no_of_words_to_plot)], y = y_axis_data[:int(no_of_words_to_plot)])
        plt.xlabel('terms')
        plt.ylabel('Count')
        plt.title(str(plot_title),fontsize=20, fontweight= 'bold')
        plt.xticks(rotation=90, fontsize=15, fontweight= 'bold')
        plt.yticks(fontsize=15, fontweight= 'bold')
        plt.savefig(os.path.join(path, plot_title)+ '.jpg', dpi=200,bbox_inches = 'tight' )
        print('Frequency plot for '+ plot_title + ' is saved at location: ', path)
        # plt.show()
        
#To test if this class is working or not uncomment below lines.      
if __name__ == '__main__':
    
    Visualizations_obj = Visualizations('JFM')
    # #Wordcloud
    # file_type_str = 'TNPS'
    # df= pd.read_csv(os.path.join('../TEXT_P_DATA/', file_type_str,'English_Feedbacks.csv'))
    # # def create_wordcloud(self, df, stopwords, file_type_str):
    # Visualizations_obj.create_wordcloud(df, stopwords= self.stop_words,file_type_str=file_type_str )
        

#     df_uni= pd.read_excel(r'C:\Users\mum435022\Desktop\TextMining_pipeline\Output\Popular_N_grams\TNPS\N-grams.xlsx',sheet_name= 'contact_centre_unigrams_')
#     print(df_uni)
#     df_bi= pd.read_excel(r'C:\Users\mum435022\Desktop\TextMining_pipeline\Output\Popular_N_grams\TNPS\N-grams.xlsx',sheet_name= 'contact_centre_bigrams_')
#     print(df_bi)
#     Visualizations_obj.plot_words_frequency_histogram(x_axis_data=df_uni['terms'], y_axis_data=df_uni['sum'], plot_title='TNPS' +' unigram'
#                                                         ,file_type_str= 'TNPS')
#     Visualizations_obj.plot_words_frequency_histogram(x_axis_data=df_bi['terms'], y_axis_data=df_bi['sum'], plot_title='TNPS' + ' bigram' ,file_type_str= 'TNPS')
#     Visualizations_obj.img_to_pdf(images_folder_path= '../Output/Word_Cloud/TNPS', pdf_name='test')
        




