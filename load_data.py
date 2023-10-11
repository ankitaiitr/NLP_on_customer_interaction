# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 12:57:46 2020

@author: Ankita
"""

 
from os import path, getcwd
import glob
import pandas as pd
import numpy as np
from main import main
from pprint import pprint
from timeit import default_timer as timer
import pickle

class LoadData(main):
    """ Loads the data from different months folder and then combines it into one data frame."""
    
    
    def saveAs_pickle_file(self, directory, file_name, data_to_pickle):
        """Save given file in given directory.
        
        directory:   Directory to save file at.
        data_to_pickle: Pickle file to save.
        file_name: Name to save with."""

        print('Creating Pickle file.....')
        start = timer()
        pickle.dump(data_to_pickle, open(path.join(str(directory), str(file_name)), 'wb'))
        end  = timer()
        # print('Saved Pickle File at location: ', path.join(str(directory), str(pickle_folder_name),str(file_name)) )
        print('Time taken to pickle file: ', str(round((end-start),2)) + ' ' + 'seconds.')
        
        
    def readTNPS(self, channel_excel_file, column_file_sheet_name, column_file_path):
        """ Reads column names from specified sheet name and outputs dataframe with subset of columns which has is_text =1. 
        
        channel_excel_file :        channel report file path. Write the name of the path including extension(.xlsx)
        column_file_path:           path to "Column_names.xlsx" file
        column_file_sheet_name:     name of the sheet in columns file which corresponds to report(contact_centre, digital, retail, network) file 

        """
        print("Reading 'Column_names.xlsx' excel File....")
        column_names_excel = pd.read_excel(column_file_path, sheet_name = str(column_file_sheet_name), usecols=['is_text', 'COLUMN NAME'])
        column_names_excel = column_names_excel[column_names_excel['is_text']==1]

        column_names_list  = list(column_names_excel['COLUMN NAME'].str.upper()) #Converting to list to use to read report files. 
        print()
        print('Done Reading....')
        print()

        #Report Files
        print('Reading Report File.....')
        df= pd.read_excel(channel_excel_file, usecols = column_names_list )

    #     df.columns = df.columns.str.upper() #uppercasing : Done if there is changes in the casing in future reports
        print()
    #     print('Report columns')
    #     print(df.columns)
        print('Done Reading channel report File....')
        return df
    
    
    def LoadTNPS(self, TNPS_folder_path, report_name, column_file_path, column_file_sheet_name, month):
        """Loads data from different months folder in TNPS folder and pass to readTNPS function 
           for given column sheet name and month. 
           
           TNPS_folder_path:        path where TNPS folder is saved.
           report_name:             contact_centre/digital/retail/network
           column_file_path:        Enter path alongwith filename and ".xlsx" extension (e.g. "column file path here/Column_names.xlsx")
           column_file_sheet_name:  Excel sheet from Columns.xlsx for corresponding reports.
           month:                   For which month is required."""
        
        
        
        excel_file = ''.join(glob.glob(TNPS_folder_path+'\\'+ str(month)+ '\\' +'*' + str(report_name)+ '_'+str(month)+'.xlsx'))
        print('Files:')
        pprint(excel_file)
        
        
        df = self.readTNPS(channel_excel_file = excel_file, column_file_sheet_name = str(column_file_sheet_name), column_file_path= column_file_path)
        
        #--------------- Sanity Check ------------------------------------
        column_file_df = pd.read_excel(column_file_path, sheet_name= str(column_file_sheet_name))
        print('columns diff:\n', set(column_file_df[column_file_df['is_text']==1]['COLUMN NAME']).difference(set(df.columns)) )    #Outputs column name If any selected column name from "Column_names.xlsx" is not present in df else empty set.
        if set(column_file_df[column_file_df['is_text']==1]['COLUMN NAME']).difference(set(df.columns)) == set()  :
    #             #Adding Month tag
                df['MONTH'] = str(month)
                print(str(month)+ ' ' + str(report_name)+str(' all clear... '))
                return df

        else:
            print('Error in '+str(report_name)+' for the month of ' + str(month) )
            print()
            
            
    #------------------ NPS ---------------------------------------

    def LoadNPS(self, NPS_path, pickle_file_name = "NPSCombinedToDF.pkl"):
        """Loads data from different months folder in NPS folder and combine all data into one dataframe.
        NPS_path:             Path to NPS folder.
        pickle_file_name:    Name of pickle file to save with."""
        
        big_list_rural= []
        big_list_urban = []
        big_list = []
        print('Specified Path given from user to load data for NPS: ', str(NPS_path))
        for m in self.months_NPS:
            path = NPS_path + '\\' + str(m)
            print('Month: ', str(m))
            
            
            #Urban
            urban_files = glob.glob(path + '\\'+'VOC*'+ '*urban *' + '*.xlsx')
            print('URBAN VOC files:')
            pprint(urban_files)
            print()
            
            #Rural
            rural_files = glob.glob(path+ '\\' +'VOC*' + '*rural *'+ '*.xlsx') #rural VOC
            print('RURAL VOC FILES: ')
            pprint(rural_files)
            print()
            
            #combining files
            print('Appending urban Files in big_list_urban....')
            for f in urban_files:
                df = pd.read_excel(f)
                df['MONTH'] = str(m)
                df['VOC_TYPE'] = 'URBAN'
                big_list_urban.append(df)

            print('Appeding rural files in big_list_rural....')
            for f in rural_files:
                df= pd.read_excel(f)
                df['MONTH'] = str(m)
                df['VOC_TYPE']= 'RURAL'
                big_list_rural.append(df)
            print('--------------------------------------------------------------------------------------')
            
            
        #concatenating all dfs from list into one df
        big_list.extend(big_list_urban)
        big_list.extend(big_list_rural)

        print()
        print('Combining into single dataframe.....')
        final_df = pd.concat(big_list, axis=0, ignore_index =True)
        final_df = final_df.loc[:,~final_df.columns.str.startswith('Unnamed')]

        #Rearranging columns
        VOC_TYPE = final_df['VOC_TYPE']
        MONTH    = final_df['MONTH']

        final_df.drop(['VOC_TYPE', 'MONTH'], axis=1, inplace=True)
        final_df.insert(loc= 3, column= 'VOC_TYPE', value= VOC_TYPE)
        final_df.insert(loc= 7, column= 'MONTH', value = MONTH)

        #Removing Inconsistent data
        final_df.fillna('Null', inplace= True)
        final_df_copy = final_df
        print()
        print("Removing Null(blank) values from Circle and Text columns('VOC', 'Areas of Improvement')")
        #Removing Circle blank values
        final_df = final_df[~(final_df['Circle']=='Null')]
        #Removing Text Columns Blank values
        final_df = final_df[~((final_df['VOC']=='Null') & final_df['Areas of Improvement']=='Null') ]
        #Combining Text columns into one text column
        print('Joining Text from text columns into 1 column....')
        final_df['text_col'] =  final_df[self.NPS_text_col].astype('str').apply(lambda x : ",".join(x),axis=1)
        
        final_df['is_null'] = final_df['text_col'].apply(lambda s : 1 if s== ','.join(['Null']*len(self.NPS_text_col)) else 0 ) 
        final_df = final_df[~(final_df['is_null']==1 )] 
        final_df.drop(['is_null'], axis=1, inplace= True)
        print()
        print('Orignal data before: ', len(final_df_copy))
        print('Data after Removing blank text columns: ', len(final_df))
        print('Relative Data left: ', round((len(final_df)/len(final_df_copy)*100),2))
        print()
        
        
        #Removing 'Null' as value from newly created text column( Add 'null' as stopword.)
        print('Removing null as value from newly created text column....')
        final_df['text_col'] = final_df['text_col'].str.lower()
        final_df['text_col_std'] = final_df['text_col'].str.replace('null', '')
        # final_df.drop(['text_col'], axis=1, inplace= True)
        print('Done!!')
        


        #Creating pickle file.
#             directory, file_name, pickle_file
        print(getcwd())
        NPS_path = r"..\TEXT_P_DATA\NPS"
        # NPS_path = r"C:\Users\MUM440264\Documents\TextMining_pipeline\Output\NPS"
        self.saveAs_pickle_file(directory = NPS_path, file_name =str(pickle_file_name), data_to_pickle = final_df )
        print('Done!!')
        return final_df

if __name__ == '__main__':
    LoadData_object = LoadData('JFM')

#----------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------
from tnps_dict2df import TNPS_Dict2DF
from load_data import LoadData
import pandas as pd
from os import path

class Mine(LoadData , TNPS_Dict2DF):
 

    def MineVOCs(self, file_type_str, pickle_path_with_file=''):
        '''
            file_type_str: NPS/TNPS.
            month : list of months i.e TNPS_months or NPS_months
            pickle_path_with_file:  Path along with filename at last if file exists,else assign '' as value to it
                         e.g. pickle_path/(your_pickle_file)'''
        
        if (file_type_str == 'TNPS') & (path.isfile(str(pickle_path_with_file))) :
            print('Reading TNPS pickle file.....')
            dict_all = pd.read_pickle(str(pickle_path_with_file))
            print('Done Reading '+ file_type_str + ' pickle file.')
            print()
            return dict_all, file_type_str
                                        
        elif (file_type_str == 'TNPS') & (pickle_path_with_file == ''):
            #----- Change path here as per you folder structure. ----
            TNPS_folder_path  =  r'..\TEXT_DATA\TNPS'  
            #pickle_folder_path = path.join(TNPS_folder_path, 'Pickle_files')
            
            for key,value in self.TNPS_channels.items(): 
                    for month in self.months_TNPS:
                            print('report_name: ',key)
                            print('column_sheet: ',value)

                            print('MONTH: ',month)

                            #----- Reading all reports--------
                            self.dict_all['df_'+str(key)+'_'+str(month)] = self.LoadTNPS(TNPS_folder_path = TNPS_folder_path,
                                                                                    report_name=key,
                                                                                    column_file_path = r'../TEXT_DATA/TNPS/Column_names.xlsx',
                                                                                    column_file_sheet_name=value,
                                                                                    month= str(month))


            #------ Creating Datframe from dict
            cc_df  = self.dict_2_DF(dictn_keys= self.dict_all.keys(), global_channel_name='contact_centre')
            dg_df  = self.dict_2_DF(dictn_keys= self.dict_all.keys(), global_channel_name='digital')
            ret_df = self.dict_2_DF(dictn_keys= self.dict_all.keys(), global_channel_name='retail')
            net_df = self.dict_2_DF(dictn_keys= self.dict_all.keys(), global_channel_name='network')
            
            # Combining all df's into one df
            final_df = self.Concat_All_DFS(cc_df, dg_df, ret_df, net_df)
            final_df.fillna('Null', inplace=True)
            df = final_df
            #Remove Blank MSISDN and combine all text columns here itself
            print('Removing records with No MSISDN.......')
            final_df = final_df[~(final_df['CUSTOMER_MOBILE_NO']== 'Null')] 
            print("After Removing records with no MSISDN's data : ", len(final_df))
            print('Data left relative to orignal dataframe: '+ str(round((len(final_df)/ len(df)*100),2) )+ '%')
            print()
            #Combine All Text columns into one
            #Creating Flag to Remove Records with No text data.( Drop based on Null count in text_col row == len(text_col))
            print('Joining Text from text columns into 1 column....')
            final_df['text_col'] =  final_df[self.TNPS_text_col].astype('str').apply(lambda x : ",".join(x),axis=1)
            print('Removing records with No text data......')
            final_df['is_null'] = final_df['text_col'].apply(lambda s : 1 if s == ','.join(['Null']*len(self.TNPS_text_col)) else 0 ) 
            
            final_df = final_df[~(final_df['is_null']==1 )] 
            final_df.drop(['is_null'], axis=1, inplace= True)
            print("After Removing records with no text data : ", len(final_df))
            print('Data left relative to orignal dataframe: '+  str(round((len(final_df)/ len(df)*100),2)) +'%'  )
            
             #Removing 'Null' as value from newly created text column( Add 'null' as stopword.)
            print('Removing null as value from newly created text column....')
            final_df['text_col'] = final_df['text_col'].str.lower()
            final_df['text_col_std'] = final_df['text_col'].str.replace('null', '')
            # final_df.drop(['text_col'], axis=1, inplace= True)
            print()
            

            #Saving Dictionary to pickle file
            TNPS_folder_path_save_at = r'..\TEXT_P_DATA\TNPS'
            self.saveAs_pickle_file(directory = TNPS_folder_path_save_at, file_name = 'dict_all.pkl', data_to_pickle = self.dict_all ) #dictionary
            self.saveAs_pickle_file(directory = TNPS_folder_path_save_at, file_name = 'dict_to_df.pkl', data_to_pickle = final_df ) #dataframe
            return final_df, file_type_str

            

        elif(file_type_str == 'NPS') & (path.isfile(pickle_path_with_file)):
            print('Reading NPS pickle file.....')
            final_df_NPS = pd.read_pickle(pickle_path_with_file)
            print('Done Reading ' + file_type_str + ' pickle file.')
            return final_df_NPS, file_type_str
             
             
        elif (file_type_str =='NPS') & (pickle_path_with_file == ''):
                path_for_NPS_folder = r"..\TEXT_DATA\NPS"

                final_df_NPS = self.LoadNPS(NPS_path = path_for_NPS_folder, pickle_file_name = "NPSCombinedToDF.pkl" )
                return final_df_NPS, file_type_str
            
             
if __name__ == '__main__':
    Mine_object = Mine("JFM")
    #-------------------------------------------------------------------------------
    #-------------------------------------------------------------------------------
    import pandas as pd
from main import main

class TNPS_Dict2DF(main):
    """Converts dictionary which has key in format (global channel_month) and its value corresponding dataframe from Mine VOCs
       into Global Channelwise data frame first and then combine these created datframe into one final dataframe."""
    
    def dict_2_DF (self, dictn_keys, global_channel_name ):
        """global_channel_name: Name for which to search and filter in the given dictn_keys."""
        
        """Steps: 1. Filter out required values based on input  global_channel_name, This function output will be list
                        
                  2. Use this functions output to create a list of series, where each series in the list of dataframes which were values to the corr.dictn key.
                  3. Conctenate the list to create data frame. """
        subset_list = list(filter(lambda x: str(global_channel_name) in x, dictn_keys) )  #1.
        list_of_dfs = pd.Series(subset_list).map(lambda x : self.dict_all[str(x)]).to_list()   #2.
        df          = pd.concat(list_of_dfs, axis=0, ignore_index= True)
        return df
    
    def Concat_All_DFS(self, *args):
            """Concatenates multiple DFS to one dataframe."""
            final_df = pd.concat([*args], axis=0, ignore_index= True)
            return final_df


if __name__ == '__main__':
    TNPS_Dict2DF_object = TNPS_Dict2DF()
    
