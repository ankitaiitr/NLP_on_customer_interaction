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
    