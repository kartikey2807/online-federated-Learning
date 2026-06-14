import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset
from config import *
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib.pyplot as plt

class CausalChamberDataset(Dataset):

    '''
    Causal chamber data is divided 
    into subsequences, and pytorch 
    training set is created.
    '''

    def __init__(
            self,
            anomaly:bool=False,
            type='baseline',
            FOLDER_PATH:str=None,
            ANOMALY_DATASET_FOLDER:str=None,
            online:bool=False,
            index=0
        ):

        folder = None
        file_name = None

        if anomaly == False:
            folder = FOLDER_PATH
            file_name = 'trainset'
        else:
            folder = ANOMALY_DATASET_FOLDER
            file_name = 'validset'

        print(f'Fetching dataset from the ./Dataset/{folder}/{file_name}.csv')
        df = pd.read_csv(f'./Dataset/{folder}/{file_name}.csv')
        df.drop(columns=['Unnamed: 0'],inplace=True)
        df.reset_index(inplace=True,drop=True)

        ## Juan Gamella
        ## Ambient barometer is not affected by the wind
        ## tunnel actuators. In order to isolate effects 
        ## of tunnel variables, on intake, downwind, and 
        ## upwind pressure we subtract the ambient value 
        ## from them.

        df['pressure_intake'] -= df['pressure_ambient']
        df['pressure_upwind'] -= df['pressure_ambient']
        df['pressure_downwind'] -= df['pressure_ambient']

        temp_df = df['flag']

        df = df[[
            'current_mot',
            'current_out',
            'load_in',
            'rpm_in',
            'rpm_out',
            'load_out',
            'pressure_intake',
            'pressure_upwind',
            'pressure_downwind',
            'current_in',
            'mic',
            'current_supply'
        ]]

        df['rpm_in']  /= 1000.0 ## instead of minmax
        df['rpm_out'] /= 1000.0

        df_scaled  = None
        
        if type == 'baseline':
            
            df_scaled = np.array(df)
            
            # if anomaly == False:
                
            #     scaler = MinMaxScaler()
            #     df_scaled = scaler.fit_transform(df)
            #     with open(f'./scalers/{FOLDER_PATH}/scaler_b.pkl','wb') as f:
            #         pickle.dump(scaler, f)
            # else:
            #     scaler = MinMaxScaler()
            #     with open(f'./scalers/{FOLDER_PATH}/scaler_b.pkl','rb') as f:
            #         scaler = pickle.load(f)
                
            #     df_scaled = scaler.transform(df)
        else:
            if anomaly == False:

                scaler = MinMaxScaler(feature_range=(-1,1))
                df_scaled = scaler.fit_transform(df)
                with open('scaler_g.pkl','wb') as f:
                    pickle.dump(scaler, f)
            else:
                with open('scaler_g.pkl','rb') as f:
                    scaler = pickle.load(f)
                
                df_scaled = scaler.transform(df)

        data = torch.tensor(df_scaled.astype(np.float32),dtype=torch.float32)
        flag = torch.tensor(temp_df.values,dtype=torch.float32)

        array = []
        label = []
        i = 0
        while (i + MAX_LENGTH) <= data.shape[0]:
            array.append(data[i:i+MAX_LENGTH])
            label.append(flag[i:i+MAX_LENGTH])
            i += MAX_STRIDE
        
        self.array = torch.stack(array,dim=0) ## B,L,input_dim
        self.label = torch.stack(label,dim=0)

        if online:
            self.array = self.array[(index*BUFFER_SIZE):((index+1)*BUFFER_SIZE)]
            self.label = self.label[(index*BUFFER_SIZE):((index+1)*BUFFER_SIZE)]

    def __len__(self):
        return len(self.array)
    
    def __getitem__(self, index):
        return self.array[index],self.label[index]