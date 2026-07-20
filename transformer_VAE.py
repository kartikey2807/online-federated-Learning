import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from config import *
from train import train
from sklearn.metrics import precision_score,recall_score,f1_score,precision_recall_curve
from torch.utils.data import DataLoader
from data import CausalChamberDataset
from models.AE import AutoEncoder

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

for _, folder in enumerate(['Folder1','Folder2','Folder3', 'Folder4', 'Folder5']):
    record_dict = {
        'Validation set':[],
        'Precision':[],
        'Recall':[],
        'F1-score':[],
        'TP':[],'FP':[],'FN':[]
    }

    for index in range(1):

        print(f'Training Transformer-VAE on the ./Dataset/{folder}/trainset.csv')
        train(FOLDER_PATH=folder,ANOMALY_DATASET_FOLDER=folder,model_index=index)

        learned_transformerAE = AutoEncoder(
            EMBEDDING_DIMENSION,
            NUM_HEADS,
            ENC_NUM_LAYERS,
            DEC_NUM_LAYERS,
            INPUT_SIZE,
            MAX_LENGTH,
            BOTTLENECK_DIM
        )
        learned_transformerAE.load_state_dict(torch.load(f'./saved_models/{folder}/vae_autoencoder_model_{index}.pt',weights_only=True))
        learned_transformerAE.to(DEVICE)

        threshold = None
        with open(f'./results/{folder}/threshold.txt','r') as f: # train threshold
            threshold = float(f.read().strip())

        for test_folder in ['Folder1','Folder2','Folder3','Folder4']:
            print(f'Evaluating./Dataset/{test_folder}/validset.csv')

            validsets = CausalChamberDataset(
                anomaly=True,
                type='baseline',
                FOLDER_PATH=folder,
                ANOMALY_DATASET_FOLDER=test_folder
            )
            validload = DataLoader(validsets,BATCH_SIZE)
            
            all_ts_errors = []
            all_ts_labels = []
            learned_transformerAE.eval()
            with torch.no_grad():
                
                for image,label in validload:
                    image = image.to(DEVICE)
                    label = label.to(DEVICE)

                    outputs = learned_transformerAE(image)['reconstructed_tensor']
                    error = (outputs - image)**2
                    error_per_timestep = error.mean(dim=-1)

                    all_ts_errors.append(
                        error_per_timestep.cpu().numpy()
                    )
                    all_ts_labels.append(
                        label.cpu().numpy()
                    )
            
            ts_errors = np.concatenate(all_ts_errors).flatten()
            ts_labels = np.concatenate(all_ts_labels).flatten()
            predictions = (ts_errors > threshold).astype(float)
            precision = precision_score(ts_labels, predictions)
            recall = recall_score(ts_labels,predictions)
            f1 = f1_score(ts_labels,predictions)
            tp = ((predictions == 1.0)& (ts_labels==1.0)).sum()
            fp = ((predictions == 1.0)& (ts_labels==0.0)).sum()
            fn = ((predictions == 0.0)& (ts_labels==1.0)).sum()

            ## compute AUC for precision-recall curve
            record_dict['Validation set'].append(f'Client data {test_folder[-1]}')
            record_dict['Precision'].append(precision)
            record_dict['Recall'].append(recall)
            record_dict['F1-score'].append(f1)
            record_dict['TP'].append(tp)
            record_dict['FP'].append(fp)
            record_dict['FN'].append(fn)

    pr_auc_dataframe = pd.DataFrame(record_dict)
    print(pr_auc_dataframe)
    pr_auc_dataframe.to_csv(f'./results/{folder}/results.csv')
    # pr_auc_barplot(pr_auc_dataframe,folder)