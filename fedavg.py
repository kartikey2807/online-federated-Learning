import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from models.AE import AutoEncoder
from config import *
from torch.optim import Adam
from data import CausalChamberDataset
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score,recall_score,f1_score,precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

def trainer(model,FOLDER_PATH:str,ANOMALY_DATASET_FOLDER:str,online:bool=False,index=0):
    criterion = nn.MSELoss()
    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    trainsets = CausalChamberDataset(
        anomaly=False,
        type='baseline',
        FOLDER_PATH=FOLDER_PATH,
        ANOMALY_DATASET_FOLDER=ANOMALY_DATASET_FOLDER,
        online=online,
        index=index
    )
    validsets = CausalChamberDataset(
        anomaly=True,
        type='baseline',
        FOLDER_PATH=FOLDER_PATH,
        ANOMALY_DATASET_FOLDER=ANOMALY_DATASET_FOLDER
    )
    trainload = DataLoader(trainsets,BATCH_SIZE,shuffle=True)
    validload = DataLoader(validsets,BATCH_SIZE)
    
    log_train_error = []
    for epoch in range(EPOCHS):

        model.train()
        rolling_train_error = []
        for image,label in trainload:
            optimizer.zero_grad()

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            tae_output  = model(image)
            output      = tae_output['reconstructed_tensor']
            mu          = tae_output['mu']
            logvar      = tae_output['logvar']
            
            recon_loss = criterion(output,image)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2)-logvar.exp())
            kl_loss = torch.clamp(kl_loss,min=0.0)
            error = recon_loss + BETA*kl_loss
            error.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            rolling_train_error.append(error.item())
        
        train_error = sum(er for er in rolling_train_error) / len(trainload)

        model.eval()
        all_tr_errors = []
        with torch.no_grad():
            for image,label in trainload:
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                reconstruction = model(image)['reconstructed_tensor']
                mask_error = (reconstruction - image)**2
                mask_error_per_step = mask_error.mean(dim=-1)
                
                all_tr_errors.append(
                    mask_error_per_step.cpu().numpy()
                )
        
        tr_errors = np.concatenate(all_tr_errors).flatten()
        average_train_error = tr_errors.mean()
        std_train_error = tr_errors.std()

        log_train_error.append(average_train_error)
        threshold = average_train_error + 3*std_train_error

        # all_ts_errors = []
        # all_ts_labels = []
        # with torch.no_grad():
        #     for image,label in validload:
        #         image = image.to(DEVICE)
        #         label = label.to(DEVICE)

        #         rec = model(image)['reconstructed_tensor']
        #         test_error = (rec - image)**2
        #         test_error_per_step = test_error.mean(dim=-1)
        
        #         all_ts_errors.append(
        #             test_error_per_step.cpu().numpy()
        #         )
        #         all_ts_labels.append(
        #             label.cpu().numpy()
        #         )
        
        # ts_errors = np.concatenate(all_ts_errors).flatten()
        # ts_labels = np.concatenate(all_ts_labels).flatten()
        # predictions = (ts_errors > threshold).astype(float)
        # precision = precision_score(ts_labels, predictions)
        # recall = recall_score(ts_labels,predictions)
        # f1 = f1_score(ts_labels,predictions)

        ## ------------------- Log stats --------------------
        print(f'Epoch: {epoch}\tTraining error: {train_error:.4f}')
            #   Training error: {train_error:.4f}\t
            #   Threshold: {threshold:.4f}\t
            #   Precision: {precision:.4f}\t
            #   Recall: {recall:.4f}\t
            #   F1 score: {f1:.4f}''')
                
    return model,threshold,log_train_error

for trial in range(1):

    ## ----------------- Initialize Global Model ----------------
    global_vae_model = AutoEncoder(
        EMBEDDING_DIMENSION,
        NUM_HEADS,
        ENC_NUM_LAYERS,
        DEC_NUM_LAYERS,
        INPUT_SIZE,
        MAX_LENGTH,
        BOTTLENECK_DIM
    )
    global_vae_model.to(DEVICE)
    train_error_clients = {
        'Folder1':[],
        'Folder2':[],
        'Folder3':[],
        'Folder4':[]
    }

    master_df = pd.DataFrame({
        'Validation set':[],
        'Precision':[],
        'Recall':[],
        'F1 score':[],
        'TP':[],'FP':[],'FN':[]
    })
    for round in range(GLOBAL_ROUNDS):
        print(f'Round:{round}/{GLOBAL_ROUNDS}')

        train_client_array = []
        threshold_dict = {}
        for client in ['Folder1','Folder2','Folder3','Folder4']:
            print(f'Training on {client}')
            temp_model,threshold,log_train_error = \
                trainer(
                    global_vae_model,
                    client,
                    client,
                    online=False,
                    index=round
                )
            train_client_array.append(temp_model)
            threshold_dict[client] = threshold
            train_error_clients[client].extend(log_train_error)

        ## federared averaging
        global_state_dict = global_vae_model.state_dict()
        for key in global_state_dict.keys():
            global_state_dict[key] = torch.stack([client.state_dict()[key].float() for client in train_client_array],dim=0).mean(dim=0)
        global_vae_model.load_state_dict(global_state_dict)


        record_dict = {
            'Validation set':[],
            'Precision':[],
            'Recall':[],
            'F1 score':[],
            'TP':[],'FP':[],'FN':[]
        }
        for folder in ['Folder1','Folder2','Folder3','Folder4']:

            validsets = CausalChamberDataset(
                anomaly=True,
                type='baseline',
                FOLDER_PATH=folder,
                ANOMALY_DATASET_FOLDER=folder
            )
            validload = DataLoader(validsets,BATCH_SIZE)

            all_ts_errors = []
            all_ts_labels = []
            global_vae_model.eval()
            with torch.no_grad():
                
                for image,label in validload:
                    image = image.to(DEVICE)
                    label = label.to(DEVICE)

                    outputs = global_vae_model(image)['reconstructed_tensor']
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
            threshold = threshold_dict[folder]
            predictions = (ts_errors > threshold).astype(float)
            precision = precision_score(ts_labels, predictions)
            recall = recall_score(ts_labels,predictions)
            f1 = f1_score(ts_labels,predictions)
            tp = ((predictions == 1.0)& (ts_labels==1.0)).sum()
            fp = ((predictions == 1.0)& (ts_labels==0.0)).sum()
            fn = ((predictions == 0.0)& (ts_labels==1.0)).sum()

            record_dict['Validation set'].append(f'Client data {folder[-1]}')
            record_dict['Precision'].append(precision)
            record_dict['Recall'].append(recall)
            record_dict['F1 score'].append(f1)
            record_dict['TP'].append(tp)
            record_dict['FP'].append(fp)
            record_dict['FN'].append(fn)
        
        df = pd.DataFrame(record_dict)
        print(df)
        master_df = df.copy()

    fig,axes = plt.subplots(2,2,figsize=(15,15))
    # for idx,folder in enumerate(['Folder1', 'Folder2', 'Folder3', 'Folder4']):
    #     axes[idx].plot(train_error_clients[folder])
    #     axes[idx].set_title(f'Client {folder[-1]}',fontsize=25)
    #     axes[idx].set_xlabel('Epochs',fontsize=20)
    #     axes[idx].set_ylabel('Training error',fontsize=20)
    folder = 'Folder1'
    axes[0,0].plot(train_error_clients[folder])
    axes[0,0].set_title(f'Client {folder[-1]}',fontsize=25)
    axes[0,0].set_xlabel('Epochs',fontsize=20)
    axes[0,0].set_ylabel('Training error',fontsize=20)
    folder = 'Folder2'
    axes[0,1].plot(train_error_clients[folder])
    axes[0,1].set_title(f'Client {folder[-1]}',fontsize=25)
    axes[0,1].set_xlabel('Epochs',fontsize=20)
    axes[0,1].set_ylabel('Training error',fontsize=20)
    folder = 'Folder3'
    axes[1,0].plot(train_error_clients[folder])
    axes[1,0].set_title(f'Client {folder[-1]}',fontsize=25)
    axes[1,0].set_xlabel('Epochs',fontsize=20)
    axes[1,0].set_ylabel('Training error',fontsize=20)
    folder = 'Folder4'
    axes[1,1].plot(train_error_clients[folder])
    axes[1,1].set_title(f'Client {folder[-1]}',fontsize=25)
    axes[1,1].set_xlabel('Epochs',fontsize=20)
    axes[1,1].set_ylabel('Training error',fontsize=20)

    plt.tight_layout()
    plt.savefig('FedAvg_loss_trends.png',dpi=300)
    print(master_df)
    master_df.to_csv(f'FedAvg_results_{BUFFER_SIZE}_{trial}.csv',index=False)