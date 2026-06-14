import torch
import numpy as np
import torch.nn as nn

from sklearn.metrics import precision_score,recall_score,f1_score,precision_recall_curve
from config import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from models.AE import AutoEncoder
from data import CausalChamberDataset

def train(FOLDER_PATH:str,ANOMALY_DATASET_FOLDER:str, model_index:int=None):
    '''
    Train function to output reconstructed
    image and compute error. We use metric
    AUC under the precision & recall curve
    '''
    transformer_autoencoder = AutoEncoder(
        EMBEDDING_DIMENSION,
        NUM_HEADS,
        ENC_NUM_LAYERS,
        DEC_NUM_LAYERS,
        INPUT_SIZE,
        MAX_LENGTH,
        BOTTLENECK_DIM
    )
    transformer_autoencoder.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = Adam(transformer_autoencoder.parameters(),lr=LEARNING_RATE)

    trainsets = CausalChamberDataset(
        anomaly=False,
        type='baseline',
        FOLDER_PATH=FOLDER_PATH,
        ANOMALY_DATASET_FOLDER=ANOMALY_DATASET_FOLDER
    )
    validsets = CausalChamberDataset(
        anomaly=True,
        type='baseline',
        FOLDER_PATH=FOLDER_PATH,
        ANOMALY_DATASET_FOLDER=ANOMALY_DATASET_FOLDER
    )
    trainload = DataLoader(trainsets,BATCH_SIZE,shuffle=True)
    validload = DataLoader(validsets,BATCH_SIZE)

    MIN_METRIC = 1000
    FINAL_THRESHOLD = None
    for epoch in range(EPOCHS):

        ## TRAIN
        transformer_autoencoder.train()
        rolling_train_error = []

        for idx, (image, label) in enumerate(trainload):
            optimizer.zero_grad()
            image = image.to(DEVICE)
            label = label.to(DEVICE)

            tae_output  = transformer_autoencoder(image)
            output      = tae_output['reconstructed_tensor']
            mu          = tae_output['mu']
            logvar      = tae_output['logvar']
            
            recon_loss = criterion(output,image)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2)-logvar.exp())
            kl_loss = torch.clamp(kl_loss,min=0.0)
            error = recon_loss + BETA*kl_loss
            error.backward()
            torch.nn.utils.clip_grad_norm_(transformer_autoencoder.parameters(),max_norm=1.0)
            optimizer.step()
            rolling_train_error.append(error.item())
        
        train_error = sum(er for er in rolling_train_error) / len(trainload)

        ## EVAL
        transformer_autoencoder.eval()
        all_tr_errors = []
        with torch.no_grad():
            for image,label in trainload:
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                rec = transformer_autoencoder(image)['reconstructed_tensor']
                mask_error = (rec - image)**2
                mask_error_per_step = mask_error.mean(dim=-1)

                all_tr_errors.append(
                    mask_error_per_step.cpu().numpy()
                )
        
        tr_errors = np.concatenate(all_tr_errors).flatten()
        average_train_error = tr_errors.mean()
        std_train_error = tr_errors.std()

        threshold = average_train_error + 3*std_train_error

        all_ts_errors = []
        all_ts_labels = []
        with torch.no_grad():
            for image,label in validload:
                image = image.to(DEVICE)
                label = label.to(DEVICE)

                rec = transformer_autoencoder(image)['reconstructed_tensor']
                test_error = (rec - image)**2
                test_error_per_step = test_error.mean(dim=-1)
        
                all_ts_errors.append(
                    test_error_per_step.cpu().numpy()
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

        ## compute AUC for precision-recall curve
        print(f'Epoch: {epoch}\tTraining error: {train_error:.4f}\tThreshold: {threshold:.4f}\tPrecision: {precision:.4f}\tRecall: {recall:.4f}\tF1 score: {f1:.4f}')

        if average_train_error <= MIN_METRIC:
            MIN_METRIC = average_train_error
            torch.save(
                transformer_autoencoder.state_dict(),
                f'./saved_models/{FOLDER_PATH}/vae_autoencoder_model_{model_index}.pt'
            )
            FINAL_THRESHOLD = threshold
            print('-- MODEL UPDATE --')

    with open(f'./results/{FOLDER_PATH}/threshold.txt','w') as f: ## could been in IF
        f.write(f'{FINAL_THRESHOLD:.4f}')