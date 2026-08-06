import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import MSELoss
from torch.optim import Adam
from models.AE import AutoEncoder
from data import CausalChamberDataset

## --------- Hyper-parameters ---------
BATCH_SIZE = 32
EPOCHS = 100
ENC_NUM_LAYERS = 6
DEC_NUM_LAYERS = 6
LEARNING_RATE = 1e-3
EMBEDDING_DIMENSION = 256
NUM_HEADS = 4
INPUT_SIZE = 12
MAX_LENGTH = 100
MAX_STRIDE = 100
BOTTLENECK_DIM = 2
BETA = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## ------------------------------------

autoencoder_model = AutoEncoder(
    EMBEDDING_DIMENSION,
    NUM_HEADS,
    ENC_NUM_LAYERS,
    DEC_NUM_LAYERS,
    INPUT_SIZE,
    MAX_LENGTH,
    BOTTLENECK_DIM
)
autoencoder_model.to(DEVICE)

criterion = MSELoss()
optimizer = Adam(
    autoencoder_model.parameters(),
    lr=LEARNING_RATE
)
trainsets = CausalChamberDataset(
    anomaly=False,
    type='baseline',
    FOLDER_PATH='Folder5',
    ANOMALY_DATASET_FOLDER='Folder5'
)
trainload = DataLoader(trainsets,batch_size=BATCH_SIZE, shuffle=True)

# ## ------------ Training ------------
for epoch in range(EPOCHS):
    autoencoder_model.train()
    rolling_error = 0.0
    for image,label in trainload:
        optimizer.zero_grad()
        image = image.to(DEVICE)
        label = label.to(DEVICE)
        tae_output  = autoencoder_model(image)
        output      = tae_output['reconstructed_tensor']
        mu          = tae_output['mu']
        logvar      = tae_output['logvar']
        recon_loss = criterion(output,image)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2)-logvar.exp())
        kl_loss = torch.clamp(kl_loss,min=0.0)
        error = recon_loss ## + BETA*kl_loss
        error.backward()
        torch.nn.utils.clip_grad_norm_(autoencoder_model.parameters(),max_norm=1.0)
        optimizer.step()
        rolling_error += error.item()
    avg_error = rolling_error / len(trainload)
    
    print(f'Epoch: {epoch}/{EPOCHS}\tTraining Error: {avg_error:.4f}')
torch.save(autoencoder_model.state_dict(),'lstm_autoencoder.pth')
