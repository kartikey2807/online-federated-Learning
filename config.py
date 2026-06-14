import torch

## ---------- Hyper-parameters ----------
BATCH_SIZE = 64           ## [local: 64]
EPOCHS = 15               ## [local: 400]
ENC_NUM_LAYERS = 5        ## [local: 5]
DEC_NUM_LAYERS = 5        ## [local: 5]
LEARNING_RATE = 1e-3

EMBEDDING_DIMENSION = 256
NUM_HEADS = 4
INPUT_SIZE = 12
MAX_LENGTH = 100
MAX_STRIDE = 100
BOTTLENECK_DIM = 32
GLOBAL_ROUNDS = 60
BUFFER_SIZE = None

BETA = 1.0
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')