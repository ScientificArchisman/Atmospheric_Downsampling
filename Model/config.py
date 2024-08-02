import torch

NUM_EPOCHS: int = 1
PATIENCE: int = 15
BATCH_SIZE: int= 16
LOG_FOLDER: str = "Logs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######### Dataloader hyperparameters ########
LATITUDE_CHUNK_SIZE = 16
LONGITUDE_CHUNK_SIZE = 16

######### MODEL HYPERPARAMETERS #########
in_channels = 7
num_blocks = 2
n1 = 32
n2 = 128
f1 = 3
f2 = 3
f3 = 3
LEARNING_RATE: int = 3e-4