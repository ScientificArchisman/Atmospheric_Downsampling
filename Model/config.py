import torch

NUM_EPOCHS: int = 1
PATIENCE: int = 15
BATCH_SIZE: int= 16
LOG_FOLDER: str = "Logs"
TENSORBOARD_LOGS: str = "runs/SRCNN"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_STR = "cuda" if torch.cuda.is_available() else "cpu"

######### Dataloader hyperparameters ########
LATITUDE_CHUNK_SIZE = 13
LONGITUDE_CHUNK_SIZE = 13

######### MODEL HYPERPARAMETERS #########
in_channels = 7
num_blocks = 1
n1 = 64
n2 = 128
f1 = 3
f2 = 3
f3 = 3
LEARNING_RATE: int = 3e-3
WEIGHT_DECAY = 1e-2