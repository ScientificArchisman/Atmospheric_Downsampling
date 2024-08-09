import os 
import numpy as np 
import time 
from tqdm import tqdm
import torch
from mod_srcnn import ModifiedSRCNN
import config
from data_loading import WRFDataset, create_loaders, MinMaxScaleTransform
from torch.cuda.amp import autocast, GradScaler
import xarray as xr
from training_loop import train_model


print(f"Training started...Training on {config.DEVICE}")

## Load the Data
# LOAD the Data and cretae data loaders
ozone_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_2011.nc")
co_no2_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_CO_NO2.nc")
no_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_NO.nc")
humidity_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_Archi_2011_SpecificHum.nc")
temp_2011 = xr.open_dataset("/Volumes/Extreme SSD/PRL/data/high_res/WRF_2011_Archi_T.nc")

PRESSURE_LEVEL = 2
dataset = np.array([ozone_2011["o3"].sel(bottom_top=PRESSURE_LEVEL), 
            ozone_2011["PM2_5_DRY"].sel(bottom_top=PRESSURE_LEVEL),
            co_no2_2011["co"].sel(bottom_top=PRESSURE_LEVEL), 
            co_no2_2011["no2"].sel(bottom_top=PRESSURE_LEVEL), 
            no_2011["no"].sel(bottom_top=PRESSURE_LEVEL), 
            humidity_2011["QVAPOR"].sel(bottom_top=PRESSURE_LEVEL),
            temp_2011["T2"]])

min_max_transform = MinMaxScaleTransform(dataset, dataset, use_half=True)
dataset = WRFDataset(dataset, dataset, config.LATITUDE_CHUNK_SIZE, 
                         config.LONGITUDE_CHUNK_SIZE, transform=min_max_transform)
train_loader, valid_loader, test_loader = create_loaders(dataset, config.BATCH_SIZE)

## Train the Model
model_srcnn = ModifiedSRCNN(in_channels=config.in_channels, num_blocks=config.num_blocks, 
                          n1=config.n1, n2=config.n2, f1=config.f1, f2=config.f2, f3=config.f3)
    
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model_srcnn.parameters(), lr=config.LEARNING_RATE, 
                             weight_decay=config.WEIGHT_DECAY)

train_model(model = model_srcnn, train_loader=train_loader, val_loader=valid_loader, 
                criterion=criterion, optimizer=optimizer, num_epochs=config.NUM_EPOCHS, 
                log_folder=config.LOG_FOLDER, device=config.DEVICE, patience=config.PATIENCE)

