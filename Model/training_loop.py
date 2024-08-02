import os 
import numpy as np 
import time 
from tqdm import tqdm
import torch
from mod_srcnn import ModifiedSRCNN
import config
from data_loading import WRFDataset, create_loaders
from torch.cuda.amp import autocast, GradScaler
import xarray as xr

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, log_folder, patience=5):
    """ Train the model using the specified data loaders and hyperparameters.
    Saves the best model weights based on the validation loss.
    Args:
        model (torch.nn.Module): Model to be trained
        train_loader (torch.utils.data.DataLoader): Training data loader
        val_loader (torch.utils.data.DataLoader): Validation data loader
        criterion (torch.nn.Module): Loss function
        optimizer (torch.optim.Optimizer): Optimizer
        num_epochs (int): Number of epochs to train the model
        device (torch.device): Device to run the model on
        log_folder (str): Folder to store logs and model weights
        patience (int): Number of epochs to wait before early stopping
    Returns:
        torch.nn.Module: Trained model
    """
    # Move model to the specified device
    model.to(device)
    
    # Create directories for storing artifacts
    os.makedirs(log_folder, exist_ok=True)
    
    log_file = os.path.join(log_folder, 'logs.log')
    best_weights_file = os.path.join(log_folder, 'best_weights.pth')
    
    best_loss = float('inf')
    patience_counter = 0
    
    scaler = GradScaler()
    
    with open(log_file, 'w') as log:
        log.write('Epoch,Train Loss,Val Loss,Epoch Time\n')
        
        for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
            start_time = time.time()
            
            # Training phase
            model.train()
            train_losses = []
            for hr_images, lr_images in train_loader:
                hr_images, lr_images = hr_images.to(device), lr_images.to(device)
                optimizer.zero_grad()
                
                # Enable autocast context for mixed precision training
                with autocast():
                    sr_images = model(lr_images)
                    loss = criterion(sr_images, hr_images)
                
                # Scale the loss and backward pass
                scaler.scale(loss).backward()
                
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
                scaler.step(optimizer)
                scaler.update()
                
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for hr_images, lr_images in val_loader:
                    hr_images, lr_images = hr_images.to(device), lr_images.to(device)
                    with autocast():
                        sr_images = model(lr_images)
                        loss = criterion(sr_images, hr_images)
                    val_losses.append(loss.item())
            
            val_loss = np.mean(val_losses)
            
            epoch_time = time.time() - start_time
            
            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Epoch Time: {epoch_time:.2f}s')
            log.write(f'{epoch+1},{train_loss},{val_loss},{epoch_time}\n')
            
            # Check for best validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), best_weights_file)
            else:
                patience_counter += 1
                        
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    return model


if __name__ == "__main__":
    model_srcnn = ModifiedSRCNN(in_channels=config.in_channels, num_blocks=config.num_blocks, 
                          n1=config.n1, n2=config.n2, f1=config.f1, f2=config.f2, f3=config.f3)
    
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
    

    dataset = WRFDataset(dataset, dataset, config.LATITUDE_CHUNK_SIZE, config.LONGITUDE_CHUNK_SIZE)
    train_loader, valid_loader, test_loader = create_loaders(dataset, config.BATCH_SIZE)

    # TRAINING PART
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model_srcnn.parameters(), lr=config.LEARNING_RATE)
    train_model(model = model_srcnn, train_loader=train_loader, val_loader=valid_loader, 
                criterion=criterion, optimizer=optimizer, num_epochs=config.NUM_EPOCHS, 
                log_folder=config.LOG_FOLDER, device=config.DEVICE, patience=config.PATIENCE)



