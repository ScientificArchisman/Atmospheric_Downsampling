import os 
import numpy as np 
import time 
from tqdm import tqdm
import torch

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                num_epochs, device, log_folder, patience=5):
    # Move model to the specified device
    model.to(device)
    
    # Create directories for storing artifacts
    os.makedirs(log_folder, exist_ok=True)
    
    log_file = os.path.join(log_folder, 'logs.log')
    best_weights_file = os.path.join(log_folder, 'best_weights.pth')
    
    best_loss = float('inf')
    patience_counter = 0
    
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
                sr_images = model(lr_images)
                loss = criterion(sr_images, hr_images)
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
            
            train_loss = np.mean(train_losses)
            
            # Validation phase
            model.eval()
            val_losses = []
            with torch.no_grad():
                for hr_images, lr_images in val_loader:
                    hr_images, lr_images = hr_images.to(device), lr_images.to(device)
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
