import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class ConvEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super(ConvEncoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer(x)

class ConvDecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(ConvDecoderLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.layer(x)

class Autoencoder(nn.Module):
    def __init__(self, encoder, decoder, original_shape):
        super(Autoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.original_shape = original_shape

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        decoded = F.interpolate(decoded, 
                                          size=self.original_shape[2:], 
                                          mode='bilinear', align_corners=False)
        return decoded

def add_gaussian_noise(inputs, mean=0.0, std=0.1):
    """ Add gaussian noise to the input data for denoising autoencoder """
    noise = torch.randn_like(inputs) * std + mean
    return inputs + noise


def pretrain_layer(encoder, decoder, data_loader, num_epochs=10, learning_rate=0.001, add_noise=False):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
    original_shape = next(iter(data_loader))[0].shape

    for epoch in range(num_epochs):
        total_loss = 0
        for data in data_loader:
            inputs = data[0]
            if add_noise:
                inputs = add_gaussian_noise(inputs)
            outputs = data[1]
            
            optimizer.zero_grad()
            latent = encoder(inputs)
            reconstructed = decoder(latent)

            reconstructed = F.interpolate(reconstructed, 
                                          size=original_shape[2:], 
                                          mode='bilinear', align_corners=False)
            
            loss = criterion(reconstructed, outputs)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(data_loader)}')
    
    return encoder, decoder



def pretrain_autoencoder(train_loader, num_epochs_per_layer, 
                         in_channels = 1,
                         channels_per_layer = [64, 128, 256], 
                         kernel_size=3, stride=2, padding=1, 
                         output_padding=0,  learning_rate=1e-3, add_noise=False):
    """ Pretrain the autoencoder layer by layer
    Args:
        train_loader: DataLoader object
        num_epochs_per_layer: Number of epochs to train each layer
        in_channels: Number of input channels
        channels_per_layer: List of number of channels in each layer
        kernel_size: Kernel size for convolutional layers
        stride: Stride for convolutional layers
        padding: Padding for convolutional layers
        output_padding: Output padding for transposed convolutional layers
        learning_rate: Learning rate for the optimizer
        add_noise: Boolean, whether to add noise to the input data
    Returns:
        encoder: Encoder model
        decoder: Decoder model"""
    n_layers = len(channels_per_layer)

    # Initailze encoder
    encoder_layer = ConvEncoderLayer(in_channels, channels_per_layer[0], 
                                    kernel_size, stride, padding)
    decoder_layer = ConvDecoderLayer(channels_per_layer[0], in_channels,
                                        kernel_size, stride, padding, 
                                        output_padding)
    encoder, decoder = nn.Sequential(encoder_layer), nn.Sequential(decoder_layer)


    print(f"************ Pretraining layer 1/{n_layers} ***************", end="\n\n")
    encoder, decoder = pretrain_layer(encoder, decoder,
                                    train_loader, learning_rate=learning_rate, 
                                    num_epochs=num_epochs_per_layer, add_noise=add_noise)
    
    for param in encoder.parameters():
        param.requires_grad = False
    for param in decoder.parameters():
        param.requires_grad = False

    for i in range(1, len(channels_per_layer)):
        print(f"*************** Pretraining layer {i+1}/{n_layers} ***************", end="\n\n")
        encoder_layer = ConvEncoderLayer(channels_per_layer[i-1], channels_per_layer[i],
                                        kernel_size, stride, padding)
        decoder_layer = ConvDecoderLayer(channels_per_layer[i], channels_per_layer[i-1],
                                        kernel_size, stride, padding,
                                        output_padding)
        new_encoder = nn.Sequential(encoder, encoder_layer)
        new_decoder = nn.Sequential(decoder_layer, decoder)

        encoder, decoder = pretrain_layer(new_encoder, new_decoder,
                                        train_loader, learning_rate=learning_rate,
                                        num_epochs=num_epochs_per_layer)

        for param in encoder.parameters():
            param.requires_grad = False
        for param in decoder.parameters():
            param.requires_grad = False

    return encoder, decoder


def check_shapes(encoder, decoder, train_loader):
    """ Check the shapes of the data, latent and reconstructed data
    to make sure the model is working correctly
    Args:
        encoder: Encoder model
        decoder: Decoder model
        train_loader: DataLoader object
    Returns:
        None(prints the shapes)"""
    data = next(iter(train_loader))[0]
    latent = encoder(data)
    reconstructed = decoder(latent)
    print(f"Data shape: {data.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")

def check_frozen_layers(model):
    """ Check which layers are frozen in the model
    Args:
        model: Model object
    Returns:
        None(prints the layers)"""
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss

def train_autoencoder(autoencoder, train_loader, val_loader, num_epochs=20, learning_rate=0.001, patience=7):
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    # Early stopping
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder.to(device)

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        autoencoder.train()
        train_loss = 0
        for data in train_loader:
            low_res, high_res = data
            low_res = low_res.to(device)
            high_res = high_res.to(device)
            
            # Forward pass
            output = autoencoder(low_res)
            loss = criterion(output, high_res)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()

        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                low_res, high_res = data
                low_res = low_res.to(device)
                high_res = high_res.to(device)
                output = autoencoder(low_res)
                
                loss = criterion(output, high_res)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        
        early_stopping(val_loss, autoencoder)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # Load the best model weights
    autoencoder.load_state_dict(torch.load('checkpoint.pt'))
    return autoencoder, train_losses, val_losses