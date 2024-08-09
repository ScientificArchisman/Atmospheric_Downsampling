import torch 
import torch.nn as nn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, use_activation: bool, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias = True)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if use_activation else nn.Identity()

    def forward(self, x):
        return self.activation(self.conv(x))
    
class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels = 32, beta: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta 
        self.conv = nn.ModuleList()

        for block_no in range(5):
            self.conv.append(ConvBlock(in_channels + channels * block_no, 
                                       channels if block_no < 4 else in_channels,
                                         use_activation=True if block_no < 4 else False,
                                           kernel_size=3, stride=1, padding=1))
            
            
    def forward(self, x):
        new_inputs = x
        for block in self.conv:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.beta * out + x
    

class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(*[DenseResidualBlock(in_channels, beta=residual_beta) for _ in range(3)])

    def forward(self, x):
        return self.rrdb(x) * self.residual_beta + x
    

class ModifiedSRCNN(nn.Module):
    def __init__(self, in_channels: int, num_blocks: int, 
                 n1: int, n2: int, f1: int, f2: int, f3: int,
                 *args, **kwargs) -> None:
        """ Initialize the SRCNN with Dense Residual network model with the required layers 
         Below params are the hyperparameters for the SRCNN model without the 
         Bassic block which has been added extra other than the resisual connections.
        in_channels (int): Input number of channels
        num_blocks (int): Number of RRDB blocks
        n1 (int): Number of filters in the first convolutional layer
        n2 (int): Number of filters in the second convolutional layer
        f1 (int): Kernel size of the first convolutional layer
        f2 (int): Kernel size of the second convolutional layer
        f3 (int): Kernel size of the third convolutional layer
        residual_beta (float): Residual connection weight
        """
        super().__init__(*args, **kwargs)
        self.conv1 = ConvBlock(in_channels, n1, kernel_size=f1, stride=1, padding=(1, 1, 1), use_activation=True)
        self.bn1 = nn.BatchNorm3d(n1)
        self.blocks = nn.Sequential(*[RRDB(n1 + in_channels, residual_beta=0.5) for _ in range(num_blocks)])
        self.bn2 = nn.BatchNorm3d((n1 + in_channels))
        self.conv2 = ConvBlock(2 * (n1 + in_channels), n2, kernel_size=f2, stride=1, padding=(1, 1, 1), use_activation=True)
        self.bn3 = nn.BatchNorm3d(n2)
        self.conv3 = ConvBlock(n2 + n1 + in_channels, in_channels, kernel_size=f3, stride=1, padding=1, use_activation=False)
        self.bn4 = nn.BatchNorm3d(in_channels)
        self._initialize_weights()

    def forward(self, x):
        
        initial = x 
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.concat([x, initial], dim = 1)
        initial = x
        x = self.blocks(x)
        x = self.bn2(x)
        x = torch.concat([x, initial], dim = 1)
        x = self.conv2(x) # Take feature maps here
        x = self.bn3(x)
        x = torch.concat([x, initial], dim=1)
        x = self.conv3(x)
        x = self.bn4(x)
        return x 
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

if __name__ == "__main__":
    random_tensor = torch.rand(1, 7, 50, 20, 20, dtype=torch.float16).to(DEVICE)
    in_channels = 7
    num_blocks = 2
    n1 = 32
    n2 = 128
    f1 = 3
    f2 = 3
    f3 = 3
    srcnn = ModifiedSRCNN(in_channels = in_channels, num_blocks = num_blocks, n1 = n1, n2 = n2, f1 = f1, f2 = f2, f3 = f3).to(DEVICE).half()
    output = srcnn(random_tensor)
    print(output.shape)