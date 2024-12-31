import torch
from torch import nn
from src.utils import detect_environment

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int, activation=True, 
                 kernel_size:int = 3, padding:int = 1, stride:int = 1, **kwargs):
        super().__init__(**kwargs)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, stride=stride, bias=True, **kwargs)
        self.activation = nn.LeakyReLU(0.2, inplace=True) if activation else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels:int, scale_factor:int, **kwargs):
        super().__init__(**kwargs)

        # upsampling with neighboor interpolation, but we can use bilinear interpolation
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=True)
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.activation(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int = 32, residual_beta: float = 0.2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.residual_beta = residual_beta # scales the output of the dense block 
        self.conv_layers = nn.ModuleList()

        for i in range(5): # 5 conv layers
            self.conv_layers.append(
                ConvBlock(in_channels = in_channels + channels * i, 
                out_channels = channels if i < 4 else in_channels, 
                activation=True if i < 4 else False,
                kernel_size=3, padding=1, stride=1))


    def forward(self, x):
        inputs = x
        for layer in self.conv_layers:
            out = layer(inputs)
            inputs = torch.cat([inputs, out], dim=1) # last concat is unnecessary
        return out * self.residual_beta + x # scale the outputs of the dense block 


class RRDB(nn.Module):
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(in_channels) for _ in range(3)
        ])

    def forward(self, x):
        inputs = x
        for block in self.residual_blocks:
            x = block(x)
        return 0.2 * x + inputs
    


class Generator(nn.Module):
    def __init__(self, in_channels: int, channels: int=64, scale_factor:int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        1. Convolutional layer
        2. 23 Residual in Residual Dense Blocks
        3. Convolutional layer
        4. Upsampling
        5. Convolutional layer
        6. Convolutional layer

        conv1_out + conv2_out
        '''

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=channels, activation=False,
                               kernel_size=3, padding=1, stride=1)
        
        self.res_blocks = nn.Sequential(*[
            RRDB(in_channels=channels) for _ in range(3)])
        
        self.conv2 = ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, 
                               padding=1, stride=1)
        self.upsample = nn.Sequential(*[        
            UpsampleBlock(in_channels=channels, scale_factor=scale_factor) for _ in range(2)])
        
        self.out_conv_layers = nn.Sequential(*[
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, stride=1), 
            ConvBlock(in_channels=channels, out_channels=in_channels, activation=False,
                      kernel_size=3, padding=1, stride=1)])
        
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        residual = x
        x = self.res_blocks(x)
        x = self.conv2(x)
        x = x + residual
        x = self.upsample(x)
        x = self.out_conv_layers(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()

        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(ConvBlock(in_channels, feature, kernel_size=3, stride=1 + idx % 2, 
                                    padding=1, activation=True))
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

if __name__ == "__main__":
    device = detect_environment()
    print(f"Device: {device}")

    x = torch.randn(2, 3, 256, 256).to(device)
    generator = Generator(in_channels=3, channels=64).to(device)
    output = generator(x)

    print(f"Generator Output shape: {output.shape}")

    # discrim = Discriminator(in_channels=3)
    # output = discrim(output.cpu())

    # print(f"Discriminator Output shape: {output.shape}")

