from torch import nn 
import torch
from torchvision.models import vgg19, VGG19_Weights
import os
from src.utils import detect_environment
device = detect_environment()

pretrained_weights_folder = "statics/pretrained"
os.makedirs(pretrained_weights_folder, exist_ok=True)
os.environ['TORCH_HOME'] = pretrained_weights_folder



class VGGLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vgg = vgg19(weights = VGG19_Weights.IMAGENET1K_V1).features[:35].eval().to(device)

        for param in self.vgg.parameters():
            param.requires_grad = False

        self.loss = nn.L1Loss()

    def preprocess(self, x):
        """
        Ensure the input tensor has exactly 3 channels.
        """
        if x.size(1) > 3:
            # Reduce to 3 channels using a 1x1 convolution
            conv = nn.Conv2d(x.size(1), 3, kernel_size=1, stride=1, bias=False).to(x.device)
            x = conv(x)
        elif x.size(1) < 3:
            # Pad with zeros to get 3 channels
            padding = (0, 0, 0, 0, 0, 3 - x.size(1))
            x = nn.functional.pad(x, padding, mode="constant", value=0)
        return x

    def forward(self, x, y):
        # Preprocess inputs to ensure they have 3 channels
        x = self.preprocess(x)
        y = self.preprocess(y)

        vgg_x = self.vgg(x)
        vgg_y = self.vgg(y)
        return self.loss(vgg_x, vgg_y)
    


if __name__ == "__main__":
    x = torch.randn(1, 6, 1024, 1024).to(device)
    y = torch.randn(1, 6, 1024, 1024).to(device)

    print(f"Device: {device}")

    loss_metric = VGGLoss()

    loss = loss_metric(x, y)
    print(f"Loss: {loss}")


