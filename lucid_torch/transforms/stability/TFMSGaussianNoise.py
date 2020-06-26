import torch


class TFMSGaussianNoise(torch.nn.Module):
    def __init__(self, level=0.01):
        super(TFMSGaussianNoise, self).__init__()
        self.level = level

    def forward(self, img):
        return img + self.level * torch.randn_like(img)
