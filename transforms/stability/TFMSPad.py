import torch


class TFMSPad(torch.nn.Module):
    def __init__(self, w, mode='constant', constant_value=0.5):
        super(TFMSPad, self).__init__()
        self.w = w
        self.mode = mode
        self.constant_value = constant_value

    def forward(self, img):
        if self.constant_value == 'uniform':
            val = torch.rand(1)
        else:
            val = self.constant_value
        pad = torch.nn.ConstantPad2d(self.w, val)
        return pad(img)
