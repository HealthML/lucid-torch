import kornia
import torch


class TFMSNormalize(kornia.augmentation.Normalize):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(TFMSNormalize, self).__init__(
            torch.tensor(mean),
            torch.tensor(std))
