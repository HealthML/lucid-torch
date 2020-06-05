import torch


class TFMSTrainingToUnitSpace(torch.nn.Module):
    def forward(self, data: torch.Tensor):
        return data.sigmoid()


class TFMSUnitToTrainingSpace(torch.nn.Module):
    def forward(self, data: torch.Tensor):
        data = data.clamp(0.000001, 0.99999)
        return torch.log(data / (1. - data))
