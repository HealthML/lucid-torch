import torch


class TFMSTrainingToUnitSpace(torch.nn.Module):
    def forward(self, data: torch.Tensor):
        return data.sigmoid()
