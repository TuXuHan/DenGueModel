import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(self.mse(yhat, y))


class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.mae(yhat, y)


class HuberLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.huber = nn.SmoothL1Loss()

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.huber(yhat, y)
