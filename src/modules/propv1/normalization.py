import torch
import torch.nn as nn


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.fc = nn.Linear(condition_dim, channels * 2)
        self.fc.weight.data[:, :channels] = 1.0
        self.fc.weight.data[:, channels:] = 0.0

    def forward(self, x, cond):
        _, c, _, _ = x.size()

        cond = self.fc(cond)
        gamma = cond[:, :c].unsqueeze(2).unsqueeze(3)
        beta = cond[:, c:].unsqueeze(2).unsqueeze(3)

        x = gamma * self.norm(x) + beta
        return x


class BatchNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.BatchNorm2d(channels, affine=False)
        self.weight = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, channels, 1, 1)))

    def forward(self, x):
        x = self.weight * self.norm(x) + self.bias
        return x


class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            channels, affine=False, track_running_stats=False)
        self.fc = nn.Linear(condition_dim, channels * 2)
        self.fc.weight.data[:, :channels] = 1.0
        self.fc.weight.data[:, channels:] = 0.0

    def forward(self, x, cond):
        _, c, _, _ = x.size()

        cond = self.fc(cond)
        gamma = cond[:, :c].unsqueeze(2).unsqueeze(3)
        beta = cond[:, c:].unsqueeze(2).unsqueeze(3)

        x = gamma * self.norm(x) + beta
        return x


class InstanceNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(
            channels, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.norm(x)
        return x
