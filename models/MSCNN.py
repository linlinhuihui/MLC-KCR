import torch
import torch.nn as nn
import torch.nn.functional as F


class MSCNN(nn.Module):
    def __init__(self, in_channels, out_channels=128, kernel_sizes=(3,5,7), dropout=0.5):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels, k, padding=k//2)
            for k in kernel_sizes
        ])
        self.proj = nn.Conv1d(out_channels * len(kernel_sizes), out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        conv_outs = [F.relu(conv(x)) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        out = self.proj(out)
        return out



