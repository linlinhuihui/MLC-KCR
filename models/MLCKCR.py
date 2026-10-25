from typing import Dict, Any
import torch
import torch.nn as nn
from models.MLPs import MLP
from models.CBAM import *
from models.MSCNN import *
import yaml
with open(r"E:\git-kcr\config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

class  classifier(nn.Module):
    def __init__(
        self,
        in_channels,out_channels2,hidden_size,
        num_linears=[64],
        dropout_linears=[0.2],
        seqlen=33,
    ) -> None:
        super().__init__()
        self.TextCNN = MSCNN(in_channels, out_channels2)
        self.lstm = nn.LSTM(
            input_size= out_channels2,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        self.CBAM = CBAM1D(hidden_size*2)
        input_linear = hidden_size*2 *29
        self.mlp = MLP(input_linear, num_linears, dropout_linears, acti="hardswish")

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.TextCNN(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = x.permute(0, 2, 1)
        x = self.CBAM(x)
        x = torch.flatten(x, start_dim=1)
        return self.mlp(x)

    @staticmethod
    def get_model_params():
        model_params = dict(
            in_channels=config["in_channels"],
            num_linears=config["num_linears"],
            seqlen=config["seqlen"],
            dropout_linears=config["dropout_linears"],
            out_channels2=config["out_channels2"],
            hidden_size=config["hidden_size"],
        )
        return model_params

    @staticmethod
    def get_hparams() -> Dict[str, Any]:
        hparams = dict(
            batchsize=config["batchsize"],
            lr=config["lr"],
            patience=config["patience"],
            monitor=config["monitor"],
            name=config["name"],
            max_epochs=config["max_epochs"]
        )
        return hparams