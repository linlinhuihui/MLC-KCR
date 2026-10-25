import os
import torch
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, recall_score, roc_auc_score, average_precision_score,
)

import models.MLCKCR as mlckcr
from models.MLCKCR import classifier
import utils.LitAuto as litauto_module
from utils.LitAuto import LitAuto
from utils.prepare_data import get_kcr

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
os.chdir(PROJECT_ROOT)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

CKPT_DIR = os.path.join(PROJECT_ROOT, "ckpt")
CKPT_NAMES = [
    "MLC1.ckpt",
    "MLC2.ckpt",
    "MLC3.ckpt",
    "MLC4.ckpt",
    "MLC5.ckpt",
]

with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

mlckcr.config = cfg
classifier.config = cfg
litauto_module.config = cfg

_, testset = get_kcr(config=cfg)
test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)
test_labels = np.array([testset[i][1] for i in range(len(testset))])


def compute_metrics(labels, probs):
    preds = (probs >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sn = recall_score(labels, preds)
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_vals, recall_vals, _ = precision_recall_curve(labels, probs)
    return {
        "SN": sn,
        "SP": sp,
        "Accuracy": accuracy_score(labels, preds),
        "AUC": roc_auc_score(labels, probs),
        "AP": average_precision_score(labels, probs),
        "MCC": matthews_corrcoef(labels, preds),
        "F1": f1_score(labels, preds),
        "AUPR": auc(recall_vals, precision_vals),
        "bacc": (sn + sp) / 2,
    }


model_params = classifier.get_model_params()
all_probs = []
results = []

for name in CKPT_NAMES:
    ckpt_path = os.path.join(CKPT_DIR, name)
    if not os.path.exists(ckpt_path):
        continue
    lit = LitAuto(module=classifier, model_params=model_params,
                  lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 0))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    lit.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    lit.eval()
    lit.to(DEVICE)

    probs = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE).float()
            out = lit.module(x)
            probs.extend(torch.sigmoid(out).cpu().numpy().flatten())
    probs = np.array(probs)
    all_probs.append(probs)

    m = compute_metrics(test_labels, probs)
    m["ckpt"] = name
    results.append(m)

    del lit
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if all_probs:
    avg_probs = np.mean(all_probs, axis=0)
    soft_metrics = compute_metrics(test_labels, avg_probs)
    pd.DataFrame([soft_metrics]).to_csv("test.csv", index=False)
else:
    pass
