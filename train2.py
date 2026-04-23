

import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    matthews_corrcoef, f1_score,
    precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    average_precision_score
)
import pandas as pd

from sklearn.model_selection import KFold

import time
import numpy as np

from Model.Layer4 import ImprovedBaselineATT_MultiScale_BCE

# ===================== Seed =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)



# ===================== Dataset =====================
class MDataset(Dataset):
    def __init__(self, data, y):
        self.data = data
        self.y = y

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        curr = self.data[index]
        concat = np.expand_dims(curr, axis=0)
        return (
            torch.tensor(concat, dtype=torch.float32),
            torch.tensor(self.y[index], dtype=torch.float32)
        )



print("加载数据")
total_start = time.time()
start_time = time.time()
# change
# 使用pandas读取CSV文件，速度更快
X_resampled_protT5_B = pd.read_csv('ProtT5_573A.csv', delimiter=',',header=None).values.astype(np.float32)

X_resampled_esm2_B = pd.read_csv('ESM2_573A.csv', delimiter=',',header=None).values.astype(np.float32)

X_resampled_Ankh_B = pd.read_csv('Ankh_573A.csv', delimiter=',',header=None).values.astype(np.float32)


y_resampled_B = pd.read_csv('ProtT5_573A_label.csv', delimiter=',',header=None).values.astype(np.int32).ravel()


print(f"数据加载完成，总耗时: {time.time() - start_time:.4f} 秒")

print(X_resampled_protT5_B.shape)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ===================== Load Train Data =====================
X_train = np.concatenate(
    [X_resampled_esm2_B, X_resampled_protT5_B, X_resampled_Ankh_B],
    axis=1
)
y_train = y_resampled_B

dataset_full = MDataset(X_train, y_train)

kf = KFold(n_splits=10, shuffle=True, random_state=42)

num_epochs = 16



for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):

    print(f"\n================ Fold {fold+1}/10 ================")

    train_dataset = Subset(dataset_full, train_idx)
    val_dataset   = Subset(dataset_full, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=128, shuffle=False)

    class_counts = torch.zeros(2)
    for _, labels in DataLoader(train_dataset, batch_size=1):
        labels = labels.long()
        class_counts += labels.bincount(minlength=2)

    pos_weight = (class_counts[0] / class_counts[1]).to(device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ---------- Model ----------
    model = ImprovedBaselineATT_MultiScale_BCE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_mcc = -1

    for epoch in range(1, num_epochs + 1):

        # ----- Train -----
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_labels, all_preds, all_probs = [], [], []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # ---------- Metrics ----------
        val_recall = recall_score(all_labels, all_preds, zero_division=0)
        val_precision = precision_score(all_labels, all_preds, zero_division=0)
        val_f1 = f1_score(all_labels, all_preds, zero_division=0)
        MCC = matthews_corrcoef(all_labels, all_preds)

        tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        val_roc = roc_auc_score(all_labels, all_probs)
        val_pr_auc = average_precision_score(all_labels, all_probs)

        print(
            f"Fold [{fold+1}] Epoch [{epoch}/{num_epochs}] | "
            f"Spe: {specificity:.4f} | "
            f"Rec: {val_recall:.4f} | "
            f"Pre: {val_precision:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"MCC: {MCC:.4f} | "
            f"AUC: {val_roc:.4f} | "
            f"PR-AUC: {val_pr_auc:.4f}"
        )
        if MCC > best_mcc :
            best_mcc = MCC
            torch.save(
                model.state_dict(),
                # change
                f"ckpt/fold{fold}.pt"
            )
        scheduler.step()

    print(f"\nFold {fold+1} Best Val MCC: {best_mcc:.4f}")

print("\n Fold Training Finished.")
print(f"{time.time() - start_time:.2f} 秒")