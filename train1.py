

# ===================== Imports =====================
import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import KFold

from sklearn.metrics import (
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score
)

import pandas as pd
import time

# change
from Model import Model





# ===================== Dataset =====================
class DS(Dataset):
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


protT5_df = pd.read_csv(
    './protT5_573.csv',
    delimiter=',',
)


protein_ids = protT5_df.iloc[:, 0].values.astype(np.int32)


X_resampled_protT5_B = protT5_df.iloc[:, 1:].values.astype(np.float32)


X_resampled_esm2_B = pd.read_csv(
    './ESM2_573A.csv',
    delimiter=',',
    header=None
).values.astype(np.float32)

X_resampled_Ankh_B = pd.read_csv(
    './Ankh_573A.csv',
    delimiter=',',
    header=None
).values.astype(np.float32)


y_resampled_B = pd.read_csv(
    './ProtT5_573A_label.csv',
    delimiter=',',
    header=None
).values.astype(np.int32).ravel()


print(f"数据加载完成，总耗时: {time.time() - start_time:.4f} 秒")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ===================== Load Train Data =====================
X_train = np.concatenate(
    [
        X_resampled_esm2_B,
        X_resampled_protT5_B,
        X_resampled_Ankh_B
    ],
    axis=1
)

y_train = y_resampled_B

dataset_full = DS(X_train, y_train)


kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)


unique_proteins = np.unique(protein_ids)


num_epochs = 16


# =========================================================
# Cross Validation
# =========================================================
for fold, (train_prot_idx, val_prot_idx) in enumerate(
        kf.split(unique_proteins)):

    print(f"\n================ Fold {fold+1}/10 ================")
    train_proteins = unique_proteins[train_prot_idx]
    val_proteins = unique_proteins[val_prot_idx]
    train_idx = np.where(
        np.isin(protein_ids, train_proteins)
    )[0]

    val_idx = np.where(
        np.isin(protein_ids, val_proteins)
    )[0]

    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")

    print(f"Train proteins: {len(train_proteins)}")
    print(f"Val proteins: {len(val_proteins)}")

    # =====================================================
    # Dataset
    # =====================================================
    train_dataset = Subset(dataset_full, train_idx)
    val_dataset = Subset(dataset_full, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False
    )

    # pos_weight

    class_counts = torch.zeros(2)

    for _, labels in DataLoader(train_dataset, batch_size=1):

        labels = labels.long()

        class_counts += labels.bincount(minlength=2)

    pos_weight = (class_counts[0] / class_counts[1]).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    print("pos_weight:", pos_weight.item())

    # =====================================================
    # Model
    # =====================================================
    model = model().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-4
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    best_mcc = -1

    # =====================================================
    # Training
    # =====================================================
    for epoch in range(1, num_epochs + 1):

        # ================= Train =================
        model.train()

        train_loss = 0.0

        for inputs, labels in train_loader:

            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

            loss = criterion(logits, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ================= Validation =================
        model.eval()

        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():

            for inputs, labels in val_loader:

                inputs = inputs.to(device)
                labels = labels.to(device)

                logits = model(inputs)

                probs = torch.sigmoid(logits)

                preds = (probs > 0.5).long()

                all_labels.extend(
                    labels.cpu().numpy()
                )

                all_preds.extend(
                    preds.cpu().numpy()
                )

                all_probs.extend(
                    probs.cpu().numpy()
                )

        # =================================================
        # Metrics
        # =================================================
        val_recall = recall_score(
            all_labels,
            all_preds,
            zero_division=0
        )

        val_precision = precision_score(
            all_labels,
            all_preds,
            zero_division=0
        )

        val_f1 = f1_score(
            all_labels,
            all_preds,
            zero_division=0
        )

        MCC = matthews_corrcoef(
            all_labels,
            all_preds
        )

        tn, fp, fn, tp = confusion_matrix(
            all_labels,
            all_preds
        ).ravel()

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0 else 0
        )

        val_roc = roc_auc_score(
            all_labels,
            all_probs
        )

        val_pr_auc = average_precision_score(
            all_labels,
            all_probs
        )

        # =================================================
        # Print
        # =================================================
        print(
            f"Fold [{fold+1}] "
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Loss: {train_loss:.4f} | "
            f"Spe: {specificity:.4f} | "
            f"Rec: {val_recall:.4f} | "
            f"Pre: {val_precision:.4f} | "
            f"F1: {val_f1:.4f} | "
            f"MCC: {MCC:.4f} | "
            f"AUC: {val_roc:.4f} | "
            f"PR-AUC: {val_pr_auc:.4f}"
        )

        # =================================================
        # Save Best
        # =================================================
        if MCC > best_mcc:

            best_mcc = MCC

            torch.save(
                model.state_dict(),
                f"./P1_fold{fold}.pt"
            )

        scheduler.step()

    print(f"\nFold {fold+1} Best Val MCC: {best_mcc:.4f}")

print("\nFold Training Finished.")

print(
    f"整个流程所花费的时间: "
    f"{time.time() - total_start:.2f} 秒"
)