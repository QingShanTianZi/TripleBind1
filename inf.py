
import numpy as np
import time

Time1 = time.time()

all_embeddings_test_protT5_129= np.loadtxt('ProtT5_TE129.csv', delimiter=',')
all_embeddings_test_protT5_129 = all_embeddings_test_protT5_129.astype(np.float32)


all_embeddings_test_esm2_129= np.loadtxt('ESM2_TE129.csv', delimiter=',')
all_embeddings_test_esm2_129 = all_embeddings_test_esm2_129.astype(np.float32)
all_embeddings_test_Ankh_129  = np.loadtxt('Ankh_TE129.csv', delimiter=',')
all_embeddings_test_Ankh_129 = all_embeddings_test_Ankh_129.astype(np.float32)

all_labels_test_129= np.loadtxt('ProtT5_TE129_label.csv', delimiter=',')
all_labels_test_129 = all_labels_test_129.astype(np.int32)
# ===================== test.py =====================
print(all_embeddings_test_protT5_129.shape)
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    matthews_corrcoef, f1_score,
    precision_score, recall_score,
    roc_auc_score, confusion_matrix,
    accuracy_score
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# ===================== Test Dataset =====================
all_embeddings_test_129 = np.concatenate(
    [
        all_embeddings_test_esm2_129,
        all_embeddings_test_protT5_129,
        all_embeddings_test_Ankh_129
    ],
    axis=1
)

dataset_test_129 = MDataset(
    all_embeddings_test_129,
    all_labels_test_129
)

from Model.Layer4 import ImprovedBaselineATT_MultiScale_BCE

# ===================== Evaluate Function =====================
def evaluate(ckpt_path):

    model = ImprovedBaselineATT_MultiScale_BCE().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(dataset_test_129, batch_size=128, shuffle=False)

    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    mcc = matthews_corrcoef(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    pre = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    auc = roc_auc_score(all_labels, all_probs)

    print("\n===== Test Result =====")
    print(f"MCC : {mcc:.4f}")
    print(f"F1  : {f1:.4f}")
    print(f"Pre : {pre:.4f}")
    print(f"Rec : {rec:.4f}")
    print(f"SPE : {spe:.4f}")
    print(f"AUC : {auc:.4f}")
    print(f"ACC : {acc:.4f}")

    return mcc, f1, pre, rec, spe, auc,acc
def mean_std(values):
    n = len(values)
    mean_val = sum(values) / n
    variance = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_val = variance ** 0.5
    return mean_val, std_val


# 初始化列表
mcc_list, f1_list, pre_list, rec_list, spe_list, auc_list,acc_lsit = [], [], [], [], [], [],[]

# ===================== 指定模型 =====================
for i in range(10):
    ckpt_path = f"ckpt/tempa/573_best_model_fold{i}.pt"
    print(ckpt_path)

    mcc, f1, pre, rec, spe, auc ,acc= evaluate(ckpt_path)

    mcc_list.append(mcc)
    f1_list.append(f1)
    pre_list.append(pre)
    rec_list.append(rec)
    spe_list.append(spe)
    auc_list.append(auc)
    acc_lsit.append(acc)

# 计算并打印结果
metrics = [
    ('MCC', mcc_list),
    ('F1', f1_list),
    ('Precision', pre_list),
    ('Recall', rec_list),
    ('Specificity', spe_list),
    ('AUC', auc_list),
    ('Accuracy', acc_lsit)
]

print("\n" + "-" * 50)
print("Cross-validation results (mean ± std):")
print("-" * 50)

for name, values in metrics:
    avg, std = mean_std(values)
    print(f"{name:10s}: {avg:.4f} ± {std:.4f}")

Time2 = time.time()
print(f" time: {Time2 - Time1}")