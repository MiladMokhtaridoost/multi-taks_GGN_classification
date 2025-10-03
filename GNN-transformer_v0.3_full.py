#!/usr/bin/env python3
# GNN-transformer_v0.3.py

import os
import re
import sys
import gc
import time
import signal
import random
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.utils import to_undirected, coalesce

# --------------------------
# Args & paths
# --------------------------
pathway = sys.argv[1]
num_epochs = int(sys.argv[2])
learning_rate = float(sys.argv[3])

file_suffix = "cd.10000.iced.sorted.txt"
input_path = f"{pathway}/input_data"
outpath = f"{pathway}/output_full"
os.makedirs(outpath, exist_ok=True)

# Env/config knobs (safe defaults)
VERBOSE = int(os.environ.get("VERBOSE", "0"))
MAX_EDGES = int(os.environ.get("MAX_EDGES", "3000000"))   # hard cap (uniform subsample if exceeded)
MIN_FREQ = float(os.environ.get("MIN_FREQ", "0.0"))       # drop edges with freq < MIN_FREQ (0 = keep all)

# --------------------------
# Device & AMP (H100-friendly)
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

# Fast matmul on A100/H100
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# Prefer BF16 on Hopper
amp_dtype = torch.bfloat16 if use_cuda else torch.float32

# --------------------------
# Label table
# --------------------------
print(pathway)

label_csv = f"{pathway}/label_table.csv"
label_df = pd.read_csv(label_csv)

health_map = {"Healthy": 0, "Cancer": 1}
tissue_map = {"Blood": 0, "Brain": 1, "Breast": 2, "Colon": 3, "Lung": 4, "Skin": 5}
sex_map = {"M": 0, "F": 1}

valid_files, valid_labels = [], []

def _vprint(*a, **k):
    if VERBOSE:
        print(*a, **k)

for _, row in label_df.iterrows():
    srx_id = row["SRX_ID"]
    file_path = os.path.join(input_path, f"{srx_id}.{file_suffix}")
    _vprint(f"🔍 Checking file: {file_path}")

    if os.path.exists(file_path):
        _vprint(f"✅ Found file: {file_path}")
        health_label = health_map.get(row["Healthy/Cancer"], float('nan'))
        tissue_label = tissue_map.get(row["Tissue"], float('nan'))
        sex_label = sex_map.get(row["Sex"], float('nan'))
        valid_files.append(file_path)
        valid_labels.append(torch.tensor([health_label, tissue_label, sex_label], dtype=torch.float32))
    else:
        _vprint(f"❌ File does not exist: {file_path}")

print(f"✅ Found {len(valid_files)} datasets with available Hi-C files.")

# --------------------------
# Dataset
# --------------------------
class HiCDataset(Dataset):
    def __init__(self, files, labels, transform=None):
        super().__init__(transform)
        self.files = files
        self.labels = labels

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]  # shape [3]

        edge_pairs = []

        with open(file_path, "r") as f:
            next(f)  # skip header
            for line in f:
                chrA, chrB, freq = line.strip().split()
                freq = float(freq)
                if freq < MIN_FREQ:
                    continue
                binA = int(chrA.split("_")[1])
                binB = int(chrB.split("_")[1])
                edge_pairs.append((binA, binB))

        # subsample if too many edges
        if len(edge_pairs) > MAX_EDGES:
            idxs = sorted(random.sample(range(len(edge_pairs)), MAX_EDGES))
            edge_pairs = [edge_pairs[i] for i in idxs]

        if len(edge_pairs) == 0:
            # make a trivial 1-node graph
            x = torch.ones((1, 1), dtype=torch.float32)
            y = label.view(1, -1).float()
            return Data(x=x, edge_index=torch.empty((2,0), dtype=torch.long), y=y)

        edge_index = torch.tensor(edge_pairs, dtype=torch.long).t().contiguous()
        num_nodes = int(edge_index.max().item()) + 1

        # undirected + coalesce to drop duplicates
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

        x = torch.ones((num_nodes, 1), dtype=torch.float32)
        y = label.view(1, -1).float()

        # Return on CPU; we move to GPU in the train loop
        return Data(x=x, edge_index=edge_index, y=y)

dataset = HiCDataset(valid_files, valid_labels)

def collate_fn(batch):
    batch_data = Batch.from_data_list(batch)
    labels = torch.stack([d.y.squeeze(0) for d in batch], dim=0)
    batch_data.y = labels
    return batch_data

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=(device.type == "cuda")
  #  persistent_workers=False,
)

# --------------------------
# Model
# --------------------------
class GNNTransformer(nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.fc = nn.Linear(hidden_channels, 8)  # 1 (health) + 6 (tissue) + 1 (sex)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.fc(x)

model = GNNTransformer(in_channels=1, hidden_channels=16).to(device)

# --------------------------
# Checkpoint helpers
# --------------------------
def list_ckpts():
    pats = []
    for name in os.listdir(outpath):
        m = re.match(r"ckpt_epoch_(\d+)\.pt$", name)
        if m:
            pats.append((int(m.group(1)), os.path.join(outpath, name)))
    pats.sort()
    return pats

def latest_ckpt_path():
    pats = list_ckpts()
    return pats[-1][1] if pats else None

def save_checkpoint(epoch, model, optimizer, out_dir):
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "amp_dtype": str(amp_dtype),
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if use_cuda else None,
    }
    tmp = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt.tmp")
    final = os.path.join(out_dir, f"ckpt_epoch_{epoch:04d}.pt")
    latest = os.path.join(out_dir, "ckpt_latest.pt")
    torch.save(state, tmp)
    os.replace(tmp, final)
    # update latest
    torch.save(state, latest)

#def load_latest_checkpoint_if_any(model, optimizer):
#    ck = latest_ckpt_path()
#    if ck and os.path.isfile(ck):
#        print(f"🔁 Resuming from checkpoint: {ck}")
#        map_loc = device
#        state = torch.load(ck, map_location=map_loc)
#        model.load_state_dict(state["model_state"])
#        optimizer.load_state_dict(state["optimizer_state"])
#        if state.get("rng_state") is not None:
#            torch.set_rng_state(state["rng_state"])
#        if use_cuda and state.get("cuda_rng_state") is not None:
#            torch.cuda.set_rng_state_all(state["cuda_rng_state"])
#        start_epoch = int(state["epoch"]) + 1
#        return start_epoch
#    return 1

def load_latest_checkpoint_if_any(model, optimizer):
    ck = latest_ckpt_path()
    if ck and os.path.isfile(ck):
        print(f"🔁 Resuming from checkpoint: {ck}")
        # Always load checkpoint to CPU first; PyTorch will move params when loading state_dicts.
        state = torch.load(ck, map_location="cpu", weights_only=False)

        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])

        # Restore CPU RNG state (must be a CPU ByteTensor)
        try:
            rng = state.get("rng_state", None)
            if rng is not None:
                torch.set_rng_state(rng.cpu())
        except Exception as e:
            print(f"⚠️ Skipping CPU RNG restore ({e})")

        # Restore CUDA RNG state if shapes match
        try:
            if use_cuda:
                cuda_rng_all = state.get("cuda_rng_state", None)
                if isinstance(cuda_rng_all, (list, tuple)) and len(cuda_rng_all) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all([t.to("cuda") for t in cuda_rng_all])
                else:
                    if cuda_rng_all is not None:
                        print("⚠️ Skipping CUDA RNG restore (gpu count mismatch or invalid format).")
        except Exception as e:
            print(f"⚠️ Skipping CUDA RNG restore ({e})")

        start_epoch = int(state.get("epoch", 0)) + 1
        return start_epoch
    return 1

# --------------------------
# Train / Eval
# --------------------------
criterion1 = nn.BCEWithLogitsLoss()  # health
criterion2 = nn.CrossEntropyLoss()   # tissue
criterion3 = nn.BCEWithLogitsLoss()  # sex

optimizer = Adam(model.parameters(), lr=learning_rate)

# SIGTERM-safe abort flag (for walltime/preemption)
_train_state = {"abort": False, "epoch": 0}
def _sig_handler(signum, frame):
    print(f"\n⚠️ Caught signal {signum}. Will save & exit at next safe point.")
    _train_state["abort"] = True

signal.signal(signal.SIGTERM, _sig_handler)
signal.signal(signal.SIGINT, _sig_handler)

def train_model(model, loader, epochs, lr):
    start_epoch = load_latest_checkpoint_if_any(model, optimizer)
    print("Starting training...")
    t0 = time.time()

    for epoch in range(start_epoch, epochs + 1):
        _train_state["epoch"] = epoch
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            try:
                # to device
                batch = batch.to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                    out = model(batch.x, batch.edge_index, batch.batch)

                    # masks
                    health_mask = ~torch.isnan(batch.y[:, 0])
                    tissue_mask = ~torch.isnan(batch.y[:, 1])
                    sex_mask    = ~torch.isnan(batch.y[:, 2])

                    health_labels = batch.y[health_mask, 0].float()
                    tissue_labels = batch.y[tissue_mask, 1].long()
                    sex_labels    = batch.y[sex_mask, 2].float()

                    # outputs
                    health_out = out[:, 0]
                    tissue_out = out[:, 1:7]
                    sex_out    = out[:, 7]

                    # masked outputs
                    loss = 0.0
                    if health_labels.numel() > 0:
                        loss = loss + criterion1(health_out[health_mask], health_labels)
                    if tissue_labels.numel() > 0:
                        loss = loss + criterion2(tissue_out[tissue_mask], tissue_labels)
                    if sex_labels.numel() > 0:
                        loss = loss + criterion3(sex_out[sex_mask], sex_labels)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                total_loss += float(loss.item())
                n_batches += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print("⚠️ CUDA OOM on this batch — skipping.")
                    del batch
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    raise

            # exit early if SLURM signalled us
            if _train_state["abort"]:
                break

        avg_loss = total_loss / max(1, n_batches)
        print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

        # save every epoch (and also when abort flag is set)
        save_checkpoint(epoch, model, optimizer, outpath)

        if _train_state["abort"]:
            print("⏹️ Received stop signal. Saved checkpoint and exiting training loop.")
            break

    dt = time.time() - t0
    print(f"Training finished (or stopped) after {dt/60:.1f} minutes.")

#def evaluate_model(model, loader, save_path=None):
#    model.eval()
#    health_correct = 0
#    tissue_correct = 0
#    sex_correct = 0
#    total = 0

#    with torch.no_grad():
#        for batch in loader:
#            batch = batch.to(device, non_blocking=True)
#            out = model(batch.x, batch.edge_index, batch.batch)

#            health_pred = (out[:, 0] > 0).float()
#            tissue_pred = out[:, 1:7].argmax(dim=1)
#            sex_pred    = (out[:, 7] > 0).float()

#            health_true = batch.y[:, 0]
#            tissue_true = batch.y[:, 1]
#            sex_true    = batch.y[:, 2]

#            health_mask = ~torch.isnan(health_true)
#            tissue_mask = ~torch.isnan(tissue_true)
#            sex_mask    = ~torch.isnan(sex_true)

#            health_correct += (health_pred[health_mask] == health_true[health_mask]).sum().item()
#            tissue_correct += (tissue_pred[tissue_mask] == tissue_true[tissue_mask]).sum().item()
#            sex_correct    += (sex_pred[sex_mask] == sex_true[sex_mask]).sum().item()
#            total          += int(health_mask.sum().item())

#    health_acc = 100.0 * health_correct / max(1, total)
#    tissue_acc = 100.0 * tissue_correct / max(1, total)
#    sex_acc    = 100.0 * sex_correct    / max(1, total)

#    result = (
#        f"Evaluation Results:\n"
#        f"  Health Accuracy: {health_acc:.2f}%\n"
#        f"  Tissue Accuracy: {tissue_acc:.2f}%\n"
#        f"  Sex Accuracy: {sex_acc:.2f}%\n"
#    )
#    print(result)
#    if save_path:
#        with open(save_path, "w") as f:
#            f.write(result)


def evaluate_model(model, loader, save_path=None):
    model.eval()
    health_correct = tissue_correct = sex_correct = 0
    health_total = tissue_total = sex_total = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch)

            # Predictions
            health_pred = (out[:, 0] > 0).float()
            tissue_pred = out[:, 1:7].argmax(dim=1)  # still using 6 tissue classes
            sex_pred = (out[:, 7] > 0).float()

            # Truth
            health_true = batch.y[:, 0]
            tissue_true = batch.y[:, 1]
            sex_true = batch.y[:, 2]

            # Masks per task
            hmask = ~torch.isnan(health_true)
            tmask = ~torch.isnan(tissue_true)
            smask = ~torch.isnan(sex_true)

            # Per-task totals/correct
            health_total += int(hmask.sum())
            tissue_total += int(tmask.sum())
            sex_total += int(smask.sum())

            health_correct += int((health_pred[hmask] == health_true[hmask]).sum())
            tissue_correct += int((tissue_pred[tmask] == tissue_true[tmask]).sum())
            sex_correct += int((sex_pred[smask] == sex_true[smask]).sum())

    # Safe division
    health_acc = 100.0 * health_correct / max(1, health_total)
    tissue_acc = 100.0 * tissue_correct / max(1, tissue_total)
    sex_acc    = 100.0 * sex_correct    / max(1, sex_total)

    result = (
        "Evaluation Results:\n"
        f"  Health Accuracy: {health_acc:.2f}% (n={health_total})\n"
        f"  Tissue Accuracy: {tissue_acc:.2f}% (n={tissue_total})\n"
        f"  Sex Accuracy: {sex_acc:.2f}% (n={sex_total})\n"
    )
    print(result)
    if save_path:
        with open(save_path, "w") as f:
            f.write(result)

# --------------------------
# Sanity check (one batch)
# --------------------------
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(batch.x, batch.edge_index, batch.batch)
        print("Model output shape:", tuple(out.shape))
        break

# --------------------------
# Train -> Save latest -> Evaluate
# --------------------------
train_model(model, loader, epochs=num_epochs, lr=learning_rate)

# Load latest checkpoint for eval
#ck_latest = os.path.join(outpath, "ckpt_latest_40v3.pt")
#if os.path.isfile(ck_latest):
#    state = torch.load(ck_latest, map_location=device)
#    model.load_state_dict(state["model_state"])
#    print(f"🔍 Loaded latest checkpoint (epoch {state['epoch']}) for evaluation.")


ck_latest = os.path.join(outpath, "ckpt_latest.pt")
if os.path.isfile(ck_latest):
    state = torch.load(ck_latest, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state"])
    print(f"🔍 Loaded latest checkpoint (epoch {state.get('epoch','?')}) for evaluation.")


# Also save a plain weights-only file (optional convenience)
torch.save(model.state_dict(), os.path.join(outpath, "model_latest_weights_10v3.pt"))

evaluate_model(model, loader, save_path=os.path.join(outpath, "evaluation_results_10v3.txt"))

