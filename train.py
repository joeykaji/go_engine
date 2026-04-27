import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from network import GoNet
import json
import os

class GoDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states   = states
        self.policies = policies
        self.values   = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        s = torch.from_numpy(self.states[idx].copy()).float()
        p = torch.tensor(int(self.policies[idx]), dtype=torch.long)
        v = torch.tensor(float(self.values[idx]), dtype=torch.float32).unsqueeze(0)
        return s, p, v

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
meta     = json.load(open("data/meta.json"))
N        = meta["total_positions"]
states   = np.memmap("data/states.npy",   dtype="float32", mode="r", shape=(N, 3, 19, 19))
policies = np.memmap("data/policies.npy", dtype="int64",   mode="r", shape=(N,))
values   = np.memmap("data/values.npy",   dtype="float32", mode="r", shape=(N,))

dataset     = GoDataset(states, policies, values)
subset_size = min(1_000_000, N)

# ── model + optimizer ─────────────────────────────────────────────────────────

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

model          = GoNet().to(device)
optimizer      = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler      = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn  = nn.MSELoss()

# ── resume from checkpoint if available ──────────────────────────────────────

start_epoch     = 0
checkpoint_path = "checkpoint.pt"

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint["epoch"] + 1
    print(f"Resuming from epoch {start_epoch}")

# ── training loop ─────────────────────────────────────────────────────────────

print("Training...")
for epoch in range(start_epoch, 30):
    indices = np.random.choice(N, subset_size, replace=False)
    sampler = SubsetRandomSampler(indices)
    loader  = DataLoader(dataset, batch_size=256, sampler=sampler,
                         num_workers=0, pin_memory=False)

    total_loss    = 0
    total_batches = 0

    for states_b, policies_b, values_b in loader:
        states_b   = states_b.to(device)
        policies_b = policies_b.to(device)
        values_b   = values_b.to(device)

        optimizer.zero_grad()
        policy_out, value_out = model(states_b)
        loss = policy_loss_fn(policy_out, policies_b) + \
               value_loss_fn(value_out, values_b)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item()
        total_batches += 1

    scheduler.step()
    print(f"Epoch {epoch+1}/30  loss: {total_loss/total_batches:.4f}")

    torch.save({
        "epoch":     epoch,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, checkpoint_path)

# ── save final model ──────────────────────────────────────────────────────────

model.eval()
scripted = torch.jit.script(model)
scripted.save("gonet.pt")
print("Saved gonet.pt")

