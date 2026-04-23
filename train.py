import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from network import GoNet
import json

# ── lazy dataset ──────────────────────────────────────────────────────────────

class GoDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states   = states
        self.policies = policies
        self.values   = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        # reads only this slice from disk — no full load into RAM
        s = torch.from_numpy(self.states[idx].copy()).float()
        p = torch.tensor(self.policies[idx], dtype=torch.long)
        v = torch.tensor(self.values[idx],   dtype=torch.float32).unsqueeze(0)
        return s, p, v

# ── load memmaps ──────────────────────────────────────────────────────────────

print("Loading data...")
meta     = json.load(open("data/meta.json"))
N        = meta["total_positions"]
states   = np.memmap("data/states.npy",   dtype="float32", mode="r",
                     shape=(N, 3, 19, 19))
policies = np.memmap("data/policies.npy", dtype="int64",   mode="r",
                     shape=(N,))
values   = np.memmap("data/values.npy",   dtype="float32", mode="r",
                     shape=(N,))

dataset = GoDataset(states, policies, values)
loader  = DataLoader(dataset, batch_size=256, shuffle=True,
                     num_workers=4, pin_memory=True)

# ── model ─────────────────────────────────────────────────────────────────────

model = GoNet()
optimizer      = torch.optim.Adam(model.parameters(), lr=1e-3)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn  = nn.MSELoss()

# ── training loop ─────────────────────────────────────────────────────────────

print("Training...")
for epoch in range(10):
    total_loss = 0
    for states_b, policies_b, values_b in loader:
        optimizer.zero_grad()
        policy_out, value_out = model(states_b)
        loss = policy_loss_fn(policy_out, policies_b) + \
               value_loss_fn(value_out, values_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10  loss: {total_loss/len(loader):.4f}")

# ── save ──────────────────────────────────────────────────────────────────────

model.eval()
scripted = torch.jit.script(model)
scripted.save("gonet.pt")
print("Saved gonet.pt")
