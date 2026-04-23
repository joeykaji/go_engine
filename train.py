import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from network import GoNet

# load data
print("Loading data...")
states   = torch.tensor(np.load("states.npy")).float()
policies = torch.tensor(np.load("policies.npy"))
values   = torch.tensor(np.load("values.npy")).float().unsqueeze(1)

dataset = TensorDataset(states, policies, values)
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

model = GoNet()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
policy_loss_fn = nn.CrossEntropyLoss()
value_loss_fn  = nn.MSELoss()

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

# save
model.eval()
scripted = torch.jit.script(model)
scripted.save("gonet.pt")
print("Saved gonet.pt")
