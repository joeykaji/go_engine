import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = torch.jit.load("gonet.pt")
model.to(device)
model.eval()
blank = torch.zeros(1, 3, 19, 19).to(device)
with torch.no_grad():
    policy, value = model(blank)
print(f"Value: {value.item():.3f}")
probs = torch.softmax(policy, dim=1)[0]
top5 = torch.topk(probs, 5)
for p, i in zip(top5.values, top5.indices):
    pos = i.item()
    print(f"  ({pos//19},{pos%19}): {p.item():.3f}")
