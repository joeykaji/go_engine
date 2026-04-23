import torch
import torch.nn as nn

class GoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        # policy head
        self.policy_conv = nn.Conv2d(64, 2, 1)
        self.policy_fc   = nn.Linear(2 * 19 * 19, 362)

        # value head
        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_fc1  = nn.Linear(19 * 19, 64)
        self.value_fc2  = nn.Linear(64, 1)

    def forward(self, x):
        x = self.conv(x)

        p = self.policy_conv(x)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = self.value_conv(x)
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))

        return p, v

# export to TorchScript
model = GoNet()
model.eval()
scripted = torch.jit.script(model)
scripted.save("gonet.pt")
print("saved gonet.pt")
