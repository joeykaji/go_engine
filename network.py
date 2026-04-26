import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))  # skip connection

class GoNet(nn.Module):
    def __init__(self, channels=128, num_blocks=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 19 * 19, 362)

        # value head
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn   = nn.BatchNorm2d(1)
        self.value_fc1  = nn.Linear(19 * 19, 256)
        self.value_fc2  = nn.Linear(256, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        p = torch.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)

        v = torch.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = torch.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        return p, v

model = GoNet(channels=128, num_blocks=8)
model.eval()
scripted = torch.jit.script(model)
scripted.save("gonet.pt")
print("saved gonet.pt")
