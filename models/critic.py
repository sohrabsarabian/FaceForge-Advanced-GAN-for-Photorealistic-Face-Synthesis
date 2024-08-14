import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.leaky_relu(out, 0.2)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out

class Critic(nn.Module):
    def __init__(self, channels=64, num_classes=0):
        super(Critic, self).__init__()
        self.num_classes = num_classes

        self.initial = nn.Sequential(
            nn.Conv2d(3 + num_classes, channels, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels * 2, 4, 2, 1),
                nn.InstanceNorm2d(channels * 2),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(channels * 2, channels * 4, 4, 2, 1),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(channels * 4, channels * 8, 4, 2, 1),
                nn.InstanceNorm2d(channels * 8),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                nn.Conv2d(channels * 8, channels * 16, 4, 2, 1),
                nn.InstanceNorm2d(channels * 16),
                nn.LeakyReLU(0.2)
            )
        ])

        self.self_attention = SelfAttention(channels * 4)

        # Final layers
        self.final_block = nn.Sequential(
            nn.Conv2d(channels * 16, 1, 4, 1, 0),
        )

    def forward(self, x, labels=None):
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("Labels are required when num_classes > 0")
            labels = labels.float()  # Ensure labels are float tensors
            labels = labels.view(labels.size(0), -1, 1, 1)
            labels = labels.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, labels], dim=1)

        out = self.initial(x)

        for i, block in enumerate(self.blocks):
            out = block(out)
            if i == 2:  # Apply self-attention after the third block
                out = self.self_attention(out)

        out = self.final_block(out)
        return out.view(out.shape[0], -1)