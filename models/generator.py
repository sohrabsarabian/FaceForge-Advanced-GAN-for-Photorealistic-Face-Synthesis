import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


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


class Generator(nn.Module):
    def __init__(self, z_dim=200, channels=64, num_classes=0):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.initial = nn.Sequential(
            nn.ConvTranspose2d(z_dim + num_classes, channels * 16, 4, 1, 0),
            nn.BatchNorm2d(channels * 16),
            nn.ReLU(True)
        )

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(channels * 16, channels * 8, 4, 2, 1),
                nn.BatchNorm2d(channels * 8),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels * 8, channels * 4, 4, 2, 1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels * 4, channels * 2, 4, 2, 1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.ConvTranspose2d(channels * 2, channels, 4, 2, 1),
                nn.BatchNorm2d(channels),
                nn.ReLU(True)
            )
        ])

        self.self_attention = SelfAttention(channels * 4)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(channels, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, labels=None):
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("Labels are required when num_classes > 0")
            labels = labels.float()  # Ensure labels are float tensors
            labels = labels.view(labels.size(0), -1, 1, 1)
            noise = torch.cat([noise, labels], dim=1)

        x = self.initial(noise)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if i == 2:  # Apply self-attention after the third block
                x = self.self_attention(x)

        return self.final(x)


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)