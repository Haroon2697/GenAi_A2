"""CycleGAN model definitions for Flask app.
Matches the training architecture exactly: IMG=128, N_RES=6.
"""
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with reflection padding and instance normalization."""
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.b(x)


class Generator(nn.Module):
    """ResNet generator: c7s1-64, d128, d256, R256 x N, u128, u64, c7s1-3.
    Matches the training configuration with IMG=128 and N_RES=6.
    """
    def __init__(self, n_res=6, nf=64):
        super().__init__()
        L = [nn.ReflectionPad2d(3), nn.Conv2d(3, nf, 7), nn.InstanceNorm2d(nf), nn.ReLU(True)]
        c = nf
        for _ in range(2):
            L += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        L += [ResBlock(c) for _ in range(n_res)]
        for _ in range(2):
            L += [nn.ConvTranspose2d(c, c // 2, 3, 2, 1, output_padding=1),
                  nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        L += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*L)

    def forward(self, x):
        return self.net(x)
