import torch
import torch.nn as nn
import torch.nn.functional as F
NZ,NX=201,301

# Step 5 - Network: modified U-Net (domain projection (x,t) -> (x,z))
# Shots are input channels, channels 64-128-256-512-1024, skip connections, 23 conv layers,
# final feature map truncated to the velocity-model size, one output channel.
 
# %%
def dconv(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
 
 
class FCNVMB(nn.Module):
    def __init__(self, in_ch, out_hw=(NZ, NX), base=64, depth=4):
        super().__init__()
        self.out_hw, self.depth = out_hw, depth
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([dconv(in_ch, ch[0])] + [dconv(ch[i], ch[i + 1]) for i in range(depth)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2) for i in reversed(range(depth))])
        self.dec = nn.ModuleList([dconv(ch[i] * 2, ch[i]) for i in reversed(range(depth))])
        self.out = nn.Conv2d(ch[0], 1, 1)
 
    def forward(self, x):
        T, X = x.shape[-2:]
        m = 2 ** self.depth
        x = F.pad(x, (0, (-X) % m, 0, (-T) % m))
        skips = []
        for i, blk in enumerate(self.enc):
            x = blk(x)
            if i < self.depth:
                skips.append(x); x = F.max_pool2d(x, 2)
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat([up(x), skips.pop()], 1))
        x = self.out(x)
        nz, nx = self.out_hw
        if x.shape[-2] >= nz and x.shape[-1] >= nx:
            return x[..., :nz, :nx]
        return F.interpolate(x, size=(nz, nx), mode="bilinear", align_corners=False)
 
 
