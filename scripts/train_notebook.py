# Original notebook code cells, demo preset. Run locally; expensive.
# %%
import os, json, time, math, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt


PRESET = "demo"          # "tiny" | "demo" | "paper"
SEG_DIR = ""             # optional: folder with SEG salt velocity models (.npy/.mat, 201x301)
WORK_DIR = "./fcnvmb_run"
SEED = 0
 
PRESETS = {   # paper values = Tables 3 and 4
    "paper": dict(n_train=1600, n_test=100, shots=29, nt=2000, ds=5, epochs=100, batch=10,
                  base=64, eval_every=10, fwi_models=3, fwi_iters=20,
                  seg_train=130, seg_test=10, seg_epochs=50),
    "demo":  dict(n_train=48, n_test=8, shots=5, nt=1000, ds=4, epochs=40, batch=4,
                  base=16, eval_every=5, fwi_models=1, fwi_iters=8,
                  seg_train=24, seg_test=4, seg_epochs=20),
    "tiny":  dict(n_train=6, n_test=2, shots=3, nt=500, ds=4, epochs=3, batch=3,
                  base=8, eval_every=1, fwi_models=1, fwi_iters=2,
                  seg_train=4, seg_test=2, seg_epochs=2),
}
CFG = PRESETS[PRESET]
 


NZ, NX, DH = 201, 301, 10.0           # model grid (depth x width), spacing (m)
DT, FPEAK = 0.001, 25.0               # FD time step (s), Ricker peak frequency (Hz)
VMIN_NORM, VRANGE_NORM = 1500.0, 3000.0
LR, NOISE_PCT = 1e-3, 0.05
 
DATA_DIR = os.path.join(WORK_DIR, "data")
OUT_DIR = os.path.join(WORK_DIR, "results")
FIG_DIR = os.path.join(WORK_DIR, "figures")
for d in (DATA_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(d, exist_ok=True)
 
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); np.random.seed(SEED)
 
 
def show(fig, name):
    """Save the figure and display it right where it was produced."""
    fig.savefig(os.path.join(FIG_DIR, name + ".png"), dpi=130, bbox_inches="tight")
    plt.show()
 
 
print(f"PRESET = {PRESET}   device = {DEV}")
print("Configuration:", json.dumps(CFG))


def make_velocity_model(rng, nz=NZ, nx=NX, dh=DH, salt=True):
    zmax = (nz - 1) * dh
    xs, zs = np.arange(nx) * dh, np.arange(nz) * dh
    nl = int(rng.integers(5, 13))
    vel = np.sort(rng.uniform(2000, 4000, nl))
    t = np.cumsum(rng.uniform(0.5, 1.5, nl - 1))
    t = t / t[-1] * rng.uniform(0.6, 0.95) * zmax
    tops = np.zeros((nl - 1, nx))
    for k in range(nl - 1):
        curve = np.zeros(nx)
        for _ in range(2):
            amp = rng.uniform(0, 0.05) * zmax
            wl = rng.uniform(0.5, 2.0) * xs[-1]
            curve += amp * np.sin(2 * np.pi * xs / wl + rng.uniform(0, 2 * np.pi))
        tops[k] = t[k] + curve
    tops = np.maximum.accumulate(np.clip(tops, dh, zmax), axis=0)
    idx = (zs[:, None, None] >= tops[None, :, :]).sum(axis=1)
    # Keep indices within the available velocity array.
    idx = np.clip(idx, 0, len(vel) - 1)

# Assign a velocity to every grid cell and smooth slightly.
    v = gaussian_filter(vel[idx], sigma=1.0)
    if salt:
        X, Z = np.meshgrid(xs, zs)
        cx = rng.uniform(0.2, 0.8) * xs[-1]
        if rng.random() < 0.4:                                   # dome rising from the bottom
            w = rng.uniform(0.15, 0.45) * xs[-1]
            h = rng.uniform(0.3, 0.8) * zmax
            p = rng.uniform(1.0, 2.0)
            ztop = zmax - h * np.clip(1 - ((xs - cx) / w) ** 2, 0, None) ** p
            mask = (Z >= ztop[None, :]) & (np.abs(X - cx) < w)
        else:                                                    # blob with Fourier boundary
            cz = rng.uniform(0.35, 0.85) * zmax
            rx, rz = rng.uniform(0.1, 0.35) * xs[-1], rng.uniform(0.08, 0.25) * zmax
            th = np.arctan2((Z - cz) / rz, (X - cx) / rx)
            rr = np.hypot((X - cx) / rx, (Z - cz) / rz)
            bound = np.ones_like(th)
            for k in range(2, 6):
                bound += rng.uniform(-0.18, 0.18) / (k / 2) * np.cos(k * th + rng.uniform(0, 6.28))
            mask = rr < bound
        v = np.where(mask, 4500.0, v)
    return v.astype(np.float32)
 


def load_external_models(folder, nz=NZ, nx=NX):
    """Load 2-D velocity models (201x301) from .npy / .mat files (e.g. SEG salt models)."""
    files = sorted(glob.glob(os.path.join(folder, "**", "*.npy"), recursive=True) +
                   glob.glob(os.path.join(folder, "**", "*.mat"), recursive=True))
    models = []
    for f in files:
        try:
            if f.endswith(".npy"):
                arrs = [np.load(f)]
            else:
                from scipy.io import loadmat
                arrs = [a for a in loadmat(f).values() if isinstance(a, np.ndarray)]
        except Exception:
            continue
        for a in arrs:
            a = np.squeeze(a)
            items = list(a) if (a.ndim == 3 and a.shape[0] < 5000) else [a]
            for m in items:
                if m.ndim != 2:
                    continue
                if m.shape == (nx, nz):
                    m = m.T
                if m.shape == (nz, nx):
                    models.append(m.astype(np.float32))
    return models
 


# %% --- RUN: generate and look at synthetic velocity models ---
rng_demo = np.random.default_rng(SEED)
demo_models = [make_velocity_model(rng_demo) for _ in range(6)]
print(f"Generated {len(demo_models)} sample velocity models, each of shape {demo_models[0].shape} "
      f"(depth x width), velocity range {min(m.min() for m in demo_models):.0f}-"
      f"{max(m.max() for m in demo_models):.0f} m/s")
fig, axs = plt.subplots(2, 3, figsize=(12, 6))
for a, m in zip(axs.ravel(), demo_models):
    im = a.imshow(m, cmap="viridis", vmin=1500, vmax=4500, extent=[0, 3, 2, 0], aspect="auto")
    a.set_xlabel("Position (km)"); a.set_ylabel("Depth (km)")
fig.colorbar(im, ax=axs, label="Velocity (m/s)", shrink=0.8)
fig.suptitle("Synthetic velocity models (layers + salt body)")
show(fig, "01_synthetic_velocity_models")
 

print("Number of models:", len(demo_models))
print("Model shape:", demo_models[0].shape)
print("Minimum velocity:", demo_models[0].min(), "m/s")
print("Maximum velocity:", demo_models[0].max(), "m/s")
print("All values finite:", np.isfinite(demo_models[0]).all())


# %%
class Modeler:
    def __init__(self, shots, nt, ds, device, nz=NZ, nx=NX, dh=DH, dt=DT, fpeak=FPEAK,
                 pad=40, damp_a=0.010):
        self.S, self.nt, self.ds, self.dev = shots, nt, ds, device
        self.nz, self.nx, self.dh, self.dt, self.pad = nz, nx, dh, dt, pad
        c = np.array([-205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560]) / dh ** 2
        k = torch.tensor(np.concatenate([c[:0:-1], c]), dtype=torch.float32, device=device)
        self.kx, self.kz = k.view(1, 1, 1, 9), k.view(1, 1, 9, 1)
        self.Zp, self.Xp = nz + pad, nx + 2 * pad
        fx = np.ones(self.Xp)
        for i in range(pad):
            f = np.exp(-(damp_a * (pad - i)) ** 2)
            fx[i] = f; fx[self.Xp - 1 - i] = f
        fz = np.ones(self.Zp)
        for i in range(pad):
            fz[self.Zp - 1 - i] = np.exp(-(damp_a * (pad - i)) ** 2)
        self.damp = torch.tensor(fz[:, None] * fx[None, :], dtype=torch.float32,
                                 device=device).view(1, 1, self.Zp, self.Xp)
        t = np.arange(nt) * dt
        a = (np.pi * fpeak * (t - 1.0 / fpeak)) ** 2
        self.wav = torch.tensor((1 - 2 * a) * np.exp(-a), dtype=torch.float32, device=device)
        self.src_x = np.round(np.linspace(0, nx - 1, shots)).astype(int)
        mask = torch.zeros(shots, 1, self.Zp, self.Xp, device=device)
        for s, x in enumerate(self.src_x):
            mask[s, 0, 1, pad + x] = 1.0 / dh ** 2
        self.src_mask = mask
 
    def pad_vel(self, v):
        return F.pad(v.view(1, 1, self.nz, self.nx), (self.pad, self.pad, 0, self.pad), mode="replicate")
 
    def _lap(self, p):
        return F.conv2d(p, self.kx, padding=(0, 4)) + F.conv2d(p, self.kz, padding=(4, 0))
 
    def _chunk(self, vdt2, mask, p0, p1, it0, it1):
        rec = []
        for it in range(it0, it1):
            p2 = 2 * p1 - p0 + vdt2 * (self._lap(p1) + mask * self.wav[it])
            p0, p1 = p1 * self.damp, p2 * self.damp
            if it % self.ds == 0:
                rec.append(p1[:, 0, 1, self.pad:self.pad + self.nx])
        return p0, p1, torch.stack(rec, 1)
 
    def simulate(self, v, shot_ids=None, grad=False):
        """v: (nz,nx) tensor in m/s -> seismograms (n_shots, n_time_samples, nx)."""
        sid = list(range(self.S)) if shot_ids is None else list(shot_ids)
        vdt2 = (self.pad_vel(v) * self.dt) ** 2
        mask = self.src_mask[sid]
        p0 = torch.zeros(len(sid), 1, self.Zp, self.Xp, device=self.dev)
        p1 = torch.zeros_like(p0)
        recs, step = [], self.ds * 10
        for it0 in range(0, self.nt, step):
            it1 = min(self.nt, it0 + step)
            if grad:
                p0, p1, r = checkpoint(self._chunk, vdt2, mask, p0, p1, it0, it1, use_reentrant=False)
            else:
                p0, p1, r = self._chunk(vdt2, mask, p0, p1, it0, it1)
            recs.append(r)
        return torch.cat(recs, 1)
 


# %% --- RUN: forward-model one velocity model and show the shot gathers ---
modeler = Modeler(CFG["shots"], CFG["nt"], CFG["ds"], DEV)
t0 = time.time()
with torch.no_grad():
    gathers = modeler.simulate(torch.tensor(demo_models[0], device=DEV)).cpu().numpy()
print(f"Forward modeling of 1 velocity model: {time.time() - t0:.1f} s")
print(f"Seismic data shape (shots, time samples, receivers) = {gathers.shape}")
print(f"Time step after downsampling = {DT * CFG['ds'] * 1000:.0f} ms, record length = "
      f"{CFG['nt'] * DT:.1f} s, source positions (x index) = {modeler.src_x.tolist()}")
ns = min(3, gathers.shape[0])
fig, axs = plt.subplots(1, ns + 1, figsize=(4 * (ns + 1), 3.6))
axs[0].imshow(demo_models[0], cmap="viridis", vmin=1500, vmax=4500, extent=[0, 3, 2, 0], aspect="auto")
axs[0].set_title("Velocity model"); axs[0].set_xlabel("Position (km)"); axs[0].set_ylabel("Depth (km)")
amp = np.percentile(np.abs(gathers), 99.5)
for i in range(ns):
    k = int(round(i * (gathers.shape[0] - 1) / max(1, ns - 1)))
    axs[i + 1].imshow(gathers[k], cmap="seismic", vmin=-amp, vmax=amp, aspect="auto",
                      extent=[0, 3, CFG["nt"] * DT, 0])
    axs[i + 1].set_title(f"Shot {k + 1} (x = {modeler.src_x[k] * DH / 1000:.2f} km)")
    axs[i + 1].set_xlabel("Position (km)"); axs[i + 1].set_ylabel("Time (s)")
fig.tight_layout()
show(fig, "02_shot_gathers")
 


# %%
def build_split(name, models, out_dir, modeler, scale=None):
    """Forward-model all models -> data (N,S,T,X) float16, velocities (N,Z,X) float32."""
    os.makedirs(out_dir, exist_ok=True)
    n, T = len(models), math.ceil(modeler.nt / modeler.ds)
    data = np.lib.format.open_memmap(os.path.join(out_dir, f"{name}_data.npy"), mode="w+",
                                     dtype=np.float16, shape=(n, modeler.S, T, modeler.nx))
    vel = np.stack(models).astype(np.float32)
    t0 = time.time()
    for i, v in enumerate(vel):
        with torch.no_grad():
            d = modeler.simulate(torch.tensor(v, device=modeler.dev)).cpu().numpy()
        if scale is None:
            scale = float(1.0 / (d.std() + 1e-12))
        data[i] = (d * scale).astype(np.float16)
        if (i + 1) % max(1, n // 10) == 0 or i == n - 1:
            print(f"  [{name}] {i + 1}/{n} models done  ({time.time() - t0:.0f} s)", flush=True)
    data.flush()
    np.save(os.path.join(out_dir, f"{name}_vel.npy"), vel)
    return scale
 
 
def generate_data(modeler, n_train, n_test, data_dir, seed=0, regen=False):
    meta = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta) and not regen:
        print("Data already exists -> reusing it (set regen=True to rebuild).")
        return json.load(open(meta))["scale"]
    rng = np.random.default_rng(seed)
    tr = [make_velocity_model(rng) for _ in range(n_train)]
    te = [make_velocity_model(rng) for _ in range(n_test)]
    scale = build_split("train", tr, data_dir, modeler)
    build_split("test", te, data_dir, modeler, scale)
    json.dump(dict(scale=scale, shots=modeler.S, nt=modeler.nt, ds=modeler.ds), open(meta, "w"))
    return scale
 
 
# %% --- RUN: create the simulated training / testing data and report how much we have ---
SCALE = generate_data(modeler, CFG["n_train"], CFG["n_test"], DATA_DIR, SEED)
_tr_d = np.load(os.path.join(DATA_DIR, "train_data.npy"), mmap_mode="r")
_te_d = np.load(os.path.join(DATA_DIR, "test_data.npy"), mmap_mode="r")
_tr_v = np.load(os.path.join(DATA_DIR, "train_vel.npy"))
_te_v = np.load(os.path.join(DATA_DIR, "test_vel.npy"))
size_gb = (_tr_d.nbytes + _te_d.nbytes) / 1e9
print("\n===== DATA SET SUMMARY =====")
print(f"Training models : {len(_tr_v)}   seismic data {_tr_d.shape}  velocity {_tr_v.shape}")
print(f"Testing  models : {len(_te_v)}   seismic data {_te_d.shape}  velocity {_te_v.shape}")
print(f"Shots per model (= network input channels): {_tr_d.shape[1]}")
print(f"Amplitude scale factor applied to data: {SCALE:.4g}   (data stored as float16, {size_gb:.2f} GB)")
fig, axs = plt.subplots(1, 3, figsize=(14, 3.4))
for a, (lab, arr) in zip(axs[:2], [("train", _tr_v), ("test", _te_v)]):
    a.hist(arr.ravel(), bins=60, color="C0" if lab == "train" else "C1")
    a.set_title(f"Velocity histogram ({lab}, {len(arr)} models)"); a.set_xlabel("Velocity (m/s)")
axs[2].imshow(np.asarray(_tr_d[0, 0], dtype=np.float32), cmap="seismic", vmin=-3, vmax=3, aspect="auto")
axs[2].set_title("Normalised shot 1 of training model 0"); axs[2].set_xlabel("Receiver"); axs[2].set_ylabel("Time sample")
fig.tight_layout()
show(fig, "03_dataset_summary")
 


# ## Step 4 - Dataset / loaders and low-frequency filter
 
# %%
def lowcut_filter(x, frac=0.1):
    """Remove the lowest `frac` of the normalised Fourier spectrum along time (x: ...,T,X)."""
    Xf = torch.fft.rfft(x, dim=-2)
    Xf[..., :max(1, int(frac * Xf.shape[-2])), :] = 0
    return torch.fft.irfft(Xf, n=x.shape[-2], dim=-2)
 
 
class SeisDataset(Dataset):
    def __init__(self, folder, split, lowcut=False):
        self.d = np.load(os.path.join(folder, f"{split}_data.npy"), mmap_mode="r")
        self.v = np.load(os.path.join(folder, f"{split}_vel.npy"))
        self.lowcut = lowcut
 
    def __len__(self):
        return len(self.v)
 
    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.d[i], dtype=np.float32))
        if self.lowcut:
            x = lowcut_filter(x)
        y = torch.from_numpy((self.v[i] - VMIN_NORM) / VRANGE_NORM).unsqueeze(0)
        return x, y
 
 
def denorm(y):
    return y * VRANGE_NORM + VMIN_NORM
 
 
# %% --- RUN: check the data loader ---
train_ds, test_ds = SeisDataset(DATA_DIR, "train"), SeisDataset(DATA_DIR, "test")
xb, yb = next(iter(DataLoader(train_ds, batch_size=CFG["batch"], shuffle=True)))
print(f"Train set: {len(train_ds)} samples | Test set: {len(test_ds)} samples")
print(f"One batch -> input {tuple(xb.shape)} (batch, shots, time, receivers), "
      f"target {tuple(yb.shape)} (batch, 1, depth, width)")
print(f"Number of mini-batches per epoch: {math.ceil(len(train_ds) / CFG['batch'])}")
x0 = train_ds[0][0]
x0f = lowcut_filter(x0.clone())
fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
for a, d, t in zip(axs, [x0[0], x0f[0]], ["Original (shot 1)", "Lowest 10% of spectrum removed"]):
    a.imshow(d.numpy(), cmap="seismic", vmin=-3, vmax=3, aspect="auto"); a.set_title(t)
    a.set_xlabel("Receiver"); a.set_ylabel("Time sample")
fig.tight_layout()
show(fig, "04_lowcut_example")
 


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
 
 
# %% --- RUN: build the network and check one forward pass ---
net_test = FCNVMB(CFG["shots"], (NZ, NX), CFG["base"]).to(DEV)
n_params = sum(p.numel() for p in net_test.parameters())
n_conv = sum(isinstance(m, nn.Conv2d) for m in net_test.modules())
with torch.no_grad():
    out = net_test(xb.to(DEV))
print(f"Network: {n_params / 1e6:.2f} M parameters, {n_conv} conv layers "
      f"(incl. 1x1 output), base channels = {CFG['base']}")
print(f"Forward pass: input {tuple(xb.shape)} -> output {tuple(out.shape)}")
del net_test, out
 

# Step 6 - Metrics and evaluation helpers
 
# %%
def ssim_t(a, b, data_range, win=7):
    C1, C2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mu_a, mu_b = F.avg_pool2d(a, win, 1), F.avg_pool2d(b, win, 1)
    saa = F.avg_pool2d(a * a, win, 1) - mu_a ** 2
    sbb = F.avg_pool2d(b * b, win, 1) - mu_b ** 2
    sab = F.avg_pool2d(a * b, win, 1) - mu_a * mu_b
    s = ((2 * mu_a * mu_b + C1) * (2 * sab + C2)) / ((mu_a ** 2 + mu_b ** 2 + C1) * (saa + sbb + C2))
    return s.mean(dim=(1, 2, 3))
 
 
def metrics(pred, true):
    mse = ((pred - true) ** 2).mean(dim=(1, 2, 3))
    psnr = 10 * torch.log10(VRANGE_NORM ** 2 / mse.clamp_min(1e-9))
    return mse, psnr, ssim_t(pred, true, VRANGE_NORM)
 
 
@torch.no_grad()
def evaluate(net, loader, transform=None):
    net.eval()
    ms, ps, ss = [], [], []
    for x, y in loader:
        x, y = x.to(DEV), denorm(y.to(DEV))
        if transform is not None:
            x = transform(x)
        m, p, s = metrics(denorm(net(x)), y)
        ms.append(m); ps.append(p); ss.append(s)
    cat = lambda l: torch.cat(l).mean().item()
    return dict(mse=cat(ms), psnr=cat(ps), ssim=cat(ss))
 
 
# %% --- RUN: sanity check of the metrics (perfect prediction vs. constant guess) ---
_t = denorm(yb[:2].to(DEV))
print("perfect prediction  ->", [round(v.mean().item(), 3) for v in metrics(_t, _t)], "(MSE, PSNR, SSIM)")
print("constant 3000 m/s   ->", [round(v.mean().item(), 3) for v in metrics(torch.full_like(_t, 3000.), _t)])
 

# Step 7 - Training (Adam, MSE loss in the model domain, Eq. 7)
 
# %%
def plot_history(h, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].semilogy(h["epoch"], h["train_mse"], label="train")
    if h["test_epoch"]:
        ax[0].semilogy(h["test_epoch"], h["test_mse"], "o-", label="test")
    ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("MSE (m/s)$^2$"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[0].set_title("Loss decrease during training")
    if h["test_epoch"]:
        ax[1].plot(h["test_epoch"], h["test_psnr"], "o-", label="PSNR")
        ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("PSNR (dB)")
        ax2 = ax[1].twinx(); ax2.plot(h["test_epoch"], h["test_ssim"], "s--", c="C1"); ax2.set_ylabel("SSIM (orange)")
        ax[1].set_title("Test PSNR / SSIM")
    fig.suptitle(title); fig.tight_layout()
    return fig
 
 
def train(folder, ckpt_name, epochs, init_ckpt=None, lowcut=False, title="Training"):
    scale = json.load(open(os.path.join(folder, "meta.json")))["scale"]
    tr, te = SeisDataset(folder, "train", lowcut), SeisDataset(folder, "test", lowcut)
    tl = DataLoader(tr, batch_size=CFG["batch"], shuffle=True)
    el = DataLoader(te, batch_size=CFG["batch"])
    S = tr.d.shape[1]
    net = FCNVMB(S, (NZ, NX), CFG["base"]).to(DEV)
    if init_ckpt and os.path.exists(init_ckpt):
        net.load_state_dict(torch.load(init_ckpt, map_location=DEV)["state"])
        print("Initialised with pretrained network (transfer learning):", init_ckpt)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    print(f"Training on {len(tr)} models, testing on {len(te)}, {S} shots, "
          f"{sum(p.numel() for p in net.parameters()) / 1e6:.2f} M params, device = {DEV}")
    hist = dict(epoch=[], train_mse=[], test_epoch=[], test_mse=[], test_psnr=[], test_ssim=[])
    t0 = time.time()
    for ep in range(1, epochs + 1):
        net.train(); run = n = 0
        for x, y in tl:
            x, y = x.to(DEV), y.to(DEV)
            loss = F.mse_loss(net(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            run += loss.item() * len(x); n += len(x)
        hist["epoch"].append(ep); hist["train_mse"].append(run / n * VRANGE_NORM ** 2)
        msg = f"epoch {ep:3d}/{epochs}  train MSE {hist['train_mse'][-1]:12.1f}"
        if ep % CFG["eval_every"] == 0 or ep == epochs:
            r = evaluate(net, el)
            for k, key in [("test_mse", "mse"), ("test_psnr", "psnr"), ("test_ssim", "ssim")]:
                hist[k].append(r[key])
            hist["test_epoch"].append(ep)
            msg += f" | test MSE {r['mse']:12.1f}  PSNR {r['psnr']:6.2f} dB  SSIM {r['ssim']:.3f}"
        print(msg, flush=True)
    train_time = time.time() - t0
    torch.save(dict(state=net.state_dict(), in_ch=S, base=CFG["base"], scale=scale, lowcut=lowcut,
                    train_time=train_time, hist=hist), os.path.join(OUT_DIR, ckpt_name))
    print(f"\nTraining finished in {train_time / 60:.2f} min -> saved {ckpt_name}")
    return net, hist, train_time
 
 
# %% --- RUN: train the network on the simulated data set ---
net_sim, hist_sim, train_time_sim = train(DATA_DIR, "fcnvmb_sim.pt", CFG["epochs"],
                                          title="Simulated models")
fig = plot_history(hist_sim, "Training on simulated velocity models")
show(fig, "05_training_curves_sim")
 

# Step 8 - Prediction and robustness tests
# Clean data, 5 % Gaussian noise, doubled amplitude, lowest 10 % of the spectrum removed.
 
# %%
def add_noise(x, pct=NOISE_PCT, seed=0):
    g = torch.Generator(device=x.device).manual_seed(seed)
    amp = torch.quantile(x.abs().flatten(1), 0.995, dim=1).view(-1, 1, 1, 1)
    return x + pct * amp * torch.randn(x.shape, generator=g, device=x.device)
 
 
def test_conditions(net, loader, include_lowcut=True):
    conds = {"clean": None, "noisy (5%)": add_noise, "amplitude x2": lambda x: 2 * x}
    if include_lowcut:
        conds["no low freq (0-10% removed)"] = lowcut_filter
    res = {}
    print(f"{'condition':32s} {'MSE':>12s} {'PSNR(dB)':>9s} {'SSIM':>7s}")
    for name, tf in conds.items():
        res[name] = evaluate(net, loader, tf)
        r = res[name]
        print(f"{name:32s} {r['mse']:12.1f} {r['psnr']:9.2f} {r['ssim']:7.3f}")
    return res
 
 
def predict_time(net, ds):
    x = ds[0][0].unsqueeze(0).to(DEV)
    net.eval()
    with torch.no_grad():
        net(x)
        if DEV.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5): net(x)
        if DEV.type == "cuda": torch.cuda.synchronize()
    return (time.time() - t0) / 5
 
 
def plot_predictions(net, ds, nshow=3, title="", extra_rows=None):
    nshow = min(nshow, len(ds))
    xs = torch.stack([ds[i][0] for i in range(nshow)]).to(DEV)
    truth = np.stack([ds.v[i] for i in range(nshow)])
    net.eval()
    with torch.no_grad():
        pred = denorm(net(xs)).cpu().numpy()[:, 0]
        pred_n = denorm(net(add_noise(xs))).cpu().numpy()[:, 0]
    rows = [("Ground truth", truth), ("Prediction", pred), ("Prediction (5% noise)", pred_n)]
    if extra_rows:
        rows += extra_rows
    fig, axs = plt.subplots(len(rows), nshow, figsize=(3.6 * nshow, 2.3 * len(rows)), squeeze=False)
    for r, (t, arr) in enumerate(rows):
        for c in range(nshow):
            im = axs[r, c].imshow(arr[c], cmap="viridis", vmin=1500, vmax=4500, extent=[0, 3, 2, 0], aspect="auto")
            axs[r, c].set_title(t, fontsize=8); axs[r, c].tick_params(labelsize=6)
    fig.colorbar(im, ax=axs, shrink=0.6, label="Velocity (m/s)")
    fig.suptitle(title)
    return fig, truth, pred
 
 
def plot_profiles(truth, pred, fwi=None, title=""):
    xi_list, n = [90, 200], len(truth)
    fig, axs = plt.subplots(len(xi_list), n, figsize=(3.4 * n, 3.6 * len(xi_list)), squeeze=False)
    z = np.arange(NZ) * DH
    for r, xi in enumerate(xi_list):
        for c in range(n):
            a = axs[r, c]
            a.plot(truth[c][:, xi], z, "g", label="ground truth")
            a.plot(pred[c][:, xi], z, "b", label="prediction")
            if fwi is not None and c < len(fwi):
                a.plot(fwi[c][:, xi], z, "r", label="FWI")
            a.invert_yaxis(); a.set_title(f"x = {xi * DH:.0f} m", fontsize=8)
            a.set_xlabel("Velocity (m/s)", fontsize=7); a.set_ylabel("Depth (m)", fontsize=7)
            if r == 0 and c == 0: a.legend(fontsize=6)
    fig.suptitle(title); fig.tight_layout()
    return fig
 
 
# %% --- RUN: evaluate the trained network on the simulated test set ---
test_loader = DataLoader(test_ds, batch_size=CFG["batch"])
print(f"Testing on {len(test_ds)} unseen velocity models\n")
res_sim = test_conditions(net_sim, test_loader)
pred_time_sim = predict_time(net_sim, test_ds)
print(f"\nPrediction time for ONE model: {pred_time_sim:.3f} s   |   training time: {train_time_sim / 60:.2f} min")
fig, truth_sim, pred_sim = plot_predictions(net_sim, test_ds, 3, "Simulated models: ground truth vs prediction")
show(fig, "06_predictions_sim")
fig = plot_profiles(truth_sim, pred_sim, None, "Vertical velocity profiles (simulated models)")
show(fig, "07_profiles_sim")
 
# %% --- RUN: scatter of predicted vs true velocity over the whole test set ---
net_sim.eval()
P, T_ = [], []
with torch.no_grad():
    for x, y in test_loader:
        P.append(denorm(net_sim(x.to(DEV))).cpu().numpy().ravel()[::40]); T_.append(denorm(y).numpy().ravel()[::40])
P, T_ = np.concatenate(P), np.concatenate(T_)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
ax.scatter(T_, P, s=2, alpha=0.3); ax.plot([1500, 4600], [1500, 4600], "r--")
ax.set_xlabel("True velocity (m/s)"); ax.set_ylabel("Predicted velocity (m/s)")
ax.set_title(f"Pixel-wise agreement, corr = {np.corrcoef(T_, P)[0, 1]:.3f}")
show(fig, "08_scatter_pred_vs_true")
 

# Step 9 - Simplified FWI baseline (for comparison and timing)
# Multiscale time-domain FWI, initial model = Gaussian-smoothed true model (as in the paper),
# gradient by autograd through the FD solver (adjoint-equivalent), Adam updates.
 
# %%
def lowpass_time(x, dt, fc):
    Xf = torch.fft.rfft(x, dim=1)
    f = torch.fft.rfftfreq(x.shape[1], dt).to(x.device)
    m = torch.clamp((1.2 * fc - f) / (0.2 * fc), 0, 1).view(1, -1, 1)
    return torch.fft.irfft(Xf * m, n=x.shape[1], dim=1)
 
 
def run_fwi(mod, v_true, iters, shots_per_iter=3, sigma=10, lr=15.0, bands=(5., 10., 15., 21.)):
    t0 = time.time()
    with torch.no_grad():
        d_obs = mod.simulate(torch.tensor(v_true, device=DEV))
    v0 = gaussian_filter(v_true, sigma)
    v = torch.tensor(v0, device=DEV).requires_grad_(True)
    dts, rng = mod.dt * mod.ds, np.random.default_rng(0)
    losses = []
    for fc in bands:
        opt = torch.optim.Adam([v], lr=lr)
        dobs_f = lowpass_time(d_obs, dts, fc)
        for _ in range(iters):
            sid = sorted(rng.choice(mod.S, size=min(shots_per_iter, mod.S), replace=False))
            syn = mod.simulate(v, sid, grad=True)
            loss = ((lowpass_time(syn, dts, fc) - dobs_f[sid]) ** 2).mean() * 1e6
            opt.zero_grad(); loss.backward()
            v.grad = F.avg_pool2d(v.grad.view(1, 1, *v.grad.shape), 5, 1, 2).view_as(v.grad)
            opt.step()
            with torch.no_grad(): v.clamp_(1500.0, 4600.0)
            losses.append(loss.item())
    return v0, v.detach().cpu().numpy(), time.time() - t0, losses
 
 
# %% --- RUN: FWI on the first test models and comparison with the network ---
nf = min(CFG["fwi_models"], len(test_ds), len(truth_sim))
fwi_init, fwi_inv, fwi_t = [], [], []
for i in range(nf):
    print(f"FWI on test model {i + 1}/{nf} ...", flush=True)
    a, b, t, ls = run_fwi(modeler, truth_sim[i], CFG["fwi_iters"])
    fwi_init.append(a); fwi_inv.append(b); fwi_t.append(t)
    print(f"  FWI time {t:.0f} s, final data misfit {ls[-1]:.4g}")
if nf:
    mse_fwi = [float(np.mean((fwi_inv[i] - truth_sim[i]) ** 2)) for i in range(nf)]
    mse_net = [float(np.mean((pred_sim[i] - truth_sim[i]) ** 2)) for i in range(nf)]
    print(f"\nMSE vs truth  | network: {np.mean(mse_net):.0f}   FWI: {np.mean(mse_fwi):.0f}")
    print(f"Time per model | network prediction: {pred_time_sim:.3f} s   FWI: {np.mean(fwi_t):.0f} s "
          f"(speed-up x{np.mean(fwi_t) / max(pred_time_sim, 1e-9):.0f}; network training "
          f"{train_time_sim / 60:.1f} min once)")
    pad_ = lambda arr: np.concatenate([np.array(arr), np.full((3 - len(arr),) + arr[0].shape, np.nan)])[:len(truth_sim)] \
        if len(arr) < len(truth_sim) else np.array(arr)
    fig, _, _ = plot_predictions(net_sim, test_ds, 3, "Network vs FWI (simulated models)",
                                 extra_rows=[("FWI initial model", pad_(fwi_init)), ("FWI result", pad_(fwi_inv))])
    show(fig, "09_network_vs_fwi")
    fig = plot_profiles(truth_sim, pred_sim, fwi_inv, "Vertical profiles: truth / network / FWI")
    show(fig, "10_profiles_with_fwi")
else:
    print("fwi_models = 0 -> FWI skipped")
 

# Step 10 - Missing low frequencies: retrain on high-pass data (paper Fig. 22)
 
# %%
# (kept short: reuses train() with lowcut=True and shows the comparison)
net_lc, hist_lc, tt_lc = train(DATA_DIR, "fcnvmb_sim_lowcut.pt", max(1, CFG["epochs"] // 2), lowcut=True,
                               title="Low-frequency-missing data")
test_ds_lc = SeisDataset(DATA_DIR, "test", lowcut=True)
r_lc = evaluate(net_lc, DataLoader(test_ds_lc, batch_size=CFG["batch"]))
r_nolc_net = res_sim["no low freq (0-10% removed)"]
print(f"\nHigh-pass test data | network trained on full data : PSNR {r_nolc_net['psnr']:.2f} dB, SSIM {r_nolc_net['ssim']:.3f}")
print(f"High-pass test data | network trained on high-pass  : PSNR {r_lc['psnr']:.2f} dB, SSIM {r_lc['ssim']:.3f}")
fig, _, _ = plot_predictions(net_lc, test_ds_lc, 3, "Prediction when low frequencies are missing (trained on high-pass data)")
show(fig, "11_lowfreq_missing")
 

# Step 11 - Optional: SEG salt models with transfer learning
# Set SEG_DIR in the first cell to a folder with the SEG salt velocity models
# (from the authors' FCNVMB-data.zip), then re-run this cell.
 
# %%
def prepare_seg(seg_dir, modeler, scale):
    models = load_external_models(seg_dir)
    if len(models) < 2:
        print("!! No usable 201x301 velocity models found in SEG_DIR.")
        return None
    rng = np.random.default_rng(SEED + 1)
    models = [models[i] for i in rng.permutation(len(models))]
    ntr = min(CFG["seg_train"], max(1, len(models) - CFG["seg_test"]))
    nte = min(CFG["seg_test"], len(models) - ntr)
    print(f"SEG models found: {len(models)} -> {ntr} for training, {nte} for testing")
    out = os.path.join(DATA_DIR, "seg")
    build_split("train", models[:ntr], out, modeler, scale)
    build_split("test", models[ntr:ntr + nte], out, modeler, scale)
    json.dump(dict(scale=scale), open(os.path.join(out, "meta.json"), "w"))
    return out
 
 
# %% --- RUN: SEG salt stage (skipped when SEG_DIR is empty) ---
if SEG_DIR:
    seg_folder = prepare_seg(SEG_DIR, modeler, SCALE)
    if seg_folder:
        net_seg, hist_seg, tt_seg = train(seg_folder, "fcnvmb_seg.pt", CFG["seg_epochs"],
                                          init_ckpt=os.path.join(OUT_DIR, "fcnvmb_sim.pt"))
        show(plot_history(hist_seg, "SEG salt models (transfer learning)"), "12_training_curves_seg")
        seg_test = SeisDataset(seg_folder, "test")
        test_conditions(net_seg, DataLoader(seg_test, batch_size=CFG["batch"]))
        fig, t_seg, p_seg = plot_predictions(net_seg, seg_test, 3, "SEG salt models: truth vs prediction")
        show(fig, "13_predictions_seg")
        show(plot_profiles(t_seg, p_seg, None, "Vertical profiles (SEG salt)"), "14_profiles_seg")
else:
    print("SEG_DIR is empty -> SEG stage skipped (set SEG_DIR = 'path/to/SEG/models' to enable).")
 

# Step 12 - Final summary
 
# %%
summary = dict(preset=PRESET, train_models=CFG["n_train"], test_models=CFG["n_test"], shots=CFG["shots"],
               epochs=CFG["epochs"], train_time_min=train_time_sim / 60, predict_s=pred_time_sim,
               results_simulated=res_sim)
json.dump(summary, open(os.path.join(OUT_DIR, "summary.json"), "w"), indent=2)
print("======== FINAL SUMMARY ========")
print(f"Training models: {CFG['n_train']} | Testing models: {CFG['n_test']} | Shots/model: {CFG['shots']}")
print(f"Epochs: {CFG['epochs']} | Train time: {train_time_sim / 60:.2f} min | Prediction: {pred_time_sim:.3f} s/model")
print(f"Clean test PSNR {res_sim['clean']['psnr']:.2f} dB, SSIM {res_sim['clean']['ssim']:.3f}")
print(f"Figures saved in  : {os.path.abspath(FIG_DIR)}")
print(f"Models/metrics in : {os.path.abspath(OUT_DIR)}")
 