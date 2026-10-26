import numpy as np
from .physics import gaussian_filter
NZ,NX,DH=201,301,10.0

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
 