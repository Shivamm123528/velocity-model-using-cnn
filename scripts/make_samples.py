"""Generate physically simulated examples. This does not create network predictions."""
import sys,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from lab.generator import make_velocity_model
from lab.physics import simulate
for seed in [17,29]:
    v=make_velocity_model(np.random.default_rng(seed))
    print("Simulating seed",seed,flush=True);d=simulate(v)
    np.savez_compressed(root/f"data/sample_{seed}.npz",seismic=d,velocity=v,
       metadata=json.dumps(dict(seed=seed,preprocessing="raw",dx_m=10,dt_s=.004,source_hz=25,
             shots=5,time_samples=250,receivers=301,source="NumPy notebook-stencil port; generated for app")))
