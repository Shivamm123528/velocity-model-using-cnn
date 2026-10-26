"""Export real notebook output arrays and checkpoint into the app. Run from repository root."""
import argparse,shutil,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument("run_dir",type=Path);args=p.parse_args()
root=Path(__file__).resolve().parents[1];r=args.run_dir
ck=r/"results/fcnvmb_sim.pt"
if not ck.exists():raise SystemExit(f"Missing {ck}")
shutil.copy2(ck,root/"models/fcnvmb_sim.pt")
scale=json.loads((r/"data/meta.json").read_text())["scale"]
data=np.load(r/"data/test_data.npy",mmap_mode="r",allow_pickle=False)
truth=np.load(r/"data/test_vel.npy",mmap_mode="r",allow_pickle=False)
for i in range(min(3,len(data))):
    np.savez_compressed(root/f"data/notebook_{i}.npz",seismic=np.asarray(data[i],dtype=np.float32),
        velocity=truth[i],metadata=json.dumps(dict(preprocessing="checkpoint_scaled",scale=scale,
        dx_m=10,dt_s=.004,source_hz=25,shots=5,receivers=301,time_samples=250,source="original notebook test set")))
print("Checkpoint and test examples exported.")
