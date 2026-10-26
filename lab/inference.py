import time
from pathlib import Path
import numpy as np
MODEL_PATH=Path(__file__).resolve().parents[1]/"models/fcnvmb_sim.pt"
def load_model():
    import torch
    from .network import FCNVMB
    checkpoint=torch.load(MODEL_PATH,map_location="cpu",weights_only=True)
    if not isinstance(checkpoint,dict) or not all(k in checkpoint for k in ["state","in_ch","base","scale"]):
        raise ValueError("Checkpoint requires state, in_ch, base and scale.")
    if checkpoint["in_ch"]!=5: raise ValueError("Checkpoint must accept five shots.")
    scale=float(checkpoint["scale"])
    if not np.isfinite(scale) or scale<=0:raise ValueError("Invalid checkpoint input scaling.")
    base=int(checkpoint["base"])
    if base not in [8,16,32,64]:raise ValueError("Unsupported network width.")
    net=FCNVMB(5,base=base);net.load_state_dict(checkpoint["state"],strict=True);net.eval()
    return net,scale
def predict(net,scale,data,preprocessing="raw"):
    import torch
    if data.shape!=(5,250,301): raise ValueError("Expected five shots, 250 samples, 301 receivers.")
    if preprocessing not in ["raw","checkpoint_scaled"]:raise ValueError("Unknown preprocessing.")
    x=torch.from_numpy(np.ascontiguousarray(data*(scale if preprocessing=="raw" else 1))).unsqueeze(0)
    start=time.perf_counter()
    with torch.inference_mode(): pred=net(x).squeeze().numpy()*3000+1500
    return pred.astype(np.float32),time.perf_counter()-start
