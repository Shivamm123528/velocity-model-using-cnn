"""v2 notebook checkpoint compatibility; app.py uses v2_adapter directly."""
from pathlib import Path
import time
from .v2_adapter import load_trained,predict_map
MODEL_PATH=Path(__file__).resolve().parents[1]/"models/fcnvmb_sim.pt"
def load_model():
    net,ck=load_trained(MODEL_PATH,"cpu")
    net._v2_checkpoint=ck
    return net,ck["scale"]
def predict(net,scale,data,preprocessing="raw"):
    import numpy as np
    if not hasattr(net,"_v2_checkpoint"):raise ValueError("Load the v2 checkpoint first.")
    if preprocessing not in ["raw","checkpoint_scaled"]:raise ValueError("Unknown preprocessing.")
    d=np.asarray(data,np.float32)
    if preprocessing=="checkpoint_scaled":d=d/scale
    start=time.perf_counter();p=predict_map(net,net._v2_checkpoint,d,"cpu")
    return p,time.perf_counter()-start
