"""Bounded, pickle-free scientific array loading."""
import io, json, zipfile
from pathlib import Path
import numpy as np
INPUT_SHAPE = (5, 250, 301)
MODEL_SHAPE = (201, 301)
MAX_BYTES = 12 * 1024**2
DEFAULT_META = dict(dx_m=10., dt_s=.004, source_hz=25., shots=5,
                    receivers=301, time_samples=250, preprocessing="raw")
def numeric(a, shape, name):
    if not isinstance(a, np.ndarray) or a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if a.dtype.kind not in "fiu" or not np.isfinite(a).all():
        raise ValueError(f"{name} must contain finite real numbers.")
    with np.errstate(over="ignore", invalid="ignore"):
        out=a.astype(np.float32)
    if not np.isfinite(out).all(): raise ValueError(f"{name} exceeds float32 range.")
    return out
def load_sample(payload, name):
    if len(payload)>MAX_BYTES: raise ValueError("Maximum upload size is 12 MiB.")
    if name.lower().endswith(".npz"):
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            if len(z.infolist())>8 or sum(x.file_size for x in z.infolist())>MAX_BYTES:
                raise ValueError("Expanded archive exceeds the limit.")
        with np.load(io.BytesIO(payload),allow_pickle=False) as z:
            if "seismic" not in z: raise ValueError("NPZ requires a seismic array.")
            d=numeric(z["seismic"],INPUT_SHAPE,"seismic")
            v=numeric(z["velocity"],MODEL_SHAPE,"velocity") if "velocity" in z else None
            meta=json.loads(str(z["metadata"].item())) if "metadata" in z else {}
            if not isinstance(meta,dict): raise ValueError("metadata must be a JSON object.")
    elif name.lower().endswith(".npy"):
        d=numeric(np.load(io.BytesIO(payload),allow_pickle=False),INPUT_SHAPE,"seismic")
        v,meta=None,{}
    else: raise ValueError("Use .npy or .npz.")
    return d,v,meta
def pack(d,v=None,pred=None,meta=None):
    b=io.BytesIO(); values={"seismic":d,"metadata":json.dumps(meta or {})}
    if v is not None: values["velocity"]=v
    if pred is not None: values["prediction"]=pred
    np.savez_compressed(b,**values);return b.getvalue()
def transform(d, noise=0., amplitude=1., cutoff=0., seed=0, mode="Hz"):
    out=d.astype(np.float32).copy()*amplitude
    if noise:
        # Notebook add_noise quantiles each entire multi-shot sample.
        sigma=float(np.quantile(np.abs(out),.995))*noise
        out += np.random.default_rng(seed).normal(0,sigma,out.shape).astype(np.float32)
    if cutoff:
        f=np.fft.rfft(out,axis=1)
        if mode=="Notebook bins": f[:,:int(f.shape[1]*cutoff),:]=0
        else: f[:,np.fft.rfftfreq(out.shape[1],.004)<cutoff,:]=0
        out=np.fft.irfft(f,n=out.shape[1],axis=1).astype(np.float32)
    return out
