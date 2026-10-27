"""SEG-Y inspection and strict acquisition-aware conversion. Never silently resize."""
from contextlib import contextmanager
import tempfile
from pathlib import Path
import numpy as np
from .data import numeric

MAX_SEGY_BYTES=200*1024**2
SHOT_FIELDS={"Field record (byte 9)":9,"Energy source point (byte 17)":17}
LEGACY=dict(input_shape=[5,250,301],model_shape=[201,301],shots=5,time_samples=250,receivers=301,
            dx_m=10.,dt_s=.004,source_x_m=[0,750,1500,2250,3000],receiver_x_m=list(range(0,3001,10)))
@contextmanager
def opened(payload,endian="big"):
    if not 3600<=len(payload)<=MAX_SEGY_BYTES:raise ValueError("SEG-Y must be between 3600 bytes and 200 MiB.")
    try:import segyio
    except ImportError as e:raise ValueError("Install segyio from requirements.txt.") from e
    with tempfile.TemporaryDirectory(prefix="velocity_segy_") as folder:
        path=Path(folder)/"input.sgy";path.write_bytes(payload)
        try:handle=segyio.open(str(path),"r",ignore_geometry=True,endian=endian)
        except Exception as e:raise ValueError("Cannot read SEG-Y. Check byte order and supported trace format.") from e
        with handle as f:
            if f.tracecount>100000 or len(f.samples)>100000:raise ValueError("Inspection limits exceeded.")
            yield f
def attr(f,field):return np.asarray(f.attributes(field)[:]).copy()
def inspect_segy(payload,shot_field=9,endian="big"):
    with opened(payload,endian) as f:
        ids,counts=np.unique(attr(f,shot_field),return_counts=True)
        return dict(traces=int(f.tracecount),samples_per_trace=len(f.samples),binary_dt_us=int(f.bin[3217]),
                    shots=[dict(shot_id=int(i),traces=int(n)) for i,n in zip(ids,counts)])
def scalar(values):
    values=np.asarray(values,dtype=float)
    return np.where(values>0,values,1/np.maximum(np.abs(values),1))
def convert_arrays(traces,h,selected,measurement_system=1,expected=None):
    a=LEGACY if expected is None else expected;selected=list(selected)
    if len(selected)!=a["shots"] or len(set(selected))!=a["shots"]:
        raise ValueError(f"Select exactly {a['shots']} distinct shot IDs.")
    if measurement_system not in (1,2):raise ValueError("Identify metres or feet in the binary header.")
    factor=1. if measurement_system==1 else .3048
    scale=scalar(h["scalar"])*factor;sx,sy=h["sx"]*scale,h["sy"]*scale;gx,gy=h["gx"]*scale,h["gy"]*scale
    if traces.ndim!=2 or traces.shape[1]!=a["time_samples"]:
        raise ValueError(f"Exactly {a['time_samples']} samples per trace required.")
    groups=[]
    for shot in selected:
        idx=np.flatnonzero(h["shot"]==shot)
        if len(idx)!=a["receivers"]:raise ValueError(f"Shot {shot}: exactly {a['receivers']} receivers required.")
        if not np.all(h["units"][idx]==1):raise ValueError("Coordinates must use length units.")
        if not np.all(h["dt"][idx]==round(a["dt_s"]*1e6)) or not np.all(h["delay"][idx]==0):
            raise ValueError("Sampling interval or recording delay does not match checkpoint.")
        if not np.allclose(sx[idx],sx[idx[0]],atol=.1,rtol=0) or not np.allclose(sy[idx],sy[idx[0]],atol=.1,rtol=0):
            raise ValueError("Shot group mixes source locations.")
        idx=idx[np.argsort(gx[idx],kind="stable")]
        target=np.asarray(a["receiver_x_m"])
        if not np.allclose(gx[idx]-gx[idx[0]],target-target[0],atol=.1,rtol=0):
            raise ValueError("Receiver locations do not match checkpoint.")
        if not np.allclose(gy[idx],sy[idx[0]],atol=.1,rtol=0):raise ValueError("Only a straight 2-D line parallel to X is supported.")
        groups.append((float(sx[idx[0]]),idx))
    groups.sort(key=lambda item:item[0]);ref=groups[0][1];origin=gx[ref[0]]
    for j,(source,idx) in enumerate(groups):
        if not np.allclose(gx[idx],gx[ref],atol=.1,rtol=0) or not np.allclose(gy[idx],gy[ref],atol=.1,rtol=0):
            raise ValueError("All shots must share a fixed receiver spread.")
        if not np.isclose(source-origin,a["source_x_m"][j]-a["receiver_x_m"][0],atol=.1,rtol=0):
            raise ValueError("Source locations do not match checkpoint.")
    return numeric(np.stack([traces[idx].T for _,idx in groups]),tuple(a["input_shape"]),"SEG-Y seismic")
def load_segy(payload,selected,shot_field=9,endian="big",expected=None):
    a=LEGACY if expected is None else expected
    with opened(payload,endian) as f:
        all_ids=attr(f,shot_field);indices=np.flatnonzero(np.isin(all_ids,selected))
        if len(indices)!=a["shots"]*a["receivers"]:raise ValueError("Selected shots have the wrong trace count.")
        fields=dict(scalar=71,sx=73,sy=77,gx=81,gy=85,units=89,dt=117,delay=109)
        h={k:attr(f,v)[indices] for k,v in fields.items()};h["shot"]=all_ids[indices]
        traces=np.stack([f.trace[int(i)].copy() for i in indices])
        d=convert_arrays(traces,h,selected,int(f.bin[3255]),a)
    return d,None,dict(dx_m=a["dx_m"],dt_s=a["dt_s"],shots=a["shots"],receivers=a["receivers"],
        time_samples=a["time_samples"],preprocessing="raw",input_format="SEG-Y",experimental=True,
        shot_ids=[int(x) for x in selected],shot_header_byte=shot_field,byte_order=endian,
        note="Header geometry validated. Source wavelet, elevations, amplitudes and field accuracy require independent verification.")
