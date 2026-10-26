"""SEG-Y inspection and strict conversion for the existing synthetic checkpoint."""
from contextlib import contextmanager
import tempfile
from pathlib import Path
import numpy as np
from .data import numeric, INPUT_SHAPE

MAX_SEGY_BYTES = 200 * 1024**2
SHOT_FIELDS = {"Field record (byte 9)": 9, "Energy source point (byte 17)": 17}

@contextmanager
def opened(payload, endian="big"):
    if not 3600 <= len(payload) <= MAX_SEGY_BYTES:
        raise ValueError("SEG-Y must be between 3600 bytes and 200 MiB.")
    try:
        import segyio
    except ImportError as e:
        raise ValueError("Install the updated requirements.txt (segyio is required).") from e
    with tempfile.TemporaryDirectory(prefix="velocity_segy_") as folder:
        path = Path(folder) / "input.sgy"
        path.write_bytes(payload)
        try:
            handle = segyio.open(str(path), "r", ignore_geometry=True, endian=endian)
        except Exception as e:
            raise ValueError("Cannot read SEG-Y. Check byte order; variable-length traces and unsupported formats need external conversion.") from e
        with handle as f:
            if f.tracecount > 100000 or len(f.samples) > 100000:
                raise ValueError("File exceeds inspection limits (100,000 traces/samples).")
            yield f

def attr(f, field):
    return np.asarray(f.attributes(field)[:]).copy()

def inspect_segy(payload, shot_field=9, endian="big"):
    with opened(payload, endian) as f:
        ids, counts = np.unique(attr(f, shot_field), return_counts=True)
        return dict(traces=int(f.tracecount), samples_per_trace=len(f.samples),
                    binary_dt_us=int(f.bin[3217]), shots=[
                    {"shot_id":int(i), "traces":int(n)} for i,n in zip(ids, counts)])

def scalar(values):
    values = np.asarray(values, dtype=float)
    return np.where(values > 0, values, 1 / np.maximum(np.abs(values), 1))

def convert_arrays(traces, h, selected, measurement_system=1):
    """Validate physical coordinates before assembling shots; never resize geometry."""
    selected = list(selected)
    if len(selected) != 5 or len(set(selected)) != 5:
        raise ValueError("Select exactly five distinct shot IDs.")
    if measurement_system not in (1, 2):
        raise ValueError("Binary header must identify metres or feet.")
    factor = 1. if measurement_system == 1 else .3048
    scale = scalar(h["scalar"]) * factor
    sx, sy = h["sx"] * scale, h["sy"] * scale
    gx, gy = h["gx"] * scale, h["gy"] * scale
    if traces.shape[1] != 250:
        raise ValueError("This checkpoint requires exactly 250 samples per trace. Process other record lengths externally.")
    groups = []
    for shot in selected:
        idx = np.flatnonzero(h["shot"] == shot)
        if len(idx) != 301:
            raise ValueError(f"Shot {shot} has {len(idx)} traces; exactly 301 receivers are required.")
        if not np.all(h["units"][idx] == 1):
            raise ValueError("Coordinates must use length units; angular or undefined coordinates are unsupported.")
        if not np.all(h["dt"][idx] == 4000) or not np.all(h["delay"][idx] == 0):
            raise ValueError("All selected traces need 4000 microsecond sampling and zero recording delay.")
        if not np.allclose(sx[idx], sx[idx[0]], atol=.1, rtol=0) or not np.allclose(sy[idx], sy[idx[0]], atol=.1, rtol=0):
            raise ValueError("Shot header groups mix source positions. Choose another shot grouping header.")
        idx = idx[np.argsort(gx[idx], kind="stable")]
        if not np.allclose(np.diff(gx[idx]), 10, atol=.1, rtol=0):
            raise ValueError("Receiver X coordinates must be unique and spaced 10 m apart.")
        if not np.allclose(gy[idx], sy[idx[0]], atol=.1, rtol=0):
            raise ValueError("Only a straight 2-D line parallel to X is supported.")
        groups.append((float(sx[idx[0]]), idx))
    groups.sort(key=lambda item:item[0])
    ref = groups[0][1]
    origin = gx[ref[0]]
    for j, (source, idx) in enumerate(groups):
        if not np.allclose(gx[idx], gx[ref], atol=.1, rtol=0) or not np.allclose(gy[idx], gy[ref], atol=.1, rtol=0):
            raise ValueError("All shots must share the same fixed receiver spread.")
        if not np.isclose(source-origin, j*750, atol=.1, rtol=0):
            raise ValueError("Source positions must be 0, 750, 1500, 2250, 3000 m relative to the first receiver.")
    data = numeric(np.stack([traces[idx].T for _,idx in groups]), INPUT_SHAPE, "SEG-Y seismic")
    return data

def load_segy(payload, selected, shot_field=9, endian="big"):
    with opened(payload, endian) as f:
        all_ids = attr(f, shot_field)
        indices = np.flatnonzero(np.isin(all_ids, selected))
        if len(indices) != 1505:
            raise ValueError("Selected five shots must contain exactly 1505 traces in total.")
        fields = dict(scalar=71,sx=73,sy=77,gx=81,gy=85,units=89,dt=117,delay=109)
        h = {k:attr(f,v)[indices] for k,v in fields.items()}
        h["shot"] = all_ids[indices]
        traces = np.stack([f.trace[int(i)].copy() for i in indices])
        d = convert_arrays(traces, h, selected, int(f.bin[3255]))
    # Unknown source frequency stays missing so the UI requires acknowledgement.
    return d, None, dict(dx_m=10.,dt_s=.004,shots=5,receivers=301,time_samples=250,
        preprocessing="raw", input_format="SEG-Y", experimental=True,
        shot_ids=[int(x) for x in selected], shot_header_byte=shot_field,
        byte_order=endian, source_frequency_verified=False,
        note="Header geometry checked; wavelet, amplitude calibration, elevations and field validity are not verified.")
