"""App compatibility for velocity_modelling_v2.ipynb checkpoints.

This module performs inference and forward modelling only. It never trains on import.
"""
import csv,io,math,os
from pathlib import Path
import numpy as np
from .v2_runtime import runtime

REFERENCES=[
 ("Telford, Geldart & Sheriff — Applied Geophysics","https://doi.org/10.1017/CBO9781139167932"),
 ("Sheriff & Geldart — Exploration Seismology","https://doi.org/10.1017/CBO9781139168359"),
 ("Yang & Ma (2019) — Deep-learning inversion","https://doi.org/10.1190/geo2018-0249.1"),
 ("OpenFWI — synthetic benchmark datasets","https://arxiv.org/abs/2111.02926"),
]
PHYSICS="notebook_v2: 2D constant-density acoustic; 8th-order space, 2nd-order time; side/bottom sponge; no physical free surface"

def configuration(**overrides):
    c=dict(schema="fcnvmb_v2",seed=0,n_train=2,n_test=1,counts_known=True,
           nz=101,nx=151,dx_m=40.,dz_m=40.,dt_sim_s=.002,source_hz=3.5,
           nt=3750,ds=10,shots=2,base=8,n_layers=4,min_layer_m=200.,
           velocity_min=1500.,velocity_max=6000.,velocity_offset=1500.,velocity_range=4500.,
           pad=40,epochs=1,batch=1,quick_test=True)
    c.update(overrides)
    for k in ['nz','nx','nt','ds','shots','base','pad']:
        if not isinstance(c[k],int) or c[k]<1:raise ValueError(f"Invalid {k}.")
    if not (17<=c['nz']<=1001 and 17<=c['nx']<=1001 and 1<=c['shots']<=64 and c['nt']<=20000):
        raise ValueError("Unsupported acquisition size.")
    if c['base'] not in [8,16,32,64]:raise ValueError("Unsupported network width.")
    if not np.isclose(c['dx_m'],c['dz_m']):raise ValueError("This notebook uses equal x/z spacing.")
    for k in ['dx_m','dz_m','dt_sim_s','source_hz','velocity_offset','velocity_range']:
        if not np.isfinite(c[k]) or c[k]<=0:raise ValueError(f"Invalid {k}.")
    if not 2<=c['n_layers']<=4:raise ValueError("Choose two to four geological units.")
    return c

def acquisition(c):
    src=np.round(np.linspace(0,c['nx']-1,c['shots'])).astype(int)
    return dict(schema="fcnvmb_v2",input_shape=[c['shots'],math.ceil(c['nt']/c['ds']),c['nx']],
        model_shape=[c['nz'],c['nx']],dx_m=c['dx_m'],dz_m=c['dz_m'],
        dt_s=c['dt_sim_s']*c['ds'],dt_sim_s=c['dt_sim_s'],nt=c['nt'],ds=c['ds'],
        source_hz=c['source_hz'],shots=c['shots'],receivers=c['nx'],time_samples=math.ceil(c['nt']/c['ds']),
        source_x_m=(src*c['dx_m']).tolist(),receiver_x_m=(np.arange(c['nx'])*c['dx_m']).tolist(),
        source_z_m=c['dz_m'],receiver_z_m=c['dz_m'],preprocessing="raw",physics=PHYSICS,
        velocity_offset=c['velocity_offset'],velocity_range=c['velocity_range'],projection="notebook_crop_or_resize")

def load_trained(path,device="cpu"):
    import torch
    ck=torch.load(path,map_location="cpu",weights_only=True)
    required={'state','in_ch','base','scale','out_hw','velocity_offset','velocity_range','acquisition'}
    if not isinstance(ck,dict) or not required<=set(ck):
        raise ValueError("This is not the revised v2 checkpoint. Use fcnvmb_sim.pt from velocity_modelling_v2.ipynb.")
    a=ck['acquisition'];quick=bool(ck.get('quick_test',False));nz,nx=map(int,ck['out_hw'])
    c=configuration(nz=nz,nx=nx,shots=int(ck['in_ch']),base=int(ck['base']),
        dx_m=float(a['dx_m']),dz_m=float(a['dz_m']),dt_sim_s=float(a['dt_sim_s']),
        nt=int(a['nt']),ds=int(a['ds']),source_hz=float(a['source_hz']),
        velocity_offset=float(ck['velocity_offset']),velocity_range=float(ck['velocity_range']),
        quick_test=quick,n_train=2 if quick else 0,n_test=1 if quick else 0,counts_known=quick,
        epochs=int(ck.get('hist',{}).get('epoch',[0])[-1]))
    expected=acquisition(c)
    for k in ['input_shape','model_shape','shots','receivers','nt','ds','dx_m','dz_m','dt_s','dt_sim_s','source_hz','source_x_m','receiver_x_m','source_z_m','receiver_z_m']:
        if k not in a or np.shape(a[k])!=np.shape(expected[k]) or not np.allclose(a[k],expected[k]):
            raise ValueError(f"Checkpoint has inconsistent acquisition metadata: {k}.")
    scale=float(ck['scale'])
    if not np.isfinite(scale) or scale<=0:raise ValueError("Invalid checkpoint amplitude scale.")
    if ck.get('lowcut',False):raise ValueError("Use the full-band fcnvmb_sim.pt; this app does not silently apply low-cut preprocessing.")
    if str(device).startswith('cpu'):torch.set_num_threads(min(2,os.cpu_count() or 1))
    env=runtime(c,device)
    net=env['FCNVMB'](c['shots'],(nz,nx),c['base']).to(device)
    net.load_state_dict(ck['state'],strict=True);net.eval()
    # Expose the app contract while retaining the original notebook metadata.
    ck={**ck,'schema':'fcnvmb_v2','config':c,'acquisition':expected,'notebook_acquisition':a}
    return net,ck

def predict_map(net,ck,d,device="cpu"):
    import torch
    d=np.asarray(d,dtype=np.float32)
    if list(d.shape)!=ck['acquisition']['input_shape'] or not np.isfinite(d).all():
        raise ValueError(f"Expected finite raw seismic with shape {ck['acquisition']['input_shape']}.")
    with torch.inference_mode():
        x=torch.from_numpy(np.ascontiguousarray(d*ck['scale']))[None].to(device)
        p=net(x).cpu().numpy()[0,0]*ck['velocity_range']+ck['velocity_offset']
    if not np.isfinite(p).all():raise ValueError("Network produced nonfinite velocities.")
    return p.astype(np.float32)

def make_earth(rng,c=None,n_layers=None):
    c=configuration() if c is None else c;env=runtime(c)
    if n_layers is not None:
        if n_layers not in [2,3,4]:raise ValueError("Choose two to four units.")
        env['N_LAYERS_MIN']=env['N_LAYERS_MAX']=int(n_layers)
    v,info=env['make_velocity_model'](rng,return_info=True)
    labels=info.pop('layer_labels')
    info.update(prior="v2 notebook lithology/gradient prior; not local field calibration")
    return v,labels,info

def simulate_earth(v,c,device="cpu",progress=None):
    import torch
    v=np.asarray(v,np.float32)
    if v.shape!=(c['nz'],c['nx']) or not np.isfinite(v).all() or v.min()<1500 or v.max()>6000:
        raise ValueError("Candidate must match the notebook grid and 1500–6000 m/s range.")
    cfl=6000*c['dt_sim_s']*np.sqrt(2*(205/72+2*(8/5+1/5+8/315+1/560)))/c['dx_m']
    if cfl>=2:raise ValueError("Unstable simulation settings.")
    if str(device).startswith('cpu'):torch.set_num_threads(min(2,os.cpu_count() or 1))
    env=runtime(c,device);mod=env['Modeler'](c['shots'],c['nt'],c['ds'],torch.device(device),pad=c['pad'])
    if progress:progress(0.)
    with torch.no_grad():d=mod.simulate(torch.tensor(v,device=device)).cpu().numpy()
    if not np.isfinite(d).all():raise ValueError("Nonfinite wavefield.")
    if progress:progress(1.)
    return d

def time_depth(v,dz_m=40.):
    v=np.asarray(v,dtype=np.float64)
    if v.ndim not in [1,2] or not np.isfinite(v).all() or np.any(v<=0):raise ValueError("Positive finite interval velocities required.")
    t=np.zeros_like(v);t[1:]=np.cumsum(dz_m*(1/v[:-1]+1/v[1:]),axis=0);return t

def read_well(source):
    text=source.read() if hasattr(source,'read') else Path(source).read_text(encoding='utf-8-sig')
    if isinstance(text,bytes):text=text.decode('utf-8-sig')
    rows=list(csv.DictReader(io.StringIO(text)))
    if not rows or not {'depth_m','vp_ms'}<=set(rows[0]):raise ValueError("CSV needs depth_m,vp_ms columns, one well per file.")
    z=np.array([float(r['depth_m']) for r in rows]);v=np.array([float(r['vp_ms']) for r in rows]);order=np.argsort(z);z,v=z[order],v[order]
    if len(z)<2 or not np.isfinite(z).all() or not np.isfinite(v).all() or z.min()<0 or np.any(np.diff(z)<=0) or np.any((v<500)|(v>8000)):
        raise ValueError("Well data need unique nonnegative depths and plausible velocities in m/s.")
    return z,v

def map_scores(pred,true):
    from numpy.lib.stride_tricks import sliding_window_view
    p,t=np.asarray(pred,np.float64),np.asarray(true,np.float64)
    if p.shape!=t.shape or p.ndim!=2 or min(p.shape)<7 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Matching finite 2D maps, at least 7 x 7, required.")
    mse=float(np.mean((p-t)**2));avg=lambda a:sliding_window_view(a,(7,7)).mean(axis=(-1,-2))
    a,b=avg(p),avg(t);aa,bb=avg(p*p)-a*a,avg(t*t)-b*b;ab=avg(p*t)-a*b
    c1,c2=(.01*4500)**2,(.03*4500)**2
    s=((2*a*b+c1)*(2*ab+c2))/((a*a+b*b+c1)*(aa+bb+c2))
    return dict(MSE=mse,RMSE=math.sqrt(mse),PSNR=10*math.log10(4500**2/max(mse,1e-9)),SSIM=float(s.mean()))

def rank_candidates(observed,candidates,c,device="cpu"):
    observed=np.asarray(observed,np.float32)
    if list(observed.shape)!=acquisition(c)['input_shape']:raise ValueError("Observed acquisition mismatch.")
    rows=[]
    for name,v in candidates:
        d=simulate_earth(v,c,device);delta=d.astype(np.float64)-observed
        rows.append(dict(name=str(name),waveform_MSE=float(np.mean(delta**2)),relative_L2=float(np.linalg.norm(delta.ravel())/max(np.linalg.norm(observed.ravel()),1e-20))))
    return sorted(rows,key=lambda r:r['waveform_MSE'])
