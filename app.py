from pathlib import Path
import csv, io, json, time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from lab import auth, store
from lab.authors import render as render_authors
from lab.v2_adapter import (configuration, make_earth, acquisition, time_depth, read_well,
    simulate_earth, load_trained, predict_map, map_scores, rank_candidates, REFERENCES)

ROOT=Path(__file__).parent
MODEL_PATH=ROOT/"models/fcnvmb_sim.pt"
st.set_page_config(page_title="Seismic Velocity Modelling",page_icon="🌍",layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1500px}
h1,h2,h3{letter-spacing:-.025em}
.hero{padding:2.5rem;background:linear-gradient(120deg,#203c4a,#496365);border-radius:20px;color:white;margin:1rem 0 2rem}
.hero h1{color:white;font-size:3rem}.hero p{color:#e4eeee}
[data-testid="stMetricValue"]{color:#c36858}
</style>""",unsafe_allow_html=True)
PAGES=["Home","Model Explorer","Seismic Acquisition","Velocity Prediction","Results and Error Analysis",
       "Time–Depth and Well Comparison","Robustness Lab","Candidate Least Squares",
       "Theory and Methodology","My Experiments","Profile and Settings","Authors"]
if "_pending_nav" in st.session_state:st.session_state.nav=st.session_state.pop("_pending_nav")
def goto(page):st.session_state["_pending_nav"]=page;st.rerun()
user=auth.identity()
with st.sidebar:
    st.markdown("### SEISMIC\nVelocity Modelling")
    page=st.radio("Workspace",PAGES,key="nav",label_visibility="collapsed")
    st.divider()
    if user:
        st.caption("Signed in");st.write(user.email)
        if st.button("Log out"):
            try:auth.client().auth.sign_out()
            finally:auth.clear();st.rerun()
    else:st.caption("Guest workspace. Sign in on Home for private cloud saves.")
    st.caption("4 km · up to four geological units · synthetic research")

@st.cache_resource
def cached_checkpoint(signature):return load_trained(MODEL_PATH,"cpu")
def trained():
    if not MODEL_PATH.exists():raise ValueError("Train the revised notebook and copy results/fcnvmb_sim.pt into models/.")
    net,ck=cached_checkpoint((MODEL_PATH.stat().st_mtime_ns,MODEL_PATH.stat().st_size))
    if ck.get("schema")!="fcnvmb_v2":raise ValueError("Use the checkpoint from velocity_modelling_v2.ipynb.")
    return net,ck
def active_config():
    if MODEL_PATH.exists():
        try:return trained()[1]["config"]
        except Exception as e:st.sidebar.caption(str(e))
    return configuration()
C=active_config();A=acquisition(C)
if C.get("quick_test"):
    st.sidebar.warning("Quick-test mode: the tiny training set checks execution, not prediction accuracy.")
current_contract=json.dumps(A,sort_keys=True)
if st.session_state.get("_active_contract")!=current_contract:
    for key in ["sample","prediction","velocity","layer_labels","geology"]:st.session_state.pop(key,None)
    st.session_state._active_contract=current_contract
def chart(fig):
    fig.update_layout(height=410,margin=dict(l=25,r=20,t=50,b=35),paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family="Arial",color="#233044"))
    st.plotly_chart(fig,use_container_width=True)
def heat(v,title,kind="velocity",meta=None):
    a=A if meta is None else meta
    gather=kind=="gather";error=kind=="error"
    y=np.arange(v.shape[0])*(a["dt_s"] if gather else a["dz_m"]/1000)
    x=np.asarray(a["receiver_x_m"])/1000 if gather else np.arange(v.shape[1])*a["dx_m"]/1000
    lim=max(float(np.quantile(np.abs(v),.995)),1e-12)
    opts=dict(zmin=-lim,zmax=lim) if gather or error else dict(zmin=C["velocity_min"],zmax=C["velocity_max"])
    fig=go.Figure(go.Heatmap(z=v,x=x,y=y,colorscale="RdBu_r" if gather or error else "Viridis",**opts))
    fig.update_layout(title=title,xaxis_title="Horizontal distance (km)",yaxis_title="Recording time (s)" if gather else "Depth (km)")
    fig.update_yaxes(autorange="reversed");chart(fig)
def profiles(v,p,x):
    fig=go.Figure()
    for label,a in [("True / synthetic",v),("Predicted",p)]:
        if a is not None:fig.add_scatter(x=a[:,x],y=np.arange(a.shape[0])*C["dz_m"]/1000,name=label,mode="lines")
    fig.update_layout(xaxis_title="P-wave velocity (m/s)",yaxis_title="Depth (km)");fig.update_yaxes(autorange="reversed");chart(fig)
def validate_meta(meta):
    # Require matching physical coordinates and source, not only matching tensor shape.
    keys=["schema","input_shape","model_shape","dx_m","dz_m","dt_s","source_hz","source_x_m",
          "receiver_x_m","source_z_m","receiver_z_m","preprocessing","physics"]
    missing=[k for k in keys if k not in meta]
    if missing:raise ValueError("Acquisition metadata missing: "+", ".join(missing))
    for k in keys:
        if k=="preprocessing":
            if meta[k]!="raw":raise ValueError("Only calibrated raw amplitudes are accepted by this workflow.")
        elif isinstance(A[k],(list,int,float)):
            try:match=np.shape(meta[k])==np.shape(A[k]) and np.allclose(meta[k],A[k],rtol=1e-6,atol=1e-7)
            except (TypeError,ValueError):match=meta[k]==A[k]
            if not match:raise ValueError(f"Checkpoint acquisition mismatch: {k}.")
        elif meta[k]!=A[k]:raise ValueError(f"Checkpoint acquisition mismatch: {k}.")
def load_arrays(payload,name):
    from lab.data import load_research_sample
    d,v,m=load_research_sample(payload,name,A)
    validate_meta(m);return d,v,m
def set_sample(d,v,m,name):
    for key in ["prediction","inference_seconds","robust"]:st.session_state.pop(key,None)
    st.session_state.sample=dict(data=d,truth=v,meta=m,name=name)
def input_picker():
    with st.expander("Load seismic data",expanded="sample" not in st.session_state):
        examples=sorted((ROOT/"data").glob("v2_*.npz"))
        if examples:
            chosen=st.selectbox("Prepared v2 example",examples,format_func=lambda p:p.stem)
            if st.button("Load prepared v2 example"):
                try:
                    d,v,m=load_arrays(chosen.read_bytes(),chosen.name)
                    set_sample(d,v,m,chosen.stem);st.success("Prepared example loaded.")
                except Exception as e:st.error(str(e))
        f=st.file_uploader("Revised NPZ sample (raw seismic, optional velocity, acquisition metadata)",type=["npz"],key="seismic_input")
        st.caption(f"Expected seismic shape {tuple(A['input_shape'])}; velocity shape {tuple(A['model_shape'])}. Old 2 km samples require their original checkpoint.")
        if f and st.button("Validate and load NPZ"):
            try:d,v,m=load_arrays(f.getvalue(),f.name);set_sample(d,v,m,f.name);st.success("Loaded.")
            except Exception as e:st.error(str(e))
        with st.expander("SEG-Y input"):
            from lab.segy import inspect_segy,load_segy,SHOT_FIELDS
            seg=st.file_uploader("Pre-stack SEG-Y",type=["sgy","segy"])
            field=st.selectbox("Shot grouping",list(SHOT_FIELDS));endian=st.selectbox("Byte order",["big","little"])
            if seg:
                try:
                    payload=seg.getvalue();info=inspect_segy(payload,SHOT_FIELDS[field],endian)
                    st.dataframe(info["shots"],use_container_width=True)
                    ids=[r["shot_id"] for r in info["shots"]]
                    selected=st.multiselect(f"Choose {A['shots']} shot IDs",ids,max_selections=A["shots"])
                    st.warning("Field data remain experimental. Matching headers does not validate the synthetic-trained model.")
                    confirm=st.checkbox(f"I verified the {A['source_hz']:g} Hz source wavelet, raw amplitude convention, source/receiver depths and compatible acoustic assumptions.")
                    if st.button("Validate SEG-Y",disabled=len(selected)!=A["shots"] or not confirm):
                        d,v,m=load_segy(payload,selected,SHOT_FIELDS[field],endian,expected=A)
                        m.update({k:A[k] for k in ["schema","input_shape","model_shape","dz_m","source_hz","source_x_m",
                            "receiver_x_m","source_z_m","receiver_z_m","physics"]})
                        set_sample(d,v,m,seg.name);st.success("Headers validated; physical assumptions acknowledged by user.")
                except Exception as e:st.error(str(e))
    s=st.session_state.get("sample")
    if s:
        try:
            validate_meta(s["meta"])
            if list(s["data"].shape)!=A["input_shape"]:raise ValueError("Shape changed.")
        except Exception:
            st.warning("The active sample belongs to another acquisition. Load a matching sample.");return None
        st.caption(f"Active sample: {s['name']} · {A['shots']} shots · {A['time_samples']} samples · {A['receivers']} receivers")
    return s
def metrics(p,v):
    values=map_scores(p,v)
    for col,(k,val) in zip(st.columns(4),values.items()):col.metric(k,f"{val:,.3f}")
    st.caption("MSE: (m/s)²; RMSE: m/s; PSNR: dB. PSNR/SSIM use a fixed 4500 m/s range; SSIM is not percentage accuracy.")
    return values
def pack_sample(s,p=None):
    b=io.BytesIO();arrays=dict(seismic=s["data"],metadata=json.dumps(s["meta"]))
    if s["truth"] is not None:arrays["velocity"]=s["truth"]
    if p is not None:arrays["prediction"]=p
    np.savez_compressed(b,**arrays);return b.getvalue()
def save(s,p=None):
    if not user:st.info("Sign in on Home to save privately.");return
    with st.form("save_experiment"):
        name=st.text_input("Experiment name",s["name"],max_chars=100)
        notes=st.text_area("Research notes",max_chars=4000)
        keep=st.checkbox("Save arrays (Supabase limit: 12 MiB)",True)
        ok=st.form_submit_button("Save experiment")
    if ok:
        try:
            blob=pack_sample(s,p) if keep else None
            if blob and len(blob)>12*1024**2:raise ValueError("Arrays exceed 12 MiB. Uncheck arrays to save metadata only.")
            if not name.strip():raise ValueError("Enter a name.")
            vals=map_scores(p,s["truth"]) if p is not None and s["truth"] is not None else {}
            store.save(user,name.strip(),notes,s["meta"],vals,blob);st.success("Saved privately.")
        except Exception as e:st.error(f"Save failed: {e}")

if page=="Home":
    st.markdown('<div class="hero"><h1>Four kilometres.<br>Up to four units.</h1><p>Build constrained synthetic models, inspect seismic evidence and evaluate every prediction.</p></div>',unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Model depth",f"{(C['nz']-1)*C['dz_m']/1000:g} km")
    b.metric("Default units",C["n_layers"]);c.metric("Checkpoint dataset",f"{C['n_train']:,} train + {C['n_test']:,} test" if C.get("counts_known") else "Counts not recorded")
    st.write("Four units are a project-scale simplification. The real Earth can have more layers; no universal four-layer limit is claimed.")
    st.caption(f"Input {tuple(A['input_shape'])} → output {tuple(A['model_shape'])}; {A['dx_m']:g} m grid, {A['dt_s']*1000:g} ms saved sampling, {A['source_hz']:g} Hz source.")
    st.caption("Use your v2 notebook checkpoint. The quick-test weights demonstrate execution only.")
    if st.button("Explore the geological model",type="primary"):goto("Model Explorer")
    if not user:auth.panel()

elif page=="Model Explorer":
    st.title("Model Explorer")
    with st.form("earth"):
        a,b=st.columns(2);seed=a.number_input("Random seed",0,1000000,17)
        n=b.select_slider("Number of main geological units",options=[2,3,4],value=4)
        generate=st.form_submit_button("Generate velocity model",type="primary")
    if "velocity" not in st.session_state or generate:
        v,labels,info=make_earth(np.random.default_rng(seed),C,n)
        st.session_state.update(velocity=v,layer_labels=labels,geology=info,model_seed=int(seed),model_layers=int(n))
    v=st.session_state.velocity
    if list(v.shape)!=A["model_shape"]:
        st.warning("Regenerate the model after changing the checkpoint.");st.stop()
    a,b=st.columns([2,1])
    with a:heat(v,"Synthetic P-wave velocity")
    with b:
        st.metric("Geological units",st.session_state.geology["n_layers"])
        st.metric("Minimum layer thickness",f"{st.session_state.geology['minimum_thickness_m']:.0f} m")
        st.caption("Non-crossing boundaries; no added salt body. Velocity gradients inside a unit are not extra geological layers.")
    x=st.slider("Vertical profile position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
    profiles(v,None,int(x/C["dx_m"]))
    with st.expander("How was this model designed?"):
        st.write("Use the v2 notebook lithology/gradient generator, with 2–4 units, at least 300 m initial column thickness and at least 200 m interface spacing before optional fault displacement. Interfaces are folded and smoothed; weak lateral heterogeneity is included. No additional salt body is inserted.")
        st.json(st.session_state.geology)
        st.caption("This matches the v2 notebook prior, whose training examples may contain two, three or four units. The lithology labels and velocities are synthetic assumptions, not local field measurements.")
    b=io.BytesIO();np.save(b,v)
    st.download_button("Download velocity map",b.getvalue(),"velocity_model.npy")
    if user and st.button("Bookmark seed and layer count"):
        try:
            pr=store.profile(user);items=pr.get("bookmarks",[])
            item=dict(kind="four_unit_seed",seed=int(seed),layers=int(n))
            if item not in items:items.append(item)
            store.update_profile(user,bookmarks=items[-100:]);st.success("Bookmarked.")
        except Exception as e:st.error(str(e))
    if st.button("Continue to acquisition"):goto("Seismic Acquisition")

elif page=="Seismic Acquisition":
    st.title("Seismic Acquisition");s=input_picker()
    with st.expander("Simulate the current geological model"):
        st.write(f"{C['shots']} sequential shots; {C['nt']} time steps; {C['source_hz']:g} Hz Ricker; {C['nt']*C['dt_sim_s']:g} s duration.")
        st.warning("Simulation on this server uses CPU and can take a long time. Generate the dataset on your GPU notebook and upload an NPZ for routine app use.")
        if st.button("Simulate model on CPU",disabled="velocity" not in st.session_state):
            bar=st.progress(0.)
            try:
                d=simulate_earth(st.session_state.velocity,C,"cpu",bar.progress)
                m={**A,**st.session_state.geology,"seed":st.session_state.model_seed}
                set_sample(d,st.session_state.velocity,m,"Generated model");st.rerun()
            except Exception as e:st.error(str(e))
    if s:
        shot=st.slider("Shot",1,A["shots"],max(1,A["shots"]//2))-1
        heat(s["data"][shot],f"Shot {shot+1}","gather")
        receiver=st.slider("Receiver index",0,A["receivers"]-1,A["receivers"]//2)
        f=go.Figure(go.Scatter(x=np.arange(A["time_samples"])*A["dt_s"],y=s["data"][shot,:,receiver]))
        f.update_layout(xaxis_title="Recording time (s)",yaxis_title="Amplitude");chart(f)
        st.json(A,expanded=False)
        st.download_button("Download sample",pack_sample(s),"seismic_sample.npz")

elif page=="Velocity Prediction":
    st.title("Velocity Prediction");s=input_picker()
    if not MODEL_PATH.exists():st.info("Copy the newly trained checkpoint to models/fcnvmb_sim.pt.")
    if s:
        if s["meta"].get("experimental"):st.warning("Field-data prediction is experimental and has no demonstrated field accuracy.")
        if st.button("Predict velocity",type="primary",disabled=not MODEL_PATH.exists()):
            try:
                validate_meta(s["meta"]);net,ck=trained();start=time.perf_counter()
                p=predict_map(net,ck,s["data"],"cpu")
                st.session_state.prediction=p;st.session_state.inference_seconds=time.perf_counter()-start
                st.success(f"Prediction and preprocessing: {st.session_state.inference_seconds:.3f} s")
            except Exception as e:st.error(str(e))
        if "prediction" in st.session_state:
            heat(st.session_state.prediction,"Predicted P-wave velocity")
            if st.button("Inspect errors and depth profiles"):goto("Results and Error Analysis")

elif page=="Results and Error Analysis":
    st.title("Results and Error Analysis")
    s=st.session_state.get("sample");p=st.session_state.get("prediction")
    if s is None or p is None:st.info("Load a matching sample and predict first.")
    else:
        v=s["truth"]
        if v is not None:
            for col,a,title,kind in zip(st.columns(3),[v,p,p-v],["True","Predicted","Signed error"],["velocity","velocity","error"]):
                with col:heat(a,title,kind)
            values=metrics(p,v)
            lo,hi=st.slider("Depth interval (m)",0,int((C["nz"]-1)*C["dz_m"]),(0,int((C["nz"]-1)*C["dz_m"])),int(C["dz_m"]))
            if hi-lo>=6*C["dz_m"]:st.json(map_scores(p[int(lo/C["dz_m"]):int(hi/C["dz_m"])+1],v[int(lo/C["dz_m"]):int(hi/C["dz_m"])+1]))
            else:st.caption("Choose at least seven depth samples for SSIM.")
            x=st.slider("Profile position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
            profiles(v,p,int(x/C["dx_m"]))
        else:heat(p,"Predicted velocity");st.info("No true velocity: model-domain error cannot be computed.")
        st.download_button("Download prediction and input",pack_sample(s,p),"velocity_result.npz")
        save(s,p)
    st.divider();st.subheader("All-model test report")
    report=st.file_uploader("Upload results/test/per_model.csv from the notebook",type=["csv"])
    if report:
        try:
            rows=list(csv.DictReader(io.StringIO(report.getvalue().decode("utf-8-sig"))))
            vals=np.asarray([float(r["RMSE"]) for r in rows])
            if not len(rows) or not np.isfinite(vals).all():raise ValueError("Invalid report.")
            st.write(f"{len(rows)} models · mean RMSE {vals.mean():.2f} m/s · median {np.median(vals):.2f} · 95th percentile {np.quantile(vals,.95):.2f}")
            chart(go.Figure(go.Histogram(x=vals,nbinsx=40)).update_layout(xaxis_title="RMSE (m/s)",yaxis_title="Model count"))
            st.dataframe(sorted(rows,key=lambda r:float(r["MSE"])),use_container_width=True)
            st.caption("Display every model, including failures. Lowest test error is descriptive; it must not select the trained checkpoint.")
        except Exception as e:st.error(str(e))

elif page=="Time–Depth and Well Comparison":
    st.title("Time–Depth and Well Comparison")
    st.latex(r"t_{2w}(z,x)=2\int_0^z \frac{d\zeta}{v_P(\zeta,x)}")
    st.caption("Vertical two-way time in a column approximation. Dipping reflectors and finite-offset travel paths require ray tracing or wave modelling.")
    s=st.session_state.get("sample");p=st.session_state.get("prediction")
    v=s["truth"] if s is not None else st.session_state.get("velocity")
    if v is None and p is None:st.info("Generate a model or load a sample first.")
    else:
        x=st.slider("Column position (m)",0,int((C["nx"]-1)*C["dx_m"]),int((C["nx"]-1)*C["dx_m"]/2),int(C["dx_m"]))
        ix=int(x/C["dx_m"]);z=np.arange(C["nz"])*C["dz_m"];f=go.Figure();series={}
        for label,a in [("True / synthetic",v),("Predicted",p)]:
            if a is None:continue
            if np.any(a[:,ix]<=0):
                st.warning("Prediction contains nonpositive velocities; time–depth integration is undefined.");continue
            series[label]=time_depth(a[:,ix],C["dz_m"]);f.add_scatter(x=series[label],y=z/1000,name=label)
        f.update_layout(xaxis_title="Vertical two-way time (s)",yaxis_title="Depth (km)");f.update_yaxes(autorange="reversed");chart(f)
        if len(series)==2:st.metric("Vertical time–depth RMSE",f"{np.sqrt(np.mean((series['True / synthetic']-series['Predicted'])**2)):.4f} s")
        rows=["depth_m,"+",".join(series)]+[str(depth)+","+",".join(str(t[i]) for t in series.values()) for i,depth in enumerate(z)]
        st.download_button("Download time–depth CSV","\n".join(rows),"time_depth.csv")
        well=st.file_uploader("Measured well CSV: depth_m,vp_ms (one well per file)",type=["csv"])
        if well:
            try:
                wz,wv=read_well(io.BytesIO(well.getvalue()));f=go.Figure()
                f.add_scatter(x=wv,y=wz/1000,name="Uploaded well")
                if v is not None:f.add_scatter(x=v[:,ix],y=z/1000,name="Synthetic / true model")
                if p is not None:f.add_scatter(x=p[:,ix],y=z/1000,name="Prediction")
                f.update_layout(xaxis_title="P-wave interval velocity (m/s)",yaxis_title="Depth (km)");f.update_yaxes(autorange="reversed");chart(f)
                st.caption("An unrelated well provides a range/trend comparison, not prediction ground truth. Verify origin, datum, units and processing.")
                colocated=st.checkbox("This measured well is co-located with this seismic prediction and uses the same depth datum.")
                if colocated and p is not None:
                    mask=(wz>=z[0])&(wz<=z[-1])
                    if mask.sum()<2:raise ValueError("Insufficient overlapping well samples.")
                    pp=np.interp(wz[mask],z,p[:,ix]);st.metric("Well-sample velocity RMSE",f"{np.sqrt(np.mean((pp-wv[mask])**2)):.2f} m/s")
            except Exception as e:st.error(str(e))

elif page=="Robustness Lab":
    st.title("Robustness Lab");s=input_picker()
    if s:
        noise=st.slider("Noise as fraction of sample 99.5th amplitude percentile",0.,.25,.05,.01)
        amplitude=st.slider("Amplitude multiplier",.25,3.,1.,.05)
        cutoff=st.slider("Remove frequencies below (Hz)",0.,20.,0.,.5)
        if st.button("Run robustness test"):
            try:
                d=s["data"].copy()*amplitude
                if noise:d+=np.random.default_rng(0).normal(0,np.quantile(np.abs(d),.995)*noise,d.shape).astype(np.float32)
                if cutoff:
                    freq=np.fft.rfft(d,axis=1);freq[:,np.fft.rfftfreq(d.shape[1],A["dt_s"])<cutoff,:]=0
                    d=np.fft.irfft(freq,n=d.shape[1],axis=1).astype(np.float32)
                net,ck=trained();p=predict_map(net,ck,d);heat(p,"Perturbed-input prediction")
                if s["truth"] is not None:metrics(p,s["truth"])
            except Exception as e:st.error(str(e))

elif page=="Candidate Least Squares":
    st.title("Candidate Least Squares");s=input_picker()
    st.write("Compare candidate velocity models by re-simulating their waveforms and computing the residual against the observed gathers.")
    st.latex(r"J(v)=\frac{1}{N}\sum_i\left[F(v)_i-d_i\right]^2")
    st.caption("The lowest residual ranks the supplied candidates. It does not establish a unique Earth model. Source wavelet, geometry and amplitude conventions must match.")
    candidates=st.file_uploader("Candidate velocity NPY maps",type=["npy"],accept_multiple_files=True)
    if s and candidates:
        agree=st.checkbox("Run the expensive CPU forward simulations for these candidates.")
        if st.button("Rank candidates",disabled=not agree):
            try:
                models=[]
                for f in candidates:
                    if f.size>4*1024**2:raise ValueError("Candidate file exceeds 4 MiB.")
                    v=np.load(io.BytesIO(f.getvalue()),allow_pickle=False)
                    if list(v.shape)!=A["model_shape"]:raise ValueError("Candidate grid mismatch.")
                    models.append((f.name,v))
                rows=rank_candidates(s["data"],models,C,"cpu");st.dataframe(rows,use_container_width=True)
                st.download_button("Download residual ranking",json.dumps(rows,indent=2),"candidate_residuals.json")
            except Exception as e:st.error(str(e))

elif page=="Theory and Methodology":
    st.title("Theory and Methodology")
    st.markdown("""
1. **Geology:** 4 km depth, two to four main geological units, with notebook thickness constraints; no extra salt body.
2. **Velocity prior:** illustrative increasing sedimentary interval velocities with small within-unit depth gradients. Local well calibration is optional in the notebook.
3. **Physics:** 2-D constant-density acoustics, eighth-order space/second-order time, side/bottom sponge. No calibrated free surface, attenuation, elasticity or anisotropy.
4. **Acquisition:** checkpoint metadata defines source and receiver positions, depths, source frequency, record length and sampling. The quick-test checkpoint uses 2 shots, 3.5 Hz and a 7.5 s record.
5. **Learning:** a modified U-Net maps shot gathers to a velocity grid. The v2 notebook crops the final feature map when large enough and interpolates otherwise. This learned projection is not physical time–depth conversion.
6. **Split:** the quick FCN has 2 training maps and 1 test map for execution testing only. The separate quick time–depth regression task has 170 training columns and 30 test columns. The v2 notebook’s full TD benchmark has 17,000/3,000 point samples; these are not FCN velocity-map counts.
7. **Evaluation:** save every test prediction and its error, depth-dependent error and vertical time–depth comparison. Report aggregate and failure statistics.
8. **Reality check:** real Earth can have more than four layers. Marmousi2 and OpenFWI are synthetic benchmarks. Field claims require measured seismic and independent well/velocity constraints.
""")
    st.json(A,expanded=False)
    for title,url in REFERENCES:st.link_button(title,url)
    st.caption("The numerical priors are project design choices; they are not tables copied from these references.")

elif page=="My Experiments":
    st.title("My Experiments")
    if not user:auth.panel()
    else:
        try:
            records=store.listing()
            if not records:st.info("No saved experiments.")
            else:
                st.dataframe([{k:r[k] for k in ["name","created_at","notes"]} for r in records],use_container_width=True)
                selected=st.selectbox("Experiment",records,format_func=lambda r:r["name"]+" "+r["created_at"][:16])
                st.json(selected["metrics"]);st.json(selected["settings"],expanded=False)
                if selected.get("object_path") and st.button("Reopen arrays"):
                    d,v,m=load_arrays(store.download(selected),"saved.npz");set_sample(d,v,m,selected["name"]);goto("Velocity Prediction")
                with st.form("notes"):
                    notes=st.text_area("Notes",selected["notes"],max_chars=4000);ok=st.form_submit_button("Update notes")
                if ok:auth.client().table("experiments").update({"notes":notes}).eq("id",selected["id"]).execute();st.success("Updated.")
                confirm=st.checkbox("Delete selected experiment and its arrays")
                if st.button("Delete",disabled=not confirm):store.delete(selected);st.rerun()
        except Exception as e:st.error(f"Could not load saved experiments: {e}")

elif page=="Profile and Settings":
    st.title("Profile and Settings")
    if not user:auth.panel()
    else:
        try:
            pr=store.profile(user)
            with st.form("profile"):
                name=st.text_input("Display name",pr["display_name"],max_chars=80)
                tips=st.checkbox("Show study tips",pr.get("preferences",{}).get("show_tips",True))
                ok=st.form_submit_button("Save profile")
            if ok:store.update_profile(user,display_name=name,preferences={"show_tips":tips});st.success("Saved.")
            for i,item in enumerate(pr.get("bookmarks",[])):
                if item.get("kind")!="four_unit_seed":continue
                if st.button(f"Seed {item['seed']} · {item['layers']} units",key=f"bookmark_{i}"):
                    v,labels,info=make_earth(np.random.default_rng(item["seed"]),C,item["layers"])
                    st.session_state.update(velocity=v,layer_labels=labels,geology=info,model_seed=item["seed"],model_layers=item["layers"]);goto("Model Explorer")
            st.caption("Supabase Auth manages passwords. Passwords are never saved to application profile tables.")
        except Exception as e:st.error(str(e))
elif page=="Authors":render_authors(ROOT)
st.divider();st.caption("Seismic Velocity Modelling · four-unit synthetic study · field performance unverified")
