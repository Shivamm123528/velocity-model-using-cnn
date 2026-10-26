
from pathlib import Path
import io,json,hashlib,time
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from lab import auth,store
from lab.segy import inspect_segy,load_segy,SHOT_FIELDS
from lab.data import load_sample,pack,transform,numeric,DEFAULT_META
from lab.generator import make_velocity_model
from lab.physics import simulate,wavelet
from lab.metrics import scores
from lab.plots import heat,profile,spectrum,finish
from lab.inference import MODEL_PATH,load_model,predict

ROOT=Path(__file__).parent
st.set_page_config(page_title="Seismic Velocity Lab",page_icon="〰",layout="wide")
st.markdown("""<style>
.block-container {padding-top:2rem;max-width:1480px}
h1,h2,h3 {letter-spacing:-.025em}
.hero {padding:2.6rem 2rem;background:linear-gradient(115deg,#223343,#3c4d56);border-radius:16px;color:#fff;margin-bottom:1.5rem}
.hero h1 {color:#fff;font-size:3.2rem;line-height:1.05;max-width:760px}
.hero p {color:#d6e0e2;max-width:700px;font-size:1.1rem}
.eyebrow {color:#efb5a8;font-size:.8rem;letter-spacing:.16em;font-weight:700}
[data-testid="stMetricValue"] {color:#b8554b}
</style>""",unsafe_allow_html=True)
PAGES=["Home","Model Explorer","Seismic Acquisition","Velocity Prediction","Results and Error Analysis",
       "Robustness Lab","FWI Comparison","Theory and Methodology","My Experiments","Profile and Settings"]
if "_pending_nav" in st.session_state:
    st.session_state.nav=st.session_state.pop("_pending_nav")
def goto(page):st.session_state["_pending_nav"]=page;st.rerun()
user=auth.identity()
with st.sidebar:
    st.markdown("### SEISMIC\nVelocity Modelling Lab")
    st.caption("Explore the Earth. Examine the evidence.")
    page=st.radio("Workspace",PAGES,key="nav",label_visibility="collapsed")
    st.divider()
    st.caption("Signed in" if user else "Guest workspace")
    if user:
        st.write(user.email)
        if st.button("Log out",use_container_width=True):
            try:auth.client().auth.sign_out()
            finally:auth.clear();st.rerun()
    else:
        st.caption("Guest experiments stay in this tab. Login enables private cloud saves.")
    st.caption("Synthetic research demonstration · IIT Bombay study project")

def chart(fig,key=None):
    st.plotly_chart(fig,use_container_width=True,key=key,config={"displaylogo":False})
def select_input():
    samples=sorted((ROOT/"data").glob("*.npz"))
    with st.expander("Choose or upload seismic data",expanded="sample" not in st.session_state):
        tab,up=st.tabs(["Prepared examples","Upload"])
        with tab:
            if samples:
                chosen=st.selectbox("Example",samples,format_func=lambda p:p.stem)
                if st.button("Load example"):
                    d,v,m=load_sample(chosen.read_bytes(),chosen.name)
                    set_sample(d,v,m,chosen.stem)
                if user and st.button("Bookmark prepared example"):
                    try:
                        pr=store.profile(user);items=pr.get("bookmarks",[])
                        item={"kind":"sample","filename":chosen.name}
                        if item not in items:items.append(item)
                        store.update_profile(user,bookmarks=items[-100:]);st.success("Example bookmarked.")
                    except Exception:st.error("Could not save bookmark.")
            else:st.info("No samples found. Run scripts/make_samples.py.")
        with up:
            f=st.file_uploader("NPY / NPZ (12 MiB), or SEG-Y (200 MiB)",type=["npy","npz","sgy","segy"])
            st.caption("SEG-Y: pre-stack shot gathers. Stacked sections and arbitrary survey geometry cannot feed this checkpoint.")
            if f and Path(f.name).suffix.lower() in [".sgy",".segy"]:
                grouping=st.selectbox("Shot grouping header",list(SHOT_FIELDS))
                endian=st.selectbox("SEG-Y byte order",["big","little"])
                payload=f.getvalue()
                try:
                    info=inspect_segy(payload,SHOT_FIELDS[grouping],endian)
                    st.write({k:v for k,v in info.items() if k!="shots"})
                    st.dataframe(info["shots"],use_container_width=True)
                    ids=[r["shot_id"] for r in info["shots"]]
                    selected=st.multiselect("Select exactly five shot IDs (sorted by source X on loading)",ids,max_selections=5)
                    st.warning("Experimental field-data use: matching headers does not establish reliable velocity predictions. No automatic resizing, resampling or amplitude normalization is applied.")
                    confirmed=st.checkbox("These are unstacked shot gathers with a compatible 25 Hz source wavelet, surface geometry and calibrated raw amplitudes.")
                    if st.button("Validate and load SEG-Y",disabled=len(selected)!=5 or not confirmed):
                        d,v,m=load_segy(payload,selected,SHOT_FIELDS[grouping],endian)
                        m["source_hz"]=25.
                        m["source_frequency_verified"]="user confirmed"
                        m["experimental_acknowledged"]=True
                        set_sample(d,v,m,f.name)
                except Exception as e:
                    st.error(f"Cannot load SEG-Y: {e}")
            else:
                st.caption("NPZ keys: seismic (5,250,301), optional velocity (201,301), optional metadata JSON scalar.")
                if f and st.button("Validate and load upload"):
                    try:
                        d,v,m=load_sample(f.getvalue(),f.name);set_sample(d,v,m,f.name)
                    except Exception as e:st.error(f"Cannot load this file: {e}")
    if "sample" not in st.session_state:
        st.info("Load an example above to begin.");return None
    sample=st.session_state.sample
    st.caption(f"Active example: {sample['name']} · five shots · 250 samples · 301 receivers")
    return sample
def set_sample(d,v,m,name):
    for k in ["prediction","inference_seconds","robust","last_metrics"]:
        st.session_state.pop(k,None)
    st.session_state.sample=dict(data=d,truth=v,meta=m,name=name)
    st.success("Example loaded. Original input is preserved.")
def acquisition_ok(meta):
    checks={"dx_m":10.,"dt_s":.004,"source_hz":25.,"shots":5,"receivers":301,"time_samples":250}
    return all(k not in meta or np.isclose(float(meta[k]),v) for k,v in checks.items())
@st.cache_resource
def cached_model(signature):
    return load_model()
def run_network(data,meta):
    if not MODEL_PATH.exists():raise ValueError("Add models/fcnvmb_sim.pt from your notebook run to enable predictions.")
    if not acquisition_ok(meta):raise ValueError("Metadata differs from the notebook acquisition. This checkpoint is incompatible.")
    net,scale=cached_model((MODEL_PATH.stat().st_mtime_ns,MODEL_PATH.stat().st_size))
    return predict(net,scale,data,meta.get("preprocessing","raw"))
def inference_notice():
    if not MODEL_PATH.exists():st.warning("Trained checkpoint not supplied. Data tools and saved notebook results work; neural prediction is disabled.")
def metrics_row(pred,true):
    values=scores(pred,true)
    for col,(k,v) in zip(st.columns(4),values.items()):
        col.metric(k,f"{v:,.2f}" if k!="SSIM" else f"{v:.3f}",
            help={"MSE":"Mean squared velocity error, (m/s)²","RMSE":"Root mean squared velocity error, m/s",
                  "PSNR":"dB, fixed velocity data range 3000 m/s","SSIM":"Valid 7×7 windows, fixed data range 3000 m/s. Not percent accuracy."}[k])
    return values
def result_view(d,v,pred):
    cols=st.columns(3 if v is not None else 1)
    if v is not None:
        with cols[0]:chart(heat(v,"True velocity"))
        with cols[1]:chart(heat(pred,"Predicted velocity"))
        with cols[2]:chart(heat(pred-v,"Signed error","error"))
        m=metrics_row(pred,v)
        a,b=st.columns(2)
        with a:
            x=st.slider("Profile position (receiver/grid index)",0,300,150)
            chart(profile(v,pred,x))
        with b:
            step=20;xx=v.ravel()[::step];yy=pred.ravel()[::step]
            fig=go.Figure(go.Scattergl(x=xx,y=yy,mode="markers",marker=dict(size=3,opacity=.35,color="#D4685D")))
            fig.add_scatter(x=[1500,4500],y=[1500,4500],mode="lines",name="1:1")
            fig.update_layout(xaxis_title="True velocity (m/s)",yaxis_title="Predicted velocity (m/s)")
            chart(finish(fig))
        with st.expander("Error within a depth interval"):
            lo,hi=st.slider("Depth interval (m)",0,2000,(0,2000),10)
            if hi-lo>=60:st.json(scores(pred[lo//10:hi//10+1],v[lo//10:hi//10+1]))
            else:st.caption("Choose at least 60 m for a 7-sample SSIM window.")
    else:
        chart(heat(pred,"Predicted velocity"));m={}
        st.info("Ground truth unavailable: no error or accuracy metrics are calculated.")
    return m
def save_form(sample,pred=None,settings=None):
    if not user:
        st.info("Sign in on Home to save this experiment.");return
    with st.expander("Save this experiment"):
        with st.form("save_experiment"):
            name=st.text_input("Experiment name",value=sample["name"],max_chars=100)
            notes=st.text_area("Research notes",max_chars=4000)
            arrays=st.checkbox("Save private input/output arrays (up to 12 MiB)",True)
            ok=st.form_submit_button("Save to my workspace")
        if ok:
            if not name.strip():st.error("Enter an experiment name.");return
            try:
                meta={**sample["meta"],"source_name":sample["name"],"experiment":settings or {}}
                blob=pack(sample["data"],sample["truth"],pred,meta) if arrays else None
                m=scores(pred,sample["truth"]) if pred is not None and sample["truth"] is not None else {}
                store.save(user,name.strip(),notes,meta,m,blob);st.success("Experiment saved privately.")
            except Exception:st.error("Save failed. Check Supabase migrations, private bucket policies and connection.")

if page=="Home":
    st.markdown("""<div class="hero"><div class="eyebrow">INTERACTIVE GEOPHYSICS WORKSPACE</div>
    <h1>Seismic records.<br>Subsurface insight.</h1>
    <p>Explore how waves encode geology, test a learned inverse mapping, and inspect where velocity predictions succeed or fail.</p></div>""",unsafe_allow_html=True)
    a,b,c=st.columns(3)
    a.metric("Velocity grid","201 × 301");b.metric("Input channels","5 shots");c.metric("Spatial sampling","10 m")
    st.graphviz_chart('digraph { rankdir=LR; node [shape=box,style=rounded,color="#D4685D",fontname="Arial"]; "Velocity model" -> "Wave simulation" -> "Shot gathers" -> "Trained FCN" -> "Predicted velocity"; }')
    a,b=st.columns(2)
    with a:
        if st.button("Start guided demo",type="primary",use_container_width=True):goto("Model Explorer")
    with b:
        if st.button("Run a prediction",use_container_width=True):goto("Velocity Prediction")
    st.caption("Prepared arrays are physically simulated. Published/notebook results are labelled separately. No field-data validation is claimed.")
    if user:
        st.subheader("Recent experiments")
        try:
            rows=store.listing()
            st.dataframe([{k:r[k] for k in ["name","created_at"]} for r in rows[:5]],use_container_width=True)
        except Exception:st.info("Connect the database schema to display saved experiments.")
    else:auth.panel()

elif page=="Model Explorer":
    st.title("Model Explorer");st.write("Build a geological example, then examine its velocity structure.")
    with st.form("generate_model"):
        a,b=st.columns(2)
        seed=a.number_input("Random seed",0,100000,17);salt=b.checkbox("Include salt body",True)
        generate=st.form_submit_button("Generate velocity model",type="primary")
    if "velocity" not in st.session_state or generate:
        st.session_state.velocity=make_velocity_model(np.random.default_rng(seed),salt=salt)
        st.session_state.model_seed=int(seed)
        st.session_state.model_salt=bool(salt)
    v=st.session_state.velocity
    a,b=st.columns([2,1])
    with a:chart(heat(v,"Synthetic layered Earth"))
    with b:
        st.metric("Velocity range",f"{v.min():.0f}–{v.max():.0f} m/s")
        st.caption("5–12 curved layers, 2000–4000 m/s. Salt, when included, is 4500 m/s. The distribution is deliberately restricted.")
        x=st.slider("Vertical profile position (m)",0,3000,1500,10)
        chart(profile(v,None,x//10))
    if user and st.button("Bookmark this geological seed"):
        try:
            pr=store.profile(user);book=pr.get("bookmarks",[])
            item={"seed":st.session_state.model_seed,"salt":st.session_state.get("model_salt",True)}
            if item not in book:book.append(item)
            store.update_profile(user,bookmarks=book[-100:]);st.success("Bookmarked.")
        except Exception:st.error("Bookmark could not be saved.")
    buf=io.BytesIO();np.save(buf,v)
    st.download_button("Download velocity array",data=buf.getvalue(),file_name="velocity_model.npy")
    st.caption("This is a velocity map, not a seismic network input.")
    if st.button("Continue to seismic acquisition"):goto("Seismic Acquisition")

elif page=="Seismic Acquisition":
    st.title("Seismic Acquisition")
    s=select_input()
    with st.expander("Simulate the model from Model Explorer"):
        st.write("2-D constant-density acoustic propagation, second-order time and eighth-order space, side/bottom sponge. Approximately tens of seconds on a CPU.")
        st.caption("The NumPy port uses the notebook stencil. It has not been numerically cross-validated against PyTorch here.")
        if st.button("Simulate five shots",disabled="velocity" not in st.session_state):
            progress=st.progress(0.)
            try:
                v=st.session_state.velocity;d=simulate(v,progress=progress.progress)
                set_sample(d,v,{**DEFAULT_META,"seed":st.session_state.model_seed,"source":"NumPy FD simulation"},"Generated model");st.rerun()
            except Exception as e:st.error(f"Simulation failed: {e}")
    if s:
        shot=st.slider("Source shot",1,5,3)-1
        source=np.linspace(0,3,5)
        fig=go.Figure(go.Scatter(x=np.arange(301)*.01,y=np.zeros(301),mode="markers",name="Receivers",marker=dict(size=3,color="#64868B")))
        fig.add_scatter(x=source,y=np.zeros(5),mode="markers",name="Sources",marker=dict(size=13,symbol="star",color="#D4685D"))
        fig.add_scatter(x=[source[shot]],y=[0],mode="markers",name="Selected shot",marker=dict(size=19,symbol="circle-open",color="#233044"))
        fig.update_layout(xaxis_title="Horizontal distance (km)",yaxis_visible=False)
        chart(finish(fig,180))
        a,b=st.columns([2,1])
        with a:chart(heat(s["data"][shot],f"Shot {shot+1}","gather"))
        with b:
            receiver=st.slider("Receiver index",0,300,150)
            f=go.Figure(go.Scatter(x=np.arange(250)*.004,y=s["data"][shot,:,receiver],mode="lines"))
            f.update_layout(xaxis_title="Recording time (s)",yaxis_title="Amplitude");chart(finish(f))
        a,b=st.columns(2)
        with a:
            t,w=wavelet();f=go.Figure(go.Scatter(x=t,y=w))
            f.update_layout(title="25 Hz Ricker source",xaxis_title="Time (s)");chart(finish(f,260))
        with b:chart(spectrum(s["data"],"Active seismic"))
        if st.button("Continue to velocity prediction"):goto("Velocity Prediction")

elif page=="Velocity Prediction":
    st.title("Velocity Prediction");inference_notice()
    s=select_input()
    if s:
        if s["meta"].get("experimental"):
            st.warning("Experimental SEG-Y prediction: this synthetic-trained checkpoint has not been validated on field data.")
        st.json(s["meta"],expanded=False)
        meta=dict(s["meta"])
        if "preprocessing" not in meta:
            mode=st.radio("Input amplitude format",["raw","checkpoint_scaled"])
            meta["preprocessing"]=mode
        if not all(k in meta for k in ["dx_m","dt_s","source_hz","shots","receivers","time_samples"]):
            acknowledged=st.checkbox("I confirm this input matches 10 m receivers, 4 ms sampling, 25 Hz source and five notebook source positions.")
        else:acknowledged=True
        st.caption("Input scaling comes from the checkpoint. Model loading and preprocessing are excluded from the timed forward pass.")
        if st.button("Predict velocity",type="primary",disabled=not MODEL_PATH.exists() or not acknowledged):
            try:
                with st.spinner("Running the trained FCN…"):p,sec=run_network(s["data"],meta)
                st.session_state.prediction=p;st.session_state.inference_seconds=sec
                st.session_state.sample["meta"]=meta;st.success(f"Forward pass: {sec:.3f} s on this server.")
            except Exception as e:st.error(str(e))
        if "prediction" in st.session_state:
            chart(heat(st.session_state.prediction,"Current prediction"))
            if st.button("Inspect results and error"):goto("Results and Error Analysis")

elif page=="Results and Error Analysis":
    st.title("Results and Error Analysis")
    s=st.session_state.get("sample");p=st.session_state.get("prediction")
    if s is None or p is None:
        st.info("Run a prediction first. The saved notebook figure below is historical evidence, not a live result.")
        st.image(str(ROOT/"assets/notebook_41.png"),caption="Saved notebook output: truth, prediction and prediction with noise.")
    else:
        result_view(s["data"],s["truth"],p)
        st.download_button("Download input, prediction and metadata",pack(s["data"],s["truth"],p,s["meta"]),"velocity_result.npz")
        m=scores(p,s["truth"]) if s["truth"] is not None else {}
        st.download_button("Download metrics JSON",json.dumps(m,indent=2),"metrics.json")
        st.download_button("Download prediction CSV", "\n".join(",".join(f"{x:.3f}" for x in row) for row in p),"prediction.csv")
        save_form(s,p,{"inference_s":st.session_state.get("inference_seconds")})

elif page=="Robustness Lab":
    st.title("Robustness Lab");inference_notice();s=select_input()
    if s:
        with st.form("robustness"):
            a,b,c=st.columns(3)
            noise=a.slider("Gaussian noise (%)",0,25,5)/100
            amp=b.slider("Amplitude multiplier",.25,3.,1.,.05)
            mode=c.selectbox("High-pass method",["Hz","Notebook bins"])
            cutoff=st.slider("Cutoff (Hz)" if mode=="Hz" else "Fraction of lowest FFT bins",0.,50. if mode=="Hz" else .4,0.,1. if mode=="Hz" else .01)
            seed=st.number_input("Noise seed",0,100000,0)
            ok=st.form_submit_button("Apply from original input",type="primary")
        settings=dict(noise=noise,amplitude=amp,cutoff=cutoff,seed=int(seed),mode=mode)
        digest=hashlib.sha256(s["data"].tobytes()+json.dumps(settings,sort_keys=True).encode()).hexdigest()
        if ok:
            changed=transform(s["data"],**settings)
            st.session_state.robust=dict(data=changed,settings=settings,input_name=s["name"],digest=digest)
        r=st.session_state.get("robust")
        if r:
            st.caption("Displayed settings: "+json.dumps(r["settings"]))
            a,b=st.columns(2)
            with a:chart(heat(s["data"][2],"Original shot 3","gather"))
            with b:chart(heat(r["data"][2],"Altered shot 3","gather"))
            chart(spectrum(r["data"],"Altered",spectrum(s["data"],"Original")))
            if st.button("Compare predictions",disabled=not MODEL_PATH.exists()):
                try:
                    base,_=run_network(s["data"],s["meta"]);altered,_=run_network(r["data"],s["meta"])
                    r["baseline"]=base;r["pred"]=altered
                except Exception as e:st.error(str(e))
            if "pred" in r:
                a,b=st.columns(2)
                with a:chart(heat(r["baseline"],"Original-input prediction"))
                with b:chart(heat(r["pred"],"Altered-input prediction"))
                if s["truth"] is not None:
                    metrics=[dict(condition="Original",**scores(r["baseline"],s["truth"])),
                             dict(condition="Altered",**scores(r["pred"],s["truth"]))]
                    st.dataframe(metrics,use_container_width=True)
                altered_sample={**s,"data":r["data"],"name":s["name"]+" robustness"}
                save_form(altered_sample,r["pred"],r["settings"])
        st.caption("Noise sigma uses the full sample's 99.5th-percentile absolute amplitude after amplitude scaling. Each experiment starts from the original input.")

elif page=="FWI Comparison":
    st.title("Deep learning and FWI")
    historical=json.loads((ROOT/"assets/notebook_metrics.json").read_text())
    st.info("Historical notebook CPU demonstration. These are not measurements from your current session.")
    st.dataframe([{"Quantity":"MSE on first test model (m/s)²","FCN":156660,"FWI":25176},
                  {"Quantity":"Time per model (s)","FCN":.073,"FWI":461}],use_container_width=True)
    a,b=st.columns(2)
    with a:
        fig=go.Figure(go.Bar(x=["FCN inference","FWI"],y=[.073,461],marker_color=["#D4685D","#4E7378"]))
        fig.update_yaxes(type="log",title="Time (s), log scale");chart(finish(fig))
    with b:
        st.subheader("What the comparison supports")
        st.write("The network predicted faster. FWI recovered velocity more accurately on the one compared model.")
        st.write("FWI began with a Gaussian-smoothed true model. Network training (7.2 min) and dataset generation are excluded from inference timing.")
        st.write("One model and one CPU setup do not establish operational performance.")
    st.subheader("Saved robustness results")
    st.dataframe(historical["conditions"],use_container_width=True)
    st.caption("FWI arrays were not embedded in the supplied notebook. No FWI model maps are fabricated.")
    st.code("python scripts/train_notebook.py",language="powershell")
    st.caption("The original local training script includes the notebook's expensive FWI run. Never run it as a Streamlit page import.")

elif page=="Theory and Methodology":
    st.title("Theory and Methodology")
    topics=[
      ("Velocity models and inverse problems","A velocity model assigns P-wave speed to each subsurface point. Recording time depends on both velocity and geometry. Inferring the model from seismic data is an inverse problem.",r"\hat{v}=\mathrm{Net}(d;\Theta)"),
      ("Forward wave physics","2-D constant-density acoustic propagation. p is the wavefield, v is speed, x,z are position, t is time, and s is source forcing. The notebook injects a source q inside the v² factor, so s=v²q.",r"\frac{\partial^2p}{\partial t^2}=v^2\left(\frac{\partial^2p}{\partial x^2}+\frac{\partial^2p}{\partial z^2}\right)+s"),
      ("Finite differences","The notebook uses second-order time and eighth-order spatial derivatives. Side/bottom sponge damping absorbs outgoing waves. The undamped top is not an explicitly implemented physical free surface.",r"p^{n+1}=2p^n-p^{n-1}+(v\Delta t)^2(\nabla_h^2p^n+q^n)"),
      ("Supervised network training","Pairs of known models and simulated seismic data train the network. N counts output pixels. The optimizer uses gradients of the model-domain MSE to update network weights.",r"L=\frac{1}{N}\sum_i(\hat v_i-v_i)^2"),
      ("Conventional FWI and cycle skipping","FWI repeatedly simulates waveforms, compares observations and updates a velocity model. Oscillatory waveforms can create wrong local minima when the initial phases differ substantially.",r"J(m)=\tfrac12\|d_{obs}-F(m)\|_2^2"),
      ("Metrics and uncertainty","MSE has units (m/s)²; RMSE has m/s. PSNR uses range R=3000 m/s. SSIM compares local statistics and is not percent accuracy. These scores do not quantify uncertainty on unlabelled field data.",r"\mathrm{PSNR}=10\log_{10}\!\left(\frac{R^2}{\mathrm{MSE}}\right)"),
      ("Generalization","Limited geological diversity, short records, synthetic physics and acquisition changes constrain transfer to the field. A learned network is not guaranteed to satisfy the wave equation.",None)]
    for title,body,eq in topics:
        with st.expander(title):
            st.write(body)
            if eq:st.latex(eq)
    st.subheader("Exact demo architecture")
    st.graphviz_chart('digraph { rankdir=TB; node [shape=box,color="#D4685D"]; input [label="5×250×301 → pad to 5×256×304"]; e0 [label="16×256×304"];e1 [label="32×128×152"];e2 [label="64×64×76"];e3 [label="128×32×38"];b [label="256×16×19 bottleneck"];d3 [label="128×32×38"];d2 [label="64×64×76"];d1 [label="32×128×152"];d0 [label="16×256×304"];o [label="1×1 convolution → crop to 1×201×301"];input->e0->e1->e2->e3->b->d3->d2->d1->d0->o;e0->d0 [style=dashed,label="skip"];e1->d1 [style=dashed];e2->d2 [style=dashed];e3->d3 [style=dashed];}')
    st.caption("19 Conv2d layers plus 4 transposed convolutions. Demo width 16, approximately 1.94 million parameters. Cropping sets output shape; it is not a physical time–depth conversion.")
    st.image(str(ROOT/"assets/notebook_38.png"),caption="Actual saved training and test curves. No separate validation set was used.")
    st.write("Yang, F., & Ma, J. (2019). Deep-learning inversion: A next-generation seismic velocity model building method. Geophysics, 84(4), R583–R599.")
    st.link_button("Research paper","https://doi.org/10.1190/geo2018-0249.1")

elif page=="My Experiments":
    st.title("My Experiments")
    if not user:auth.panel()
    else:
        try:
            records=store.listing()
            if not records:st.info("No saved experiments yet.")
            else:
                st.dataframe([{k:r[k] for k in ["name","created_at","notes"]} for r in records],use_container_width=True)
                selected=st.selectbox("Open experiment",records,format_func=lambda r:r["name"]+" · "+r["created_at"][:16])
                st.write(selected["notes"]);st.json(selected["settings"],expanded=False)
                st.json(selected["metrics"],expanded=False)
                if st.button("Reopen saved arrays",disabled=not selected.get("object_path")):
                    blob=store.download(selected);d,v,m=load_sample(blob,"saved.npz")
                    with np.load(io.BytesIO(blob),allow_pickle=False) as z:
                        pred=numeric(z["prediction"],(201,301),"prediction") if "prediction" in z else None
                    set_sample(d,v,m,selected["name"])
                    if pred is not None:st.session_state.prediction=pred
                    goto("Results and Error Analysis" if pred is not None else "Seismic Acquisition")
                if not selected.get("object_path"):st.caption("Metadata-only save. Original arrays must be reloaded to reproduce this experiment.")
                with st.expander("Edit notes"):
                    notes=st.text_area("Saved notes",selected["notes"],max_chars=4000)
                    if st.button("Update notes"):
                        auth.client().table("experiments").update({"notes":notes}).eq("id",selected["id"]).execute();st.rerun()
                compare=st.multiselect("Compare two saved experiments",records,format_func=lambda r:r["name"]+" "+r["id"][:6],max_selections=2)
                if len(compare)==2:
                    st.dataframe([dict(name=r["name"],**r["metrics"]) for r in compare],use_container_width=True)
                    st.caption("Compare numerical values only when acquisition, truth and evaluation definitions match.")
                confirm=st.checkbox("Confirm deletion of selected experiment and its private arrays")
                if st.button("Delete selected experiment",disabled=not confirm):store.delete(selected);st.rerun()
        except Exception:st.error("Saved work could not be loaded. Check your connection and database policies.")

elif page=="Profile and Settings":
    st.title("Profile and Settings")
    if not user:auth.panel()
    else:
        try:
            pr=store.profile(user)
            with st.form("profile"):
                name=st.text_input("Display name",pr["display_name"],max_chars=80)
                show_tips=st.checkbox("Show study tips",pr.get("preferences",{}).get("show_tips",True))
                ok=st.form_submit_button("Save profile")
            if ok:store.update_profile(user,display_name=name,preferences={"show_tips":show_tips});st.success("Profile saved.")
            st.subheader("Bookmarked geological examples")
            for i,item in enumerate(pr.get("bookmarks",[])):
                if item.get("kind")=="sample":
                    if st.button("Open example "+item["filename"],key=f"sample_bookmark_{i}"):
                        f=ROOT/"data"/Path(item["filename"]).name
                        if f.exists():
                            d,v,m=load_sample(f.read_bytes(),f.name);set_sample(d,v,m,f.stem);goto("Seismic Acquisition")
                        else:st.info("The bookmarked example is no longer deployed.")
                    continue
                if st.button(f"Open seed {item['seed']} · salt {item['salt']}",key=f"bookmark_{i}"):
                    st.session_state.velocity=make_velocity_model(np.random.default_rng(item["seed"]),salt=item["salt"])
                    st.session_state.model_seed=item["seed"];st.session_state.model_salt=item["salt"];goto("Model Explorer")
            st.caption("Email/password credentials are managed by Supabase Auth. Passwords are never stored in application tables.")
            if show_tips:st.info("Compare a velocity profile and the full error map before interpreting a single similarity score.")
        except Exception:st.error("Profile could not be loaded. Check migrations and your connection.")
st.divider()
st.caption("Seismic Velocity Modelling Lab · synthetic demonstration · source notebook + Yang & Ma (2019)")
