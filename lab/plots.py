import numpy as np
import plotly.graph_objects as go
def finish(fig,height=360):
    fig.update_layout(height=height,margin=dict(l=25,r=20,t=35,b=30),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial",color="#233044"))
    return fig
def heat(a,title="",kind="velocity"):
    if kind=="gather":
        y=np.arange(a.shape[0])*.004; scale="RdBu_r";lim=max(float(np.quantile(abs(a),.995)),1e-10)
        opts=dict(zmin=-lim,zmax=lim);ylab="Recording time (s)";unit="Amplitude"
    elif kind=="error":
        y=np.arange(a.shape[0])*.01;scale="RdBu_r";lim=max(float(np.max(abs(a))),1.)
        opts=dict(zmin=-lim,zmax=lim);ylab="Depth (km)";unit="Error (m/s)"
    else:
        y=np.arange(a.shape[0])*.01;scale="Viridis";opts=dict(zmin=1500,zmax=4500)
        ylab="Depth (km)";unit="Velocity (m/s)"
    fig=go.Figure(go.Heatmap(z=a,x=np.arange(a.shape[1])*.01,y=y,colorscale=scale,
        colorbar=dict(title=unit),hovertemplate="x=%{x:.2f} km<br>y=%{y:.3f}<br>value=%{z:.2f}<extra></extra>",**opts))
    fig.update_layout(title=title,xaxis_title="Horizontal position (km)",yaxis_title=ylab)
    fig.update_yaxes(autorange="reversed");return finish(fig)
def profile(true,pred,x):
    fig=go.Figure()
    for name,a,col in [("True",true,"#1F8E7D"),("Predicted",pred,"#D4685D")]:
        if a is not None: fig.add_scatter(x=a[:,x],y=np.arange(a.shape[0])*.01,mode="lines",name=name,line_color=col)
    fig.update_yaxes(autorange="reversed",title="Depth (km)")
    fig.update_xaxes(title="Velocity (m/s)");return finish(fig)
def spectrum(d,label,fig=None):
    fig=go.Figure() if fig is None else fig
    sp=np.mean(np.abs(np.fft.rfft(d,axis=1)),axis=(0,2))
    fig.add_scatter(x=np.fft.rfftfreq(d.shape[1],.004),y=sp,mode="lines",name=label)
    fig.update_layout(xaxis_title="Frequency (Hz)",yaxis_title="Mean Fourier amplitude")
    return finish(fig)
