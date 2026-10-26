"""NumPy port of notebook FD stencil, source, sampling and sponge boundaries."""
import numpy as np
def gaussian_filter(a,sigma=1.):
    radius=int(4*sigma+.5);x=np.arange(-radius,radius+1)
    k=np.exp(-.5*(x/sigma)**2);k/=k.sum()
    out=np.asarray(a,dtype=float)
    for axis in range(out.ndim):
        pads=[(0,0)]*out.ndim;pads[axis]=(radius,radius)
        out=np.apply_along_axis(lambda t:np.convolve(t,k,mode="valid"),axis,np.pad(out,pads,mode="symmetric"))
    return out
def wavelet(nt=1000,dt=.001,fpeak=25.):
    t=np.arange(nt)*dt;a=(np.pi*fpeak*(t-1/fpeak))**2
    return t,((1-2*a)*np.exp(-a)).astype(np.float32)
def simulate(v,nt=1000,ds=4,shots=5,pad=40,progress=None):
    if v.ndim!=2 or not np.isfinite(v).all() or v.min()<=0 or v.max()>4600:
        raise ValueError("Finite positive velocities at most 4600 m/s required.")
    if not 1<=shots<=5 or not 1<=nt<=1000: raise ValueError("Simulation resource limit.")
    nz,nx=v.shape;dh,dt=10.,.001
    vp=np.pad(v,((0,pad),(pad,pad)),mode="edge").astype(np.float32)
    Z,X=vp.shape;fx=np.ones(X,np.float32);fz=np.ones(Z,np.float32)
    for i in range(pad):
        f=np.exp(-(.01*(pad-i))**2);fx[i]=f;fx[X-1-i]=f;fz[Z-1-i]=f
    damp=fz[:,None]*fx[None,:];vd=(vp*dt)**2
    src=np.round(np.linspace(0,nx-1,shots)).astype(int)
    mask=np.zeros((shots,Z,X),np.float32)
    for s,x in enumerate(src):mask[s,1,pad+x]=1/dh**2
    _,wav=wavelet(nt);p0=np.zeros_like(mask);p1=np.zeros_like(mask);rec=[]
    c=np.array([-205/72,8/5,-1/5,8/315,-1/560],np.float32)/dh**2
    for it in range(nt):
        q=np.pad(p1,((0,0),(4,4),(4,4)))
        lap=2*c[0]*p1
        for j in range(1,5):
            lap+=c[j]*(q[:,4:4+Z,4-j:4-j+X]+q[:,4:4+Z,4+j:4+j+X]+
                       q[:,4-j:4-j+Z,4:4+X]+q[:,4+j:4+j+Z,4:4+X])
        p2=2*p1-p0+vd*(lap+mask*wav[it])
        p0,p1=p1*damp,p2*damp
        if it%ds==0:rec.append(p1[:,1,pad:pad+nx].copy())
        if progress and it%50==0:progress((it+1)/nt)
    if progress:progress(1.)
    return np.stack(rec,axis=1)
