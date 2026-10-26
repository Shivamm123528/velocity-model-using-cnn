import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
def scores(pred,true):
    p,t=np.asarray(pred,dtype=np.float64),np.asarray(true,dtype=np.float64)
    if p.shape!=t.shape or p.ndim!=2 or min(p.shape)<7: raise ValueError("Matching 2D maps, minimum 7x7, required.")
    if not np.isfinite(p).all() or not np.isfinite(t).all(): raise ValueError("Non-finite maps.")
    mse=float(np.mean((p-t)**2))
    # Valid 7x7 windows match notebook avg_pool2d(...,7,1) with no padding.
    avg=lambda a: sliding_window_view(a,(7,7)).mean(axis=(-1,-2))
    a,b=avg(p),avg(t); aa,bb=avg(p*p)-a*a,avg(t*t)-b*b
    ab=avg(p*t)-a*b; c1,c2=(.01*3000)**2,(.03*3000)**2
    s=((2*a*b+c1)*(2*ab+c2))/((a*a+b*b+c1)*(aa+bb+c2))
    return dict(MSE=mse,RMSE=float(np.sqrt(mse)),
                PSNR=float(10*np.log10(3000**2/max(mse,1e-9))),SSIM=float(s.mean()))
