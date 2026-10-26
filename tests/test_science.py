import io,json,unittest,zipfile
import numpy as np
from lab.data import load_sample,pack,transform,numeric
from lab.metrics import scores
from lab.generator import make_velocity_model
from lab.physics import simulate,wavelet
class ScienceTests(unittest.TestCase):
    def test_metric_identity_and_offset(self):
        t=np.arange(121,dtype=float).reshape(11,11)+2000
        self.assertEqual(scores(t,t)["MSE"],0)
        self.assertAlmostEqual(scores(t,t)["SSIM"],1.)
        self.assertAlmostEqual(scores(t+100,t)["RMSE"],100.)
    def test_metric_known_psnr(self):
        t=np.full((9,9),2000.)
        self.assertAlmostEqual(scores(t+300,t)["PSNR"],20.)
    def test_roundtrip(self):
        d=np.zeros((5,250,301),np.float32);v=np.ones((201,301),np.float32)*3000
        a,b,m=load_sample(pack(d,v,meta={"preprocessing":"raw"}),"input.npz")
        np.testing.assert_array_equal(a,d);np.testing.assert_array_equal(b,v)
        self.assertEqual(m["preprocessing"],"raw")
    def test_bad_data(self):
        for arr in [np.zeros((2,3)),np.full((5,250,301),np.nan)]:
            buf=io.BytesIO();np.save(buf,arr)
            with self.assertRaises(ValueError):load_sample(buf.getvalue(),"input.npy")
    def test_object_arrays_rejected(self):
        b=io.BytesIO();np.save(b,np.array([{"evil":1}],dtype=object))
        with self.assertRaises(ValueError):load_sample(b.getvalue(),"input.npy")
    def test_expanded_archive_limit(self):
        b=io.BytesIO()
        with zipfile.ZipFile(b,"w",zipfile.ZIP_DEFLATED) as z:z.writestr("x",bytes(13*1024**2))
        with self.assertRaises(ValueError):load_sample(b.getvalue(),"huge.npz")
    def test_transforms_preserve_original(self):
        d=np.ones((5,250,301),np.float32)
        a=transform(d,noise=.05,seed=4);b=transform(d,noise=.05,seed=4)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(d,1)
        np.testing.assert_allclose(transform(d,amplitude=2),2)
        np.testing.assert_allclose(transform(d,cutoff=10),0,atol=1e-5)
    def test_model_reproducible(self):
        a=make_velocity_model(np.random.default_rng(17))
        b=make_velocity_model(np.random.default_rng(17))
        np.testing.assert_array_equal(a,b)
        self.assertEqual(a.shape,(201,301));self.assertEqual(a.max(),4500)
    def test_wave_solver(self):
        v=np.full((21,31),2500.,np.float32)
        d=simulate(v,nt=40,ds=4,shots=1,pad=8)
        self.assertEqual(d.shape,(1,10,31));self.assertTrue(np.isfinite(d).all())
        self.assertGreater(float(abs(d).max()),0)
    def test_prepared_samples(self):
        from pathlib import Path
        for p in Path("data").glob("sample*.npz"):
            d,v,m=load_sample(p.read_bytes(),p.name)
            self.assertTrue(np.isfinite(d).all());self.assertGreater(float(abs(d).max()),0)
            self.assertEqual(v.shape,(201,301));self.assertEqual(m["preprocessing"],"raw")
if __name__=="__main__":unittest.main()
