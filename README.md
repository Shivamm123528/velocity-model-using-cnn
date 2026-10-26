# Seismic Velocity Modelling Lab

An interactive Streamlit research workspace based on the supplied annotated FCNVMB notebook.
Ten sections cover model generation, acquisition, neural inference, errors, robustness, FWI, theory,
private experiments and profiles. White/coral scientific styling, Plotly inspection, prepared data,
Supabase Auth and private result storage are implemented.

## What works without configuration
Guest navigation, two physically simulated examples, synthetic geological models, acquisition plots,
source wavelet, spectra, controlled input transformations, original notebook result figures and tables.
Guest work is temporary. Cloud account and experiment persistence require your Supabase project.

**No trained weights were supplied.** Actual network inference is deliberately disabled until you add
the original trusted checkpoint. The app never substitutes random weights or manufactured predictions.
Two bundled datasets were newly simulated for this app; they are not the notebook's test examples.

## Windows: start here
Install Python 3.12 from https://www.python.org/downloads/ and select Add Python to PATH.
Extract the ZIP, open PowerShell in the extracted seismic_velocity_lab directory, then run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the localhost URL printed by Streamlit. Using the virtual environment's Python directly avoids
PowerShell activation-policy problems. Alternatively right-click/run run.ps1 after allowing local scripts.

For a smaller initial installation without neural inference:
```powershell
.\.venv\Scripts\python.exe -m pip install streamlit numpy scipy plotly supabase
```
PyTorch imports are lazy. Add torch when you have a checkpoint. No training occurs during app startup.

## Supabase: real accounts and saved work
1. Create a project at https://supabase.com/dashboard . Keep the database password private.
2. In SQL Editor, run supabase/migrations/001_initial.sql ONCE in a fresh project.
   It creates profiles, experiments, an auth-user profile trigger, owner-scoped RLS and a private
   experiments Storage bucket (12 MiB per object). Existing users are backfilled.
3. In Authentication, enable email/password sign-in and email confirmation. Set a minimum password
   length of at least 10. Configure SMTP before sharing beyond a small private test.
4. In Authentication > Email Templates, change **Confirm signup** to include the one-time code:
   `<p>Your confirmation code is {{ .Token }}</p>`
   Change **Reset password** similarly:
   `<p>Your recovery code is {{ .Token }}</p>`
   The app uses explicit email/code verification. This avoids fragile fragment/redirect handling
   in Streamlit. Do not remove password reset protection or disable confirmation to work around errors.
5. Authentication > URL Configuration: set Site URL to http://localhost:8501 for local testing, then
   your exact HTTPS deployment URL for production. Add the local and deployed URLs to allowed redirects
   if you retain the default link templates. This implementation uses typed OTP codes, not callback tokens.
6. Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml. Fill the project URL and publishable
   key (or legacy anon key) from the project's API settings. NEVER use a service-role key.
7. Restart Streamlit. Register on Home, read your email, use Verify email, then sign in.

Passwords are managed only by Supabase Auth. Profiles contain display name/preferences/bookmarks.
Experiments contain owner, settings, metrics, notes and optional private object references.
Each browser session gets a separate Supabase client. Tokens remain in server-side session memory,
refresh on subsequent interactions and are cleared on logout. **A browser reload/reconnection may log
you out**; this project does not implement a persistent browser cookie. No tokens in URLs or localStorage.

Data saves are explicit. Supabase database/Storage are durable; Streamlit's local disk is not.
The UI shows the latest 100 saved experiments. No account-deletion feature is included; use Supabase's
account administration if required. Configure retention and quotas for your own deployment.

## Add your real trained model and original test data
The original notebook saves:
- fcnvmb_run/results/fcnvmb_sim.pt
- fcnvmb_run/data/meta.json
- fcnvmb_run/data/test_data.npy
- fcnvmb_run/data/test_vel.npy

Download those files from the notebook execution environment (for example Kaggle). With the complete
fcnvmb_run folder accessible locally, run:
```powershell
.\.venv\Scripts\python.exe scripts/export_run.py C:\path\to\fcnvmb_run
```
This copies the checkpoint into models/fcnvmb_sim.pt and exports three test examples.
The checkpoint must include state, in_ch, base, scale. The app uses torch.load(weights_only=True),
strict state validation and the exact FCNVMB class extracted from the notebook.

If you have only the checkpoint, copy it manually to models/fcnvmb_sim.pt. Do not upload checkpoints
from untrusted sources. They are not accepted through the app.

To rerun the original notebook pipeline locally (expensive, includes training and FWI):
```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
.\.venv\Scripts\python.exe scripts/train_notebook.py
.\.venv\Scripts\python.exe scripts/export_run.py .\fcnvmb_run
```
This script preserves original code cells. It may take considerably longer on your hardware and has
not been executed here. It creates its own figures and results under fcnvmb_run.
Optional external SEG data are not included.

## Input format and preprocessing
Raw .npy: finite real array, shape (5,250,301), raw simulator amplitude, no batch dimension.
.npz: required seismic, optional velocity (201,301), optional metadata scalar containing JSON.
A sample creation example:
```python
import json
import numpy as np
np.savez_compressed("example.npz", seismic=seismic.astype("float32"),
    velocity=velocity.astype("float32"),
    metadata=json.dumps({"dx_m":10,"dt_s":0.004,"source_hz":25,
        "shots":5,"time_samples":250,"receivers":301,"preprocessing":"raw"}))
```
The raw array is multiplied by the EXACT scale in the checkpoint. Original notebook .npy data already
have that scale applied: label their metadata preprocessing="checkpoint_scaled". Never scale twice.
Metadata mismatches prevent inference. NPY uploads require an acquisition acknowledgement.
Shape compatibility alone cannot establish field-data compatibility.

The output is denormalized with v=3000*y+1500 m/s.
MSE/RMSE, PSNR and SSIM appear only when ground truth exists.
SSIM uses valid 7x7 windows, matching the notebook's pooling geometry. PSNR uses a fixed 3000 m/s range,
with the notebook's numerical 1e-9 MSE floor. SSIM is not percent accuracy.

Noise uses 5% (by default) of the sample-wide 99.5th-percentile absolute amplitude; this follows executable
notebook code rather than its inconsistent per-shot commentary. Transforms start from the original input.
The Hz filter zeros FFT bins below a selected frequency. Notebook mode removes a fraction of low FFT bins.

## Simulator and scientific boundaries
The NumPy port retains the notebook's 2-D constant-density, isotropic acoustic stencil:
second-order time, eighth-order space, dt=1 ms, dh=10 m, 25 Hz Ricker wavelet,
side/bottom exponential sponge and surface-row recording. The top is not explicitly implemented as a
physical free surface. The port has not been cross-validated against PyTorch outputs here.
Sources: evenly spaced x=0,750,1500,2250,3000 m; receivers every 10 m.

The generator samples a restricted family of curved layers and salt bodies.
Prepared examples are synthetic evidence, not real Earth measurements.
Saved notebook CPU metrics are labelled historical and are not regenerated performance claims.
FWI starts from smoothed truth, covers one test model, and its timing excludes neural training/data creation.
FWI arrays were not embedded, so no FWI velocity maps are fabricated.
Changing acquisition or geological distribution can invalidate the learned mapping.
A shared public Streamlit server can be overwhelmed by concurrent simulation requests; use prepared
examples for a class demo and move expensive simulations to a job queue before broad public use.

## GitHub
In GitHub create an empty PRIVATE repository using the web interface. Then:
```powershell
git init
git add .
git commit -m "Build seismic velocity modelling lab"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
Replace only the final URL with the repository URL returned by GitHub.
Never commit .streamlit/secrets.toml. Models are ignored by default. If you choose to deploy inference,
include your trusted small checkpoint through a deliberate private-repository commit:
`git add -f models/fcnvmb_sim.pt`. Check GitHub size limits and model sharing rights first.
No external repository was created or published during delivery.

## Streamlit Community Cloud
1. Connect your GitHub account at https://share.streamlit.io .
2. Choose the repository, main branch and app.py entrypoint.
3. In Advanced settings choose Python 3.12.
4. Add the contents of your real secrets.toml to the app's Secrets setting. Do not upload the file to Git.
5. Deploy, inspect logs and confirm the Home page loads.
6. Configure the final Supabase Site URL and email templates as above.
7. Register two test users and run the owner-isolation checks before relying on private storage.

The app uses session-state navigation in one entrypoint. Hosted local files are temporary.
Inference requires a deployed checkpoint. New model architecture changes require a compatible checkpoint.
If memory is constrained, deploy guest data exploration first and run inference locally.

## Verification
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_science -v
.\.venv\Scripts\python.exe -m unittest tests.test_app_smoke -v
```
A two-user integration script is included at scripts/test_rls.py. Read its header, use a disposable
Supabase test project and two confirmed accounts, and supply its environment variables locally.
It verifies own-record access, blocks cross-user reads/updates and owner spoofing, and checks private
object isolation. Do not post test passwords in a chat or repository.
See TEST_REPORT.md for tests actually run during creation.

## Troubleshooting
- Missing module: install requirements with the SAME virtual-environment Python used to launch.
- Registration fails: verify URL/key, email auth, SMTP and rate limits in Supabase logs.
- Verification/reset code fails: update the corresponding email template to include {{ .Token }},
  request a fresh code and use the correct tab.
- Saved experiments fail: run migration once; verify bucket exists, RLS is enabled and user is signed in.
- Model absent: export the original trained checkpoint. The app intentionally does not invent predictions.
- Checkpoint mismatch: use original architecture and training metadata, not renamed unrelated weights.
- Page reload logs you out: expected session-memory behavior; sign in again.

## Sources
- Supplied velocity_modelling_annotated(1).ipynb: code and saved demo results.
- Yang & Ma (2019), Geophysics 84(4), R583–R599, https://doi.org/10.1190/geo2018-0249.1
- Supabase Python Auth: https://supabase.com/docs/reference/python/auth-signinwithpassword
- OTP verification: https://supabase.com/docs/reference/python/auth-verifyotp
- Email templates: https://supabase.com/docs/guides/auth/auth-email-templates
- Streamlit deployment: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
