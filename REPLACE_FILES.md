# Install the v2 notebook compatibility update

This patch is based on the last Seismic Velocity Lab package supplied in this
conversation. It has not been compared with an unseen GitHub checkout. If you
made additional changes to app.py, compare them before replacing that file.

## First make a backup

Download your working GitHub repository as a ZIP and retain the old checkpoint.
Create a v2-test branch and upload these files there before updating your deployed
branch. Keep your existing Supabase credentials and database unchanged.

## Exact replacement list

Paths are relative to your GitHub repository root (the folder containing app.py).

| Path in this ZIP | Action in your repository |
|---|---|
| app.py | Replace the existing root app.py |
| lab/data.py | Replace, or add if absent |
| lab/inference.py | Replace, or add if absent |
| lab/segy.py | Replace, or add if absent |
| requirements.txt | Replace with the supplied dependency list |
| lab/v2_adapter.py | Add this new file inside lab/ |
| lab/v2_runtime.py | Add this new file inside lab/ |
| data/v2_quick_example.npz | Add this prepared example inside data/ |
| REPLACE_FILES.md | Optional installation notes |
| models/fcnvmb_sim.pt | Upload YOUR newly generated v2 checkpoint; not included here |

Keep the folder paths exactly as shown. Extract this ZIP before uploading: do not
upload the ZIP itself as app source, and do not add an extra enclosing project
folder. Keep lab/auth.py, lab/store.py, lab/authors.py, authors' HTML/images,
Supabase SQL, and .streamlit secrets in their existing locations.

The supplied requirements are those used by this app: Streamlit, NumPy, SciPy,
Plotly, Supabase, PyTorch and segyio. If you added other features/dependencies to
your own repository, merge those extra requirements into this list.

## Upload order

1. Add lab/v2_adapter.py and lab/v2_runtime.py.
2. Replace the three other lab/ files and requirements.txt.
3. Add data/v2_quick_example.npz.
4. Upload your new fcnvmb_sim.pt into models/.
5. Replace root app.py last. Prefer one commit containing the complete update.
6. Reboot/redeploy Streamlit after the commit has reached the branch your app uses.

Keep the deployed branch unchanged until the test branch is ready. When working
locally, run `python -m pip install -r requirements.txt`, then
`python -m streamlit run app.py` from the repository root.

## First prediction

1. Open Velocity Prediction.
2. Expand Load seismic data.
3. Click Load prepared v2 example.
4. Click Predict velocity.
5. Open Results and Error Analysis to inspect truth, prediction and error.

The example was physically simulated by the verified v2 quick notebook, using the
same seed/grid/acquisition. Raw amplitudes are restored from its scaled storage
before app inference. It is a synthetic test example, not field data. Your weights
may differ numerically across devices; the geometry and normalization still match.

## What the compatibility update fixes

The previous app required another checkpoint format. This update uses the network
definition and finite-difference modeler from velocity_modelling_v2.ipynb, including
the notebook's crop-or-resize output projection and side/bottom sponge. It does not
substitute the earlier research.py solver or its normalization.

Quick-test defaults:

- Seismic input: (2, 375, 151).
- Velocity output: (101, 151).
- Grid: 40 m; 4 km depth and 6 km width.
- Source: 3.5 Hz; 2 ms simulation step; keep every tenth sample -> 20 ms.
- Simulation: 3750 steps, nominal 7.5 s record.
- Base channels: 8.
- Velocity reconstruction: normalized output * 4500 + 1500 m/s.
- Raw input is multiplied by the checkpoint's saved amplitude scale exactly once.

The metadata is read from your checkpoint. Later v2 checkpoints saved with the
same explicit metadata can use different network dimensions without hard-coded
quick-test values in the app. Training counts not recorded in a non-quick
checkpoint are shown as unknown rather than invented.

The notebook's two tasks are distinct: the quick FCN has 2 training maps and 1
test map; its time-depth regression benchmark has 170 training columns and 30 test
columns. The 17,000/3,000 full TD split is not the FCN map count.

Older 2 km examples can remain in data/ for reference, but the new app lists only
the v2 prepared example. They are not resized to match the new checkpoint.
SEG-Y must match the checkpoint's shot positions, receivers and sample interval;
source wavelet/amplitude/depth assumptions still require verification.

## Verification

- Loaded the checkpoint produced by the updated v2 quick notebook with weights_only=True.
- Verified app predictions equal the notebook network's predictions.
- Verified waveform arrays equal the notebook Modeler results for the same model.
- Checked 2/3/4-unit generation, normalization, analytic time-depth result and metrics.
- Wrote and loaded an actual compatible SEG-Y file and verified the arrays.
- Opened all 12 guest pages using Streamlit AppTest.
- Loaded the prepared example, predicted through the live app controls, opened
  results/acquisition/time-depth pages, and ran the robustness test.
- Passed the 15 existing NumPy/data/SEG-Y regression tests.

Real Supabase login/cloud-save operations need your project credentials and were
not performed here. Existing auth/store code is reused without changing the
database schema. The quick-test weights demonstrate execution only; their
predictions should not be reported as scientifically useful accuracy.
