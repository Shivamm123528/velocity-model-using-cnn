# Verification report

## Executed and passed
- Python compileall for app.py, lab/, scripts/ and tests/: no syntax errors.
- Ten unittest cases in tests.test_science: passed.
- Generated two full 5×250×301 seismic examples using real numerical wave simulation.
- Confirmed prepared arrays contain finite nonzero waveforms and 201×301 velocity labels.
- Tested identity/offset metrics, known PSNR, reproducible geology/noise, input immutability,
  shape/nonfinite/pickle rejection, expanded-archive size rejection and short solver output.

## Implemented but not verified against a live service
- Supabase registration, login, email OTP verification, recovery, token refresh, profiles,
  private experiment rows, storage downloads and owner isolation.
- SQL policies have been inspected in source but have not been executed in your account.
- scripts/test_rls.py provides a two-account integration check for a disposable test project.

## Environment limits
The available Python environment does not contain Streamlit, Plotly, Supabase or PyTorch.
Installation was attempted into a project-local environment but DNS resolution for pypi.org was blocked.
Consequently a full Streamlit startup/browser test, AppTest run and dependency-resolution test
could not be completed here. requirements.txt declares supported version ranges, not a resolved lock.
Do not interpret syntax and numerical tests as proof of complete UI/service behavior.

## Missing scientific assets
The uploaded notebook contains code and saved figures, but not the trained fcnvmb_sim.pt checkpoint
or original prediction/FWI arrays. Network inference is disabled until the checkpoint is added.
scripts/export_run.py imports that checkpoint and original test arrays from your notebook run.
Full training and FWI were not run. No synthetic prediction result has been manufactured.
The NumPy forward solver is a direct stencil/source/boundary port, but its numeric equivalence
to the original PyTorch solver has not been cross-tested in this environment.

## Deployment
No external GitHub repository, Supabase project, outgoing email or Streamlit deployment was created.
Secrets remain examples only. The README contains exact local and hosted setup instructions.
