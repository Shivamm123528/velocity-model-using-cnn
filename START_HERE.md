# Open the app

1. Extract this ZIP. Open PowerShell inside seismic_velocity_lab.
2. Install Python 3.12, then run:
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
3. Open the localhost URL shown by Streamlit.
4. Guest tools and included seismic datasets work without Supabase.
5. Enable real accounts: follow README.md > Supabase. Run the SQL migration, set email-code templates,
   and add your project's URL and publishable/anon key to .streamlit/secrets.toml.
6. Enable trained predictions: add the original fcnvmb_sim.pt checkpoint. README.md explains export.
7. Deploy: follow README.md > GitHub and Streamlit Community Cloud.

No real passwords, service-role keys or trained weights are included.
See TEST_REPORT.md for actual verification and environment limitations.
