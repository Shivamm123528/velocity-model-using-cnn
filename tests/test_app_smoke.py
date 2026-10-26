"""Optional full-dependency smoke test: python -m unittest tests.test_app_smoke"""
import unittest
from streamlit.testing.v1 import AppTest
class AppSmoke(unittest.TestCase):
    def test_guest_navigation(self):
        at=AppTest.from_file("app.py",default_timeout=30).run()
        self.assertEqual(len(at.exception),0)
        for page in ["Model Explorer","Seismic Acquisition","Velocity Prediction",
                     "Results and Error Analysis","Robustness Lab","FWI Comparison",
                     "Theory and Methodology","My Experiments","Profile and Settings"]:
            at.sidebar.radio[0].set_value(page).run()
            self.assertEqual(len(at.exception),0,page)
