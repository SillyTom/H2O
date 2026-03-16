"""Run the THz-water interaction simulation."""
import subprocess
import sys

subprocess.run([sys.executable, "thz_water_simulation.py"], check=True)
