"""One command: capture normal + scan flows, build the local dataset, retrain.
Run from the src folder, in an ADMINISTRATOR terminal, venv active:
    python auto_local.py
"""
import os
import subprocess
import sys
import time

TARGET = "192.168.0.112"      # your phone
PY = sys.executable


def run(args):
    print(">", " ".join(args), flush=True)
    subprocess.run(args, check=True)


print("PART 1/2: capturing NORMAL traffic for 5 minutes.")
print("Browse, watch YouTube, scroll social media. Do NOT scan anything.")
input("Press Enter to start... ")
run([PY, "flow_tracker.py", "--live", "--seconds", "300", "--out", "local_normal_1.csv"])

print("\nPART 2/2: scanning your phone automatically. Don't touch anything (about 3 minutes).")
tracker = subprocess.Popen([PY, "flow_tracker.py", "--live", "--seconds", "180",
                            "--out", "local_scan_1.csv"])
time.sleep(10)                                   # let the capture start first
for flag in ["-sS", "-sT", "-sF", "-sX"]:
    subprocess.run(["nmap", flag, "-p", "1-1000", TARGET])
tracker.wait()

scan_files = ["local_scan_1.csv"] + (["scan_flows.csv"] if os.path.exists("scan_flows.csv") else [])
run([PY, "build_local_dataset.py", "--normal", "local_normal_1.csv",
     "--scan", *scan_files, "--target-ip", TARGET])
run([PY, "train_live.py"])
print("\nDONE. Model retrained with your own flows.")