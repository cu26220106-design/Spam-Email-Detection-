import os
import subprocess
import sys

base = os.path.dirname(os.path.abspath(__file__))
model = os.path.join(base, "models", "spam_classifier.joblib")
if not os.path.exists(model):
    subprocess.run([sys.executable, os.path.join(base, "src", "train.py")], check=True)

while True:
    text = input("\nEnter an email (or type 'exit' to quit): ").strip()
    if text.lower() == "exit":
        break
    if text:
        subprocess.run([sys.executable, os.path.join(base, "src", "predict.py"), text], check=True)
