import os
import subprocess

if not os.path.exists("./chroma_db"):
    print("No chroma_db found, running ingest...")
    subprocess.run(["python", "ingest.py"], check=True)
else:
    print("chroma_db found, skipping ingest.")

os.system(f"uvicorn main:app --host 0.0.0.0 --port {os.getenv('PORT', 7860)}")