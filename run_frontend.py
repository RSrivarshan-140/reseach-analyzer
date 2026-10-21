import subprocess
import sys
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "frontend", "app.py")
    
    print("=" * 60)
    print("Starting AI Research Assistant Streamlit UI...")
    print("Frontend URL: http://localhost:8501")
    print("=" * 60)
    
    cmd = [sys.executable, "-m", "streamlit", "run", app_path, "--server.port=8501", "--server.headless=false"]
    subprocess.run(cmd)
