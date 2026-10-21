import subprocess
import sys
import time
import os

def main():
    print("=" * 65)
    print("   AI Research Paper & PDF Assistant - Full Application Launcher")
    print("=" * 65)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    python_exe = sys.executable

    backend_script = os.path.join(current_dir, "run_backend.py")
    frontend_script = os.path.join(current_dir, "run_frontend.py")

    print("[1/2] Launching FastAPI Backend on http://127.0.0.1:8000 ...")
    backend_proc = subprocess.Popen([python_exe, backend_script])

    # Wait 2 seconds for backend to initialize
    time.sleep(2)

    print("[2/2] Launching Streamlit Frontend on http://localhost:8501 ...")
    frontend_proc = subprocess.Popen([python_exe, frontend_script])

    print("\nApplication running! Press Ctrl+C in this terminal to shut down both services.")

    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down services...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("Done.")

if __name__ == "__main__":
    main()
