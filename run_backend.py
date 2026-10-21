import uvicorn
import os
import sys

# Ensure project root is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

if __name__ == "__main__":
    print("=" * 60)
    print("Starting AI Research Assistant FastAPI Backend...")
    print("Interactive Swagger Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=True)
