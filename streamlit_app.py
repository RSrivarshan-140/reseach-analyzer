import sys
import runpy
from pathlib import Path

# Add project root to python path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute frontend app
frontend_app_path = root_dir / "frontend" / "app.py"
runpy.run_path(str(frontend_app_path), run_name="__main__")
