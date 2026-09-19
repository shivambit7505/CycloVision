import os
from pathlib import Path

def load_dotenv_custom(dotenv_path=None):
    if dotenv_path is None:
        dotenv_path = Path(__file__).resolve().parent / ".env"
    else:
        dotenv_path = Path(dotenv_path)

    if not dotenv_path.exists():
        return

    try:
        with open(dotenv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"Notice: Failed to load .env: {e}")

load_dotenv_custom()
