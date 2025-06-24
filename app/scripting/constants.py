from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts"
SCRIPT_DIR1 = Path(__file__).parent.parent.parent
DEFAULT_LANGUAGE = "python"
TRIGGERS = {
    "generate": "generate code",
    "execute": "run code",
    "update": "update code",
    "delete": "delete code"
}
FILE_PATTERN = r"(\w+)_(\w+)_(\d{3})\.py"