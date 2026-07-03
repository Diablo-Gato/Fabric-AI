import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
command = ["cmd.exe", "/c", "call", "run_pipeline.bat", "test scene prompt"]
print("command:", command)
result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
print("returncode:", result.returncode)
print("stdout:\n", result.stdout)
print("stderr:\n", result.stderr)
