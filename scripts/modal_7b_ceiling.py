"""Run ONLY the 7B ceiling experiment on Modal GPU with extended timeout.

Usage:
    modal run scripts/modal_7b_ceiling.py
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal  # type: ignore[import-untyped]

app = modal.App("agent-stego-7b-ceiling")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("curl", "ca-certificates", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({"OLLAMA_HOST": "0.0.0.0:11434", "OLLAMA_MODELS": "/root/.ollama/models"})
    .pip_install("openai", "numpy", "python-dotenv")
    .add_local_dir("src", remote_path="/app/src")
    .add_local_dir("experiments", remote_path="/app/experiments")
    .add_local_dir("prompts", remote_path="/app/prompts")
    .add_local_file("pyproject.toml", remote_path="/app/pyproject.toml")
)

model_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)


def _start_ollama():
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        try:
            import urllib.request

            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
            print("ollama ready")
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("ollama failed to start")


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/.ollama/models": model_volume},
    timeout=3600,
)
def run_7b_ceiling():
    _start_ollama()

    name = "qwen2.5:7b"
    print(f"Pulling {name}...")
    subprocess.run(["ollama", "pull", name], check=True)
    model_volume.commit()

    os.makedirs("/app/findings", exist_ok=True)
    os.chdir("/app")

    env = {
        **os.environ,
        "MODELS": "ollama/qwen2.5:7b",
        "LEVELS": "L0,L0.5,L1,L1.5,L2",
        "TRIALS_SHORT": "5",
        "TRIALS_LONG": "3",
        "WITH_CORRECTION": "0",
    }

    result = subprocess.run(
        ["python", "experiments/7b_ceiling.py"],
        cwd="/app",
        env=env,
        capture_output=True,
        text=True,
        timeout=3000,
    )

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:1000]}")

    findings = {}
    findings_dir = Path("/app/findings/7b_ceiling")
    if findings_dir.exists():
        for f in findings_dir.glob("*.json"):
            findings[f.name] = json.loads(f.read_text())

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "findings": findings,
    }


@app.local_entrypoint()
def main():
    print("Launching 7B ceiling on Modal GPU (T4, 1h timeout)...")
    result = run_7b_ceiling.remote()  # type: ignore[attr-defined]

    print(f"\nReturn code: {result['returncode']}")

    for name, data in result.get("findings", {}).items():
        out_dir = Path("findings/7b_ceiling")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / name
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path}")
