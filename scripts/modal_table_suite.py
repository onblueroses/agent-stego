"""Run all table channel experiments on Modal GPU in parallel.

3 parallel GPU jobs:
  1. Deep dive (mode comparison + length stress + cross-model + detectability)
  2. Multi-turn encoding (one char per turn)
  3. Table + error correction (RepetitionCode)

Usage:
    modal run scripts/modal_table_suite.py
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal  # type: ignore[import-untyped]

app = modal.App("agent-stego-table-suite")

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

EXPERIMENTS = [
    {
        "name": "deep_dive",
        "script": "experiments/table_deep_dive.py",
        "env": {
            "MODEL": "ollama/qwen2.5:3b",
            "TRIALS": "3",
            "BASELINE": "3",
            "SKIP_CROSS_MODEL": "1",
        },
        "timeout": 1200,
    },
    {
        "name": "multiturn",
        "script": "experiments/table_multiturn.py",
        "env": {"MODEL": "ollama/qwen2.5:3b", "TRIALS": "3"},
        "timeout": 900,
    },
    {
        "name": "correction",
        "script": "experiments/table_correction.py",
        "env": {"MODEL": "ollama/qwen2.5:3b", "TRIALS": "3"},
        "timeout": 900,
    },
]


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
    timeout=1500,
)
def run_experiment(name: str, script: str, env: dict, timeout: int):
    """Run a single experiment on a GPU container."""
    _start_ollama()

    print(f"[{name}] Pulling qwen2.5:3b...")
    subprocess.run(["ollama", "pull", "qwen2.5:3b"], check=True)
    model_volume.commit()

    os.makedirs("/app/findings", exist_ok=True)
    os.chdir("/app")

    full_env = {**os.environ, **env}

    print(f"[{name}] Starting experiment...")
    result = subprocess.run(
        ["python", script],
        cwd="/app",
        env=full_env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:1000]}")

    # Collect all findings
    findings = {}
    findings_dir = Path("/app/findings")
    if findings_dir.exists():
        for subdir in findings_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.json"):
                    findings[f"{subdir.name}/{f.name}"] = json.loads(f.read_text())

    return {
        "name": name,
        "returncode": result.returncode,
        "stdout": result.stdout[-3000:] if result.stdout else "",
        "findings": findings,
    }


@app.local_entrypoint()
def main():
    print(f"Launching {len(EXPERIMENTS)} table experiments in parallel on Modal...")

    # Fan out all experiments
    handles = []
    for exp in EXPERIMENTS:
        handle = run_experiment.spawn(  # type: ignore[attr-defined]
            exp["name"], exp["script"], exp["env"], exp["timeout"]
        )
        handles.append((exp["name"], handle))
        print(f"  Spawned: {exp['name']}")

    # Collect results
    for name, handle in handles:
        try:
            result = handle.get()
            code = result["returncode"]
            status = "OK" if code == 0 else f"FAIL({code})"
            print(f"\n[{status}] {name}")

            # Print summary (last 50 lines)
            if result["stdout"]:
                lines = result["stdout"].strip().split("\n")
                for line in lines[-50:]:
                    print(f"  {line}")

            # Save findings locally
            for fpath, data in result.get("findings", {}).items():
                out_path = Path(f"findings/{fpath}")
                out_path.parent.mkdir(parents=True, exist_ok=True)
                with open(out_path, "w") as f:
                    json.dump(data, f, indent=2)
                print(f"  Saved: {out_path}")

        except Exception as e:
            print(f"\n[ERROR] {name}: {e}")

    print("\nAll experiments complete.")
