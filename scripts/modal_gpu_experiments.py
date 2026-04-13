"""Run agent-stego experiments on Modal GPU.

Starts ollama on a GPU container, pulls models, runs experiments,
and downloads results locally.

Usage:
    modal run scripts/modal_gpu_experiments.py
"""

import json
import os
import subprocess
import time
from pathlib import Path

import modal  # type: ignore[import-untyped]

app = modal.App("agent-stego-experiments")

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


def _pull_models(models: list[str]):
    for model in models:
        name = model.removeprefix("ollama/")
        print(f"Pulling {name}...")
        subprocess.run(["ollama", "pull", name], check=True)
    model_volume.commit()


def _run_experiment(name: str, script: str, env: dict[str, str]) -> dict:
    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'=' * 60}")

    full_env = {**os.environ, **env}
    result = subprocess.run(
        ["python", f"experiments/{script}"],
        cwd="/app",
        env=full_env,
        capture_output=True,
        text=True,
        timeout=1200,
    )

    print(result.stdout)
    if result.stderr:
        print(f"STDERR: {result.stderr[:500]}")

    return {
        "name": name,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-2000:] if result.stdout else "",
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/root/.ollama/models": model_volume},
    timeout=1800,
)
def run_experiments():
    _start_ollama()
    _pull_models(["qwen2.5:3b", "qwen2.5:7b"])

    os.makedirs("/app/findings", exist_ok=True)
    os.chdir("/app")

    experiments = [
        (
            "Table Channel (3B, all modes)",
            "run_table.py",
            {
                "MODEL": "ollama/qwen2.5:3b",
                "SECRET": "HELLO",
                "TRIALS": "5",
                "MODE": "all",
            },
        ),
        (
            "Error Correction (3B)",
            "run_correction.py",
            {"MODEL": "ollama/qwen2.5:3b", "SECRET": "HI", "TRIALS": "5", "REP_N": "3"},
        ),
        (
            "7B Ceiling",
            "7b_ceiling.py",
            {
                "MODELS": "ollama/qwen2.5:7b",
                "LEVELS": "L0,L1,L1.5,L2",
                "TRIALS_SHORT": "5",
                "TRIALS_LONG": "3",
                "WITH_CORRECTION": "0",
            },
        ),
        (
            "Timing Local (7B)",
            "timing_local.py",
            {"MODEL": "ollama/qwen2.5:7b", "SECRET": "HI", "TRIALS": "3"},
        ),
    ]

    statuses = []
    for name, script, env in experiments:
        try:
            status = _run_experiment(name, script, env)
            statuses.append(status)
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT: {name}")
            statuses.append({"name": name, "returncode": -1, "stdout_tail": "TIMEOUT"})
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
            statuses.append({"name": name, "returncode": -1, "stdout_tail": str(e)})

    # Collect findings
    findings = {}
    findings_dir = Path("/app/findings")
    if findings_dir.exists():
        for subdir in findings_dir.iterdir():
            if subdir.is_dir():
                for f in subdir.glob("*.json"):
                    findings[f"{subdir.name}/{f.name}"] = json.loads(f.read_text())

    return {"statuses": statuses, "findings": findings}


@app.local_entrypoint()
def main():
    print("Launching experiments on Modal GPU (T4)...")
    result = run_experiments.remote()  # type: ignore[attr-defined]

    print(f"\n{'=' * 60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'=' * 60}")

    for key, data in result.get("findings", {}).items():
        out_path = Path(f"findings/{key}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved: {out_path}")

    print("\nStatus:")
    for s in result.get("statuses", []):
        code = s["returncode"]
        icon = "OK" if code == 0 else f"FAIL({code})"
        print(f"  [{icon}] {s['name']}")
