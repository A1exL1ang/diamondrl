from pathlib import Path

import modal

from neverwake.training.run import find_run_dir, run

app = modal.App("neverwake")

# Locked dependencies from uv.lock (no dev group), then our package and the configs
image = (
    modal.Image.debian_slim(python_version="3.13")
    .uv_sync(extra_options="--no-dev")
    .add_local_python_source("neverwake")
    .add_local_dir("configs", remote_path="/root/configs")
)

# Runs persist here, download one with `modal volume get neverwake-runs <name>`
runs = modal.Volume.from_name("neverwake-runs", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    memory=32 * 1024,  # The replay buffer alone reaches ~5 GB, twice that while loading
    timeout=24 * 60 * 60,  # Modal's maximum
    # Note that a timeout counts as a failure, and since an unfinished run with the
    # same config is resumed, each retry continues from the latest checkpoint
    retries=modal.Retries(max_retries=10, backoff_coefficient=1.0, initial_delay=10.0),
    volumes={"/runs": runs},
    secrets=[modal.Secret.from_name("wandb")],
)
def train_remote(config: str) -> None:
    run(find_run_dir(Path("/root") / config, Path("/runs")))


@app.local_entrypoint()
def main(config: str = "configs/smoke.yaml") -> None:
    # Note that a spawned call isn't tied to this process, so with --detach training
    # keeps going even if this terminal is closed or killed
    call = train_remote.spawn(config)
    print(f"training started as {call.object_id}, safe to close with --detach")
    call.get()
