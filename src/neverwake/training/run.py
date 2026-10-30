import argparse
import re
import shutil
import time
import uuid
from dataclasses import replace
from pathlib import Path

import dacite

from neverwake.config.core import Config
from neverwake.config.env import (
    AtariEnvConfig,
    CraftaxClassicAchRewEnvConfig,
    DoomDeathmatchKillRewEnvConfig,
    DoomHpSupBinRewEnvConfig,
    MarioForwardRewL1EnvConfig,
    ProcgenEnvConfig,
)
from neverwake.config.load import load_config
from neverwake.env.atari import AtariEnv
from neverwake.env.base import AbstractEnv
from neverwake.env.craftax_classic_ach_rew import CraftaxClassicAchRewEnv
from neverwake.env.doom_deathmatch_kill_rew import DoomDeathmatchKillRewEnv
from neverwake.env.doom_hp_sup_bin_rew import DoomHpSupBinRewEnv
from neverwake.env.mario_forward_rew_l1 import MarioForwardRewL1Env
from neverwake.env.procgen_env import ProcgenEnv
from neverwake.training.checkpoint import Checkpointer, load_checkpoint, purge, restore
from neverwake.training.telemetry import Telemetry
from neverwake.training.train import init_state, train


def find_run_dir(config_path: Path, runs_dir: Path) -> Path:
    """
    The newest unfinished run with the same config, otherwise a new run directory
    holding a copy of the config
    """
    cfg = load_config(config_path)
    for run_config in sorted(runs_dir.glob("*/config.yaml"), reverse=True):
        finished = (run_config.parent / "models" / "final.pt").exists()
        if not finished and same_config(run_config, cfg):
            return run_config.parent

    run_dir = runs_dir / run_name(cfg)
    run_dir.mkdir(parents=True)
    shutil.copy(config_path, run_dir / "config.yaml")
    return run_dir


def same_config(path: Path, cfg: Config) -> bool:
    # Note that runs made before a config field was added or removed never match
    try:
        return load_config(path) == cfg
    except dacite.DaciteError:
        return False


def run_name(cfg: Config) -> str:
    # e.g. SpaceInvaders --> space_invaders_20261003-101500
    if isinstance(cfg.env, AtariEnvConfig):
        game = re.sub(r"(?<!^)(?=[A-Z])", "_", cfg.env.game).lower()
    elif isinstance(cfg.env, MarioForwardRewL1EnvConfig):
        game = "mario_forward_rew_l1"
    elif isinstance(cfg.env, DoomHpSupBinRewEnvConfig):
        game = "doom_hp_sup_bin_rew"
    elif isinstance(cfg.env, DoomDeathmatchKillRewEnvConfig):
        game = "doom_deathmatch_kill_rew"
    elif isinstance(cfg.env, CraftaxClassicAchRewEnvConfig):
        game = "craftax_classic_ach_rew"
    elif isinstance(cfg.env, ProcgenEnvConfig):
        game = f"procgen_{cfg.env.game}"
    else:
        raise ValueError(f"unsupported env config {type(cfg.env).__name__}")
    return f"{game}_{time.strftime('%Y%m%d-%H%M%S')}"


def run(run_dir: Path) -> None:
    """
    Trains the run in run_dir, resuming from its latest checkpoint if it has one
    """
    if (run_dir / "models" / "final.pt").exists():
        print(f"{run_dir.name} already finished")
        return

    cfg = load_config(run_dir / "config.yaml")

    # Training ends episodes on life loss, eval plays full games
    train_env: AbstractEnv
    eval_env: AbstractEnv
    if isinstance(cfg.env, AtariEnvConfig):
        train_env = AtariEnv(cfg.env)
        eval_env = AtariEnv(replace(cfg.env, episodic_life=False))
    elif isinstance(cfg.env, MarioForwardRewL1EnvConfig):
        train_env = MarioForwardRewL1Env(cfg.env)
        eval_env = MarioForwardRewL1Env(cfg.env)
    elif isinstance(cfg.env, DoomHpSupBinRewEnvConfig):
        train_env = DoomHpSupBinRewEnv(cfg.env)
        eval_env = DoomHpSupBinRewEnv(cfg.env)
    elif isinstance(cfg.env, DoomDeathmatchKillRewEnvConfig):
        train_env = DoomDeathmatchKillRewEnv(cfg.env)
        eval_env = DoomDeathmatchKillRewEnv(cfg.env)
    elif isinstance(cfg.env, CraftaxClassicAchRewEnvConfig):
        train_env = CraftaxClassicAchRewEnv(cfg.env)
        eval_env = CraftaxClassicAchRewEnv(cfg.env)
    elif isinstance(cfg.env, ProcgenEnvConfig):
        train_env = ProcgenEnv(cfg.env)
        eval_env = ProcgenEnv(cfg.env)
    else:
        raise ValueError(f"unsupported env config {type(cfg.env).__name__}")

    # Note that anything written after the checkpoint is deleted
    loaded = load_checkpoint(run_dir)
    checkpoint, episodes = loaded if loaded is not None else (None, [])
    purge(run_dir, checkpoint)
    epoch = checkpoint["epoch"] if checkpoint is not None else 0
    run_id = checkpoint["run_id"] if checkpoint is not None else uuid.uuid4().hex[:8]

    state = init_state(cfg, train_env, eval_env, epoch)

    if checkpoint is not None:
        restore(
            checkpoint=checkpoint,
            episodes=episodes,
            agent=state.agent,
            optimizers=state.optimizers,
            replay_buffer=state.replay_buffer,
            collector=state.collector,
        )
        state.epochs_after_collection = checkpoint["epochs_after_collection"]
        print(f"resuming {run_dir.name} from epoch {epoch}")
    else:
        print(f"starting {run_dir.name}")

    # Note that telemetry is made right before training since it starts the clock
    tel = Telemetry(
        run_dir=run_dir,
        cfg=cfg,
        optimizers=state.optimizers,
        collector=state.collector,
        evaluator=state.evaluator,
        epoch=epoch,
        run_id=run_id,
    )
    checkpointer = Checkpointer(
        run_dir=run_dir,
        save_every=cfg.telemetry.save_every,
        agent=state.agent,
        optimizers=state.optimizers,
        replay_buffer=state.replay_buffer,
        collector=state.collector,
        tel=tel,
    )
    try:
        train(cfg, state, tel, checkpointer)
    finally:
        tel.close()


def main() -> None:
    # Note that an unfinished run with the same config is resumed automatically
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run(find_run_dir(args.config, args.runs_dir))


if __name__ == "__main__":
    main()
