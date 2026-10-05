import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch

from neverwake.data.episode import Episode
from neverwake.data.replay_buffer import ReplayBuffer
from neverwake.training.build import Agent
from neverwake.training.collector import Collector
from neverwake.training.optim import Optimizers
from neverwake.training.telemetry import Telemetry


class Checkpointer:
    """
    Saves everything needed to resume after every epoch, plus permanent weights
    every save_every epochs
    """

    def __init__(
        self,
        run_dir: Path,
        save_every: int,
        agent: Agent,
        optimizers: Optimizers,
        replay_buffer: ReplayBuffer,
        collector: Collector,
        tel: Telemetry,
    ):
        self.run_dir = run_dir
        self.save_every = save_every
        self.agent = agent
        self.optimizers = optimizers
        self.replay_buffer = replay_buffer
        self.collector = collector
        self.tel = tel
        self.model_names = [f.name for f in fields(optimizers)]

        # Tracking
        self.episodes_env_steps: int | None = None  # When episodes.pt was written

    def save(self, epoch: int, epochs_after_collection: int) -> None:
        weights = {
            name: getattr(self.agent, name).state_dict() for name in self.model_names
        }
        if epoch % self.save_every == 0:
            atomic_save(weights, self.run_dir / "models" / f"epoch_{epoch:04d}.pt")

        # Note that the buffer is written before the checkpoint so it is never behind
        # it, and only when it changed since it is rewritten whole
        if self.collector.env_steps != self.episodes_env_steps:
            episodes = [
                {
                    "observations": episode.observations,
                    "actions": episode.actions,
                    "rewards": episode.rewards,
                    "terminations": episode.terminations,
                    "epochs": episode.epochs,
                }
                for episode in self.replay_buffer.episodes.values()
            ]
            atomic_save(episodes, self.run_dir / "episodes.pt")
            self.episodes_env_steps = self.collector.env_steps

        checkpoint = {
            "epoch": epoch,
            "epochs_after_collection": epochs_after_collection,
            "models": weights,
            "optimizers": {
                name: getattr(self.optimizers, name).state_dict()
                for name in self.model_names
            },
            "collector": {
                "env_steps": self.collector.env_steps,
                "game_count": self.collector.game_count,
                "reward_pos": self.collector.reward_pos,
                "reward_neg": self.collector.reward_neg,
            },
            # Telemetry
            "counters": self.tel.counters(epoch),
            "run_id": self.tel.log.run.id,
        }
        atomic_save(checkpoint, self.run_dir / f"checkpoint_{epoch:04d}.pt")

        # Note that the previous checkpoint is kept in case the newest one is ever
        # on disk without its episodes.pt (e.g. a volume synced mid save)
        for path in sorted(self.run_dir.glob("checkpoint_*.pt"))[:-2]:
            path.unlink()

        # Note that this lands in the next epoch's row since this one is written
        self.tel.lap("save")

    def save_final(self) -> None:
        weights = {
            name: getattr(self.agent, name).state_dict() for name in self.model_names
        }
        atomic_save(weights, self.run_dir / "models" / "final.pt")


def load_checkpoint(run_dir: Path) -> tuple[dict[str, Any], list[Episode]] | None:
    """
    Newest checkpoint and its replay buffer, dropping everything collected after
    it. None if the run has no checkpoint yet
    """
    paths = sorted(run_dir.glob("checkpoint_*.pt"), reverse=True)
    if not paths:
        return None
    data = torch.load(run_dir / "episodes.pt")

    for path in paths:
        checkpoint = torch.load(path, map_location="cpu")

        # Transitions are stamped with the epoch they were collected in and episodes
        # are chronological, so this only ever cuts off the tail
        episodes: list[Episode] = []
        for episode in data:
            kept = int((episode["epochs"] <= checkpoint["epoch"]).sum())
            if kept == 0:
                continue
            episodes.append(
                Episode(
                    observations=list(episode["observations"][: kept + 1]),
                    actions=episode["actions"][:kept].tolist(),
                    rewards=episode["rewards"][:kept].tolist(),
                    terminations=episode["terminations"][:kept].tolist(),
                    epochs=episode["epochs"][:kept].tolist(),
                )
            )

        transitions = sum(len(episode.actions) for episode in episodes)
        if transitions == checkpoint["collector"]["env_steps"]:
            return checkpoint, episodes

    # Note that starting over would wipe the run, so stop instead
    raise ValueError(f"no checkpoint in {run_dir} matches its episodes.pt")


def restore(
    checkpoint: dict[str, Any],
    episodes: list[Episode],
    agent: Agent,
    optimizers: Optimizers,
    replay_buffer: ReplayBuffer,
    collector: Collector,
) -> None:
    for name in checkpoint["models"]:
        getattr(agent, name).load_state_dict(checkpoint["models"][name])
        getattr(optimizers, name).load_state_dict(checkpoint["optimizers"][name])

    # Note that the episode in progress at the checkpoint is kept but ends here since
    # the game itself can't be restored
    for episode_id, episode in enumerate(episodes):
        replay_buffer.update_episode(episode, episode_id)

    collector.env_steps = checkpoint["collector"]["env_steps"]
    collector.game_count = checkpoint["collector"]["game_count"]
    collector.reward_pos = checkpoint["collector"]["reward_pos"]
    collector.reward_neg = checkpoint["collector"]["reward_neg"]
    collector.episode_count = len(episodes)


def purge(run_dir: Path, checkpoint: dict[str, Any] | None) -> None:
    """
    Deletes everything written after the checkpoint (or everything without one) so
    a resumed run looks like one continuous run
    """
    counters = checkpoint["counters"] if checkpoint is not None else {}
    epoch = checkpoint["epoch"] if checkpoint is not None else -1

    def keep(row: dict[str, Any]) -> bool:
        # A final eval only happens once the run is done so it never survives
        after = checkpoint is None or any(
            row[key] > value for key, value in counters.items()
        )
        final = row.get("kind") == "eval_final" or any(
            key.startswith("eval_final/") for key in row
        )
        return not after and not final

    for name in ("metrics.jsonl", "games.jsonl"):
        path = run_dir / name
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        tmp_path = path.with_suffix(".tmp")
        kept = [json.dumps(row) + "\n" for row in rows if keep(row)]
        tmp_path.write_text("".join(kept))
        tmp_path.replace(path)

    # Files named by epoch, e.g. models/epoch_0050.pt and media/media_real_game_0050.gif
    for path in [*run_dir.glob("models/epoch_*.pt"), *run_dir.glob("media/*.gif")]:
        if int(path.stem.rsplit("_", 1)[1]) > epoch:
            path.unlink()

    # Checkpoints newer than the one resumed from could not be used
    for path in run_dir.glob("checkpoint_*.pt"):
        if int(path.stem.rsplit("_", 1)[1]) > epoch:
            path.unlink()

    for path in [*run_dir.glob("*.tmp"), *run_dir.glob("models/*.tmp")]:
        path.unlink()
    if checkpoint is None:
        (run_dir / "episodes.pt").unlink(missing_ok=True)


def atomic_save(obj: Any, path: Path) -> None:
    # Write then rename so a crash or timeout never leaves a half written file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(obj, tmp_path)
    tmp_path.replace(path)
