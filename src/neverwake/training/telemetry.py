import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

from neverwake.config.core import Config
from neverwake.rl.rollout import Trajectory
from neverwake.training.collector import Collector
from neverwake.training.eval import Evaluator
from neverwake.training.optim import ModelOptimizer, Optimizers


class MetricLog:
    """
    Aggregates values and writes each flush as one row to metrics.jsonl and W&B
    """

    def __init__(
        self,
        run_dir: Path,
        config: dict[str, Any],
        axes: dict[str, str | None],
        project: str,
        mode: Literal["online", "offline", "disabled"],
        stats_interval_sec: float,
        run_id: str,
    ):
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.metrics_file = open(run_dir / "metrics.jsonl", "a")

        # A resumed run continues the same W&B run. Note that W&B samples GPU/CPU
        # utilization, memory, power and temps on its own
        self.run = wandb.init(
            project=project,
            name=run_dir.name,
            id=run_id,
            resume="allow",
            config=config,
            dir=run_dir,
            mode=mode,
            settings=wandb.Settings(x_stats_sampling_interval=stats_interval_sec),
        )

        # Metric (or glob) -> its x-axis. Note that an x-axis has to be defined before
        # anything is plotted against it
        for metric, step_metric in axes.items():
            self.run.define_metric(metric, step_metric=step_metric)

        # State
        # key -> (agg, aggregated value)
        self.pending: dict[str, tuple[Literal["mean", "sum", "max", "last"], Any]] = {}
        self.latest: dict[str, float] = {}

    def add(
        self, key: str, value: Any, agg: Literal["mean", "sum", "max", "last"] = "mean"
    ) -> None:
        # Note that tensors stay on device so we don't sync with the GPU every step
        if torch.is_tensor(value):
            value = value.detach()
        kind, old = self.pending.get(key, (agg, None))

        if kind == "mean":
            total, count = old or (0.0, 0)
            new = (total + value, count + 1)
        elif kind == "sum":
            new = value if old is None else old + value
        elif kind == "max":
            if old is None:
                new = value
            elif torch.is_tensor(value):
                new = torch.maximum(old, value)
            else:
                new = max(old, value)
        else:  # last
            new = value

        self.pending[key] = (kind, new)

    def add_dict(self, prefix: str, values: dict[str, Any]) -> None:
        for name, value in values.items():
            self.add(f"{prefix}/{name}", value)

    def flush(
        self, counters: dict[str, int], prefixes: tuple[str, ...] | None = None
    ) -> dict[str, float]:
        """
        Writes the pending keys (only under prefixes if given) as one row
        """
        keys = [
            key
            for key in self.pending
            if prefixes is None or key.split("/")[0] in prefixes
        ]

        # Reduce
        scalars: dict[str, float] = {}
        media: dict[str, Any] = {}
        for key in keys:
            kind, value = self.pending.pop(key)
            if kind == "mean":
                value = value[0] / value[1]
            if getattr(value, "ndim", 0) == 0:
                scalars[key] = float(value)
            else:
                media[key] = self.to_media(key, value, counters["progress/epoch"])

        # Write
        row = {**counters, **scalars}
        self.metrics_file.write(json.dumps(row) + "\n")
        self.metrics_file.flush()
        self.run.log({**row, **media})
        self.latest.update(scalars)
        return scalars

    def close(self) -> None:
        self.metrics_file.close()
        self.run.finish()

    def to_media(self, key: str, value: Any, epoch: int) -> Any:
        # (H, W, C) image or (T, H, W, C) video
        frames = np.asarray(value)
        if frames.ndim == 3:
            return wandb.Image(frames)

        # Videos are written as GIFs, which avoids wandb's moviepy dependency
        name = key.replace("/", "_")
        path = self.run_dir / "media" / f"{name}_{epoch:04d}.gif"
        path.parent.mkdir(exist_ok=True)
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            path, save_all=True, append_images=images[1:], duration=66, loop=0
        )
        return wandb.Video(str(path))


class Telemetry:
    """
    Records the training run, reading counters and stats off the optimizers,
    collector and evaluator directly
    """

    def __init__(
        self,
        run_dir: Path,
        cfg: Config,
        optimizers: Optimizers,
        collector: Collector,
        evaluator: Evaluator,
        epoch: int,
        run_id: str,
    ):
        self.cfg = cfg.telemetry
        self.eval_cfg = cfg.train.eval
        self.optimizers = optimizers
        self.collector = collector
        self.evaluator = evaluator

        # The counters are plotted against the epoch, epoch prefixes against env steps
        # and each model prefix (e.g. "denoiser") against its own update counter
        self.epoch_prefixes = ("collect", "eval", "eval_final", "sys", "media")
        self.model_names = [f.name for f in fields(optimizers)]
        axes: dict[str, str | None] = {
            "progress/epoch": None,
            "progress/env_steps": "progress/epoch",
            **{
                f"progress/updates_{name}": "progress/epoch"
                for name in self.model_names
            },
            **{f"{prefix}/*": "progress/env_steps" for prefix in self.epoch_prefixes},
            **{f"{name}/*": f"progress/updates_{name}" for name in self.model_names},
        }
        self.log = MetricLog(
            run_dir=run_dir,
            config=asdict(cfg),
            axes=axes,
            project=self.cfg.project,
            mode=self.cfg.mode,
            stats_interval_sec=self.cfg.stats_interval_sec,
            run_id=run_id,
        )
        self.games_file = open(run_dir / "games.jsonl", "a")

        # Tracking
        self.epoch = epoch
        self.games_logged = collector.game_count  # Collector games already recorded
        self.imagined_obs: torch.Tensor | None = None  # Latest rollouts for videos

        # Timing
        self.epoch_start = self.last_lap = synced_time()

    def update(
        self,
        name: str,
        loss: torch.Tensor,
        info: dict[str, torch.Tensor],
        grad_norm: torch.Tensor,
    ) -> None:
        """
        One optimizer step of model `name`, writes a row every flush_every steps
        """
        self.log.add_dict(name, {"loss": loss, "grad_norm": grad_norm, **info})
        self.log.add(f"{name}/grad_norm_max", grad_norm, "max")

        optimizer: ModelOptimizer = getattr(self.optimizers, name)
        if optimizer.num_updates % self.cfg.flush_every == 0:
            self.log.add(f"{name}/lr", optimizer.schedule.get_last_lr()[0], "last")
            # Note that rows logged mid epoch belong to the epoch in progress
            self.log.flush(self.counters(self.epoch + 1), prefixes=(name,))

    def imagination(self, traj: Trajectory) -> None:
        """
        An imagined rollout for the actor critic, call it before the update so the
        rates land in the same row
        """
        reward_rate = (traj.rewards[traj.valid] != 0).float().mean()
        end_rate = traj.ends[traj.valid].float().mean()
        self.log.add("actor_critic/imagined_reward_rate", reward_rate)
        self.log.add("actor_critic/imagined_end_rate", end_rate)
        self.imagined_obs = traj.obs[: self.cfg.video_clips].detach()

    def lap(self, phase: str) -> None:
        """
        Time since the previous lap (or epoch start) goes to sys/sec_<phase>
        """
        now = synced_time()
        self.log.add(f"sys/sec_{phase}", now - self.last_lap, "sum")
        self.last_lap = now

    def end_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        self.record_collection()

        # Eval and videos when due
        if epoch % self.eval_cfg.every == 0:
            self.record_eval("eval", self.eval_cfg.num_games)
            self.lap("eval")
        if epoch % self.cfg.video_every == 0:
            self.record_videos()
            self.lap("video")

        # System
        self.log.add("sys/sec_epoch", synced_time() - self.epoch_start, "last")
        self.record_gpu_memory()

        # Note that model rows are not flushed here so their windows stay intact
        row = self.log.flush(self.counters(epoch), prefixes=self.epoch_prefixes)
        self.print_epoch(row)

        self.epoch_start = self.last_lap = synced_time()

    def final_eval(self) -> None:
        self.record_eval("eval_final", self.eval_cfg.num_games_final)
        self.lap("eval_final")

        # Flush everything including any partial model windows
        row = self.log.flush(self.counters(self.epoch))

        nan = float("nan")
        print(
            f"final | eval return {row.get('eval_final/return', nan):.1f} "
            f"(max {row.get('eval_final/return_max', nan):.1f}) over "
            f"{self.eval_cfg.num_games_final} games, "
            f"{row.get('eval_final/cut_off', 0):.0f} cut off"
        )

    def close(self) -> None:
        self.games_file.close()
        self.log.close()

    def counters(self, epoch: int) -> dict[str, int]:
        return {
            "progress/epoch": epoch,
            "progress/env_steps": self.collector.env_steps,
            **{
                f"progress/updates_{name}": getattr(self.optimizers, name).num_updates
                for name in self.model_names
            },
        }

    def record_game(self, kind: str, ret: float, length: int) -> None:
        record = {
            **self.counters(self.epoch),
            "kind": kind,
            "return": ret,
            "length": length,
        }
        self.games_file.write(json.dumps(record) + "\n")
        self.games_file.flush()

        self.log.add(f"{kind}/return", ret)
        self.log.add(f"{kind}/return_max", ret, "max")
        self.log.add(f"{kind}/length", length)

    def record_collection(self) -> None:
        collector = self.collector
        for game in range(self.games_logged, collector.game_count):
            length = collector.game_length[game]
            self.record_game("collect", collector.game_reward_sum[game], length)
            self.log.add("collect/entropy", collector.game_entropy_sum[game] / length)
        self.games_logged = collector.game_count
        self.log.add("collect/lives", collector.episode_count, "last")

    def record_eval(self, kind: str, num_games: int) -> None:
        summary = self.evaluator.evaluate(num_games)
        for ret, length in zip(summary.returns, summary.lengths, strict=True):
            self.record_game(kind, ret, length)
        self.log.add(f"{kind}/cut_off", summary.finished.count(False), "sum")

    def record_videos(self) -> None:
        scale = self.cfg.video_scale
        max_frames = self.cfg.video_max_frames

        # Real games side by side, each holding its last frame once it ends
        games = [
            self.evaluator.play_game(max_frames).obs[:max_frames]
            for _ in range(self.cfg.video_clips)
        ]
        steps = max(len(game) for game in games)
        held = [game[-1:].expand(steps - len(game), -1, -1, -1) for game in games]
        real = torch.stack([torch.cat(pair) for pair in zip(games, held, strict=True)])
        self.log.add("media/real_game", to_video(real, scale), "last")
        if self.imagined_obs is not None:
            self.log.add("media/imagined", to_video(self.imagined_obs, scale), "last")

    def record_gpu_memory(self) -> None:
        if not torch.cuda.is_available():
            return

        # Note that the peak is reset so it is per epoch
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        reserved_gb = torch.cuda.memory_reserved() / 2**30
        self.log.add("sys/gpu_mem_peak_gb", peak_gb, "last")
        self.log.add("sys/gpu_mem_reserved_gb", reserved_gb, "last")
        torch.cuda.reset_peak_memory_stats()

    def print_epoch(self, row: dict[str, float]) -> None:
        console = {
            "ret": "collect/return",
            "eval": "eval/return",
            "den": "denoiser/loss",
            "rew": "reward_end/loss",
            "ent": "actor_critic/entropy",
        }
        metrics = " | ".join(
            f"{label} {self.log.latest.get(key, float('nan')):.3g}"
            for label, key in console.items()
        )
        phases = " ".join(
            f"{key.removeprefix('sys/sec_')} {value:.1f}"
            for key, value in row.items()
            if key.startswith("sys/sec_") and key != "sys/sec_epoch"
        )
        print(
            f"epoch {self.epoch:4d} | env {self.collector.env_steps:7,d} | {metrics} | "
            f"{row['sys/sec_epoch']:.1f}s ({phases})"
        )


def synced_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def to_video(obs: torch.Tensor, scale: int) -> torch.Tensor:
    """
    (B, T, C, H, W) in [-1, 1] --> (T, scale * H, B * scale * W, C) uint8 with the
    clips side by side. Each pixel becomes a scale x scale block so nothing blurs
    """
    batch, steps = obs.shape[:2]

    # (B * T, C, H, W) --> (B * T, C, scale * H, scale * W) --> (B, T, C, ...)
    frames = F.interpolate(obs.detach().flatten(0, 1), scale_factor=scale)
    frames = ((frames.unflatten(0, (batch, steps)).clamp(-1, 1) + 1) * 127.5).round()

    # (B, T, C, H, W) --> (T, H, B, W, C) --> (T, H, B * W, C)
    return frames.byte().permute(1, 3, 0, 4, 2).flatten(2, 3).cpu()
