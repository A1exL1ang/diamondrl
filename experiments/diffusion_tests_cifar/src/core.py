import io
import time
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import make_grid

STEP_LIST = (1, 2, 4, 8, 16, 32)
NUM_RENDER_SAMPLES = 8


class Method(ABC):
    @abstractmethod
    def loss(
        self, model: nn.Module, ema: nn.Module, imgs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        "loss" is backpropagated and "report" is logged, which is usually just
        loss.detach() but e.g. MeanFlow's flow matching MSE
        """
        ...

    @abstractmethod
    def sample(self, model: nn.Module, num_sample: int, steps: int) -> torch.Tensor: ...


def compute_fid(
    method: Method,
    model: nn.Module,
    metric: FrechetInceptionDistance,
    batches: int,
    batch_size: int,
    chunk_size: int,
    steps_list: tuple[int, ...] = STEP_LIST,
) -> list[float]:
    fids = []

    for steps in steps_list:
        # Only clears the generated features since reset_real_features=False
        metric.reset()

        for _ in range(batches):
            imgs = method.sample(model, batch_size, steps=steps).clamp(
                min=-1.0, max=1.0
            )
            # Note: chunked since Inception on the whole batch can run out of memory
            for chunk in imgs.split(chunk_size):
                metric.update((chunk + 1) / 2, real=False)

        fids.append(metric.compute().item())

    return fids


def render_samples(
    method: Method,
    model: nn.Module,
    epoch: int,
    steps_list: tuple[int, ...] = STEP_LIST,
    num_samples: int = NUM_RENDER_SAMPLES,
) -> bytes:
    # Same seed for every row so only the number of steps differs
    # Note: forked so reseeding doesn't reset the training RNG after every eval
    rows = []
    with torch.random.fork_rng():
        for steps in steps_list:
            torch.manual_seed(0)
            rows.append(method.sample(model, num_samples, steps=steps))

    # Rows are step counts, columns are samples
    imgs = torch.cat(rows).add(1).div(2).clamp(0, 1)
    grid = make_grid(imgs, nrow=num_samples, padding=2)

    plt.figure(figsize=(num_samples * 0.9, len(steps_list) * 0.9))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.yticks(
        [34 * i + 18 for i in range(len(steps_list))],
        [f"{s} steps" for s in steps_list],
    )
    plt.xticks([])
    plt.title(f"epoch {epoch}")

    # PNG bytes so the figure can be sent back from Modal
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    return buf.getvalue()


def train(
    model: nn.Module,
    method: Method,
    loader: DataLoader,
    epochs: int,
    eval_every: int,
    lr: float,
    min_lr: float,
    warmup_steps: int,
    weight_decay: float,
    ema_decay: float,
    fid_batches: int,
    fid_batch_size: int,
    fid_chunk_size: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup to lr, then cosine decay to min_lr by the last step
    total_steps = epochs * len(loader)
    schedule = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(
                optimizer,
                start_factor=1 / warmup_steps,
                end_factor=1.0,
                total_iters=warmup_steps,
            ),
            CosineAnnealingLR(
                optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
            ),
        ],
        milestones=[warmup_steps],
    )

    # Decay ramp: (1 + n) / (10 + n)
    ema_model = AveragedModel(
        model=model,
        avg_fn=lambda avg, param, n: avg.lerp(
            param, 1 - min((1 + n) / (10 + n), ema_decay)
        ),
        device=device,
    )

    fid = FrechetInceptionDistance(
        feature=2048, normalize=True, reset_real_features=False
    ).to(device)

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device_type == "cuda" else torch.float32

    for img, _ in loader:
        fid.update((img.to(device) + 1) / 2, real=True)

    for epoch in range(epochs):
        model.train()
        losses = []
        grad_norms = []
        start = time.perf_counter()

        for img, label in loader:
            img, label = img.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                out = method.loss(model, ema_model.module, img)

            out["loss"].backward()

            # Note: for losses with large per-sample weights, e.g. v-loss on D
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            schedule.step()

            ema_model.update_parameters(model)

            losses.append(out["report"].item())

            # Note: the norm before clipping, to catch spikes
            grad_norms.append(grad_norm.item())

        row = {
            "epoch": epoch,
            "loss": sum(losses) / len(losses),
            "grad_norm": sum(grad_norms) / len(grad_norms),
            "grad_norm_max": max(grad_norms),
            "lr": schedule.get_last_lr()[0],
            "train_sec": time.perf_counter() - start,
            "losses": losses,
            "grad_norms": grad_norms,
        }
        png = None

        if (epoch + 1) % eval_every == 0:
            start = time.perf_counter()
            fids = compute_fid(
                method,
                ema_model.module,
                fid,
                batches=fid_batches,
                batch_size=fid_batch_size,
                chunk_size=fid_chunk_size,
            )
            row["fid"] = dict(zip(STEP_LIST, fids, strict=True))
            row["fid_samples"] = fid_batches * fid_batch_size
            row["fid_sec"] = time.perf_counter() - start
            png = render_samples(method, ema_model.module, epoch)

        yield row, png
