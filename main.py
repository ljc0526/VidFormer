import argparse
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import heartpy
import matplotlib.pyplot as plt
import numpy as np
import torch
from audtorch.metrics.functional import pearsonr
from torch import nn
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_preprocess import Dataset_Base
from VidTransformer import Vidformer as Vi

warnings.filterwarnings("ignore")


@dataclass
class TrainConfig:
    dataset_root: Path
    checkpoint_dir: Path
    dataset_name: str = "PURE"
    data_key: str = "video"
    label_key: str = "GT_ppg"
    batch_size: int = 2
    epochs: int = 150
    learning_rate: float = 8e-5
    weight_decay: float = 5e-4
    split_number: int = 250
    window_length: int = 250
    test_ratio: float = 0.25
    num_workers: int = 0
    seed: int = 1314520
    sample_rate: int = 30
    output_size: int = 128
    epoch_test: int = 300
    gpu_id: Optional[int] = None


def seed_everything(seed: int = 1314520) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_device(gpu_id: Optional[int]) -> torch.device:
    if torch.cuda.is_available():
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cuda")
    return torch.device("cpu")


class TemporalLoss(nn.Module):
    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pearsonr(prediction, target, batch_first=True)
        loss = torch.sum(1 - loss)
        if torch.isnan(loss):
            return torch.zeros((), device=prediction.device, dtype=prediction.dtype)
        return loss


def to_numpy_1d(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float64).reshape(-1)


def estimate_hr_with_fft_fallback(
    signal,
    sample_rate: int,
    bpm_min: int = 30,
    bpm_max: int = 180,
    window_size: int = 8,
    do_filter: bool = False,
    filter_order: int = 2,
) -> Dict[str, float]:
    signal = to_numpy_1d(signal)

    if do_filter:
        try:
            signal = heartpy.filter_signal(
                data=signal,
                sample_rate=sample_rate,
                cutoff=[0.75, 2.75],
                filtertype="bandpass",
                order=filter_order,
            )
        except Exception:
            pass

    try:
        _, metrics = heartpy.process(
            signal,
            sample_rate=sample_rate,
            bpmmax=bpm_max,
            bpmmin=bpm_min,
            windowsize=window_size,
        )
        bpm = metrics.get("bpm", np.nan)
        if (not np.isfinite(bpm)) or bpm < bpm_min or bpm > bpm_max:
            raise ValueError("HeartPy returned an invalid bpm.")
        breathing_rate = metrics.get("breathingrate", np.nan)
        if not np.isfinite(breathing_rate):
            breathing_rate = np.nan
        return {"bpm": float(bpm), "breathingrate": float(breathing_rate)}
    except Exception:
        bpm = estimate_hr_by_fft(signal, sample_rate, bpm_min, bpm_max)
        return {"bpm": bpm, "breathingrate": np.nan}


def estimate_hr_by_fft(
    signal: np.ndarray,
    sample_rate: int,
    bpm_min: int = 30,
    bpm_max: int = 180,
    nfft: int = 2048,
) -> float:
    signal = to_numpy_1d(signal)
    if signal.size < 16:
        return np.nan

    signal = signal - np.mean(signal)
    std = np.std(signal)
    if (not np.isfinite(std)) or std < 1e-8:
        return np.nan

    signal = signal / (std + 1e-8)
    signal = signal * np.hanning(len(signal))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal, n=nfft)) ** 2

    low_hz = bpm_min / 60.0
    high_hz = bpm_max / 60.0
    valid_mask = (freqs >= low_hz) & (freqs <= high_hz)
    if valid_mask.sum() == 0:
        return np.nan

    peak_freq = freqs[valid_mask][np.argmax(spectrum[valid_mask])]
    bpm = float(peak_freq * 60.0)
    return bpm if np.isfinite(bpm) else np.nan


def summarize_errors(errors: List[float]) -> Dict[str, float]:
    if not errors:
        return {"mae": float("nan"), "rmse": float("nan")}
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
    }


def collect_batch_errors(
    labels: torch.Tensor,
    predictions: torch.Tensor,
    sample_rate: int,
    pred_filter_order: int,
) -> Dict[str, List[float]]:
    hr_errors: List[float] = []
    rr_errors: List[float] = []

    for index in range(labels.shape[0]):
        label_metrics = estimate_hr_with_fft_fallback(
            labels[index],
            sample_rate=sample_rate,
            bpm_min=30,
            bpm_max=180,
            window_size=8,
            do_filter=False,
            filter_order=2,
        )
        pred_metrics = estimate_hr_with_fft_fallback(
            predictions[index],
            sample_rate=sample_rate,
            bpm_min=30,
            bpm_max=180,
            window_size=8,
            do_filter=True,
            filter_order=pred_filter_order,
        )

        if np.isfinite(label_metrics["bpm"]) and np.isfinite(pred_metrics["bpm"]):
            hr_errors.append(round(label_metrics["bpm"]) - round(pred_metrics["bpm"]))

        if np.isfinite(label_metrics["breathingrate"]) and np.isfinite(pred_metrics["breathingrate"]):
            rr_errors.append(label_metrics["breathingrate"] - pred_metrics["breathingrate"])

    return {
        "hr_errors": hr_errors,
        "rr_errors": rr_errors,
    }


def build_model(device: torch.device) -> nn.Module:
    model = Vi.Vidformer(
        patch_size=(25, 16, 16),
        image_size=(250, 128, 128),
        in_channels=3,
        out_channels=64,
        emd_dim=64,
        drop_out=0.0,
        depth=(1, 1, 4, 4),
        heads=6,
        dim_head=64,
        mlp_dim=256,
    )
    return model.to(device)


def build_loaders(config: TrainConfig):
    return Dataset_Base.build_dataloaders(
        root=str(config.dataset_root),
        Dataset_name=config.dataset_name,
        data_key=config.data_key,
        label_key=config.label_key,
        split_number=config.split_number,
        window_length=config.window_length,
        test_ratio=config.test_ratio,
        batch_size=config.batch_size,
        seed=config.seed,
        num_workers=config.num_workers,
        fixed_length=config.window_length,
        sample_rate=config.sample_rate,
    )


def train_one_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    temporal_loss_fn: nn.Module,
    waveform_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> Dict[str, float]:
    model.train()
    conv_losses: List[float] = []
    trans_losses: List[float] = []
    total_losses: List[float] = []
    hr_errors: List[float] = []
    rr_errors: List[float] = []

    progress = tqdm(train_loader, desc=f"Train {epoch + 1}/{config.epochs}", leave=False)
    for batch in progress:
        videos, labels, _ = batch
        videos = videos.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        batch_size = videos.shape[0]

        optimizer.zero_grad(set_to_none=True)
        conv_out, trans_out = model(videos, epoch, batch_size, config.epoch_test)

        conv_loss = temporal_loss_fn(conv_out, labels)*0.5 + waveform_loss_fn(conv_out, labels)*0.5
        trans_loss = temporal_loss_fn(trans_out, labels)*0.5 + waveform_loss_fn(trans_out, labels)*0.5
        total_loss = conv_loss + trans_loss

        total_loss.backward()
        optimizer.step()

        prediction = 0.5 * (conv_out + trans_out)
        batch_errors = collect_batch_errors(
            labels=labels,
            predictions=prediction,
            sample_rate=config.sample_rate,
            pred_filter_order=2,
        )

        conv_losses.append(float(conv_loss.item()))
        trans_losses.append(float(trans_loss.item()))
        total_losses.append(float(total_loss.item()))
        hr_errors.extend(batch_errors["hr_errors"])
        rr_errors.extend(batch_errors["rr_errors"])

        progress.set_postfix(loss=f"{np.mean(total_losses):.4f}")
    scheduler.step()
    hr_summary = summarize_errors(hr_errors)
    rr_summary = summarize_errors(rr_errors)
    return {
        "lr": float(scheduler.get_last_lr()[0]),
        "conv_loss": float(np.mean(conv_losses)) if conv_losses else float("nan"),
        "trans_loss": float(np.mean(trans_losses)) if trans_losses else float("nan"),
        "total_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "hr_mae": hr_summary["mae"],
        "hr_rmse": hr_summary["rmse"],
        "rr_mae": rr_summary["mae"],
        "rr_rmse": rr_summary["rmse"],
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader,
    temporal_loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    config: TrainConfig,
) -> Dict[str, float]:
    model.eval()
    conv_losses: List[float] = []
    trans_losses: List[float] = []
    hr_errors: List[float] = []
    rr_errors: List[float] = []

    progress = tqdm(data_loader, desc=f"Eval {epoch + 1}/{config.epochs}", leave=False)
    for batch in progress:
        videos, labels, _ = batch
        videos = videos.to(device=device, dtype=torch.float32)
        labels = labels.to(device=device, dtype=torch.float32)
        batch_size = videos.shape[0]

        conv_out, trans_out = model(videos, epoch, batch_size, config.epoch_test)
        conv_losses.append(float(temporal_loss_fn(conv_out, labels).item()))
        trans_losses.append(float(temporal_loss_fn(trans_out, labels).item()))

        prediction = 0.5 * (conv_out + trans_out)
        batch_errors = collect_batch_errors(
            labels=labels,
            predictions=prediction,
            sample_rate=config.sample_rate,
            pred_filter_order=3,
        )

        hr_errors.extend(batch_errors["hr_errors"])
        rr_errors.extend(batch_errors["rr_errors"])

    hr_summary = summarize_errors(hr_errors)
    rr_summary = summarize_errors(rr_errors)
    return {
        "conv_loss": float(np.mean(conv_losses)) if conv_losses else float("nan"),
        "trans_loss": float(np.mean(trans_losses)) if trans_losses else float("nan"),
        "total_loss": float(np.mean(conv_losses) + np.mean(trans_losses)) if conv_losses and trans_losses else float("nan"),
        "hr_mae": hr_summary["mae"],
        "hr_rmse": hr_summary["rmse"],
        "rr_mae": rr_summary["mae"],
        "rr_rmse": rr_summary["rmse"],
    }


def save_training_artifacts(
    history: Dict[str, List[float]],
    config: TrainConfig,
    final_metrics: Dict[str, float],
) -> None:
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (config.checkpoint_dir / "train_config.json").open("w", encoding="utf-8") as handle:
        serializable_config = asdict(config)
        serializable_config["dataset_root"] = str(config.dataset_root)
        serializable_config["checkpoint_dir"] = str(config.checkpoint_dir)
        json.dump(serializable_config, handle, indent=2)

    with (config.checkpoint_dir / "history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    with (config.checkpoint_dir / "final_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_metrics, handle, indent=2)

    for key in ("train_total_loss", "train_hr_mae"):
        values = history.get(key, [])
        if not values:
            continue
        plt.figure()
        plt.plot(values)
        plt.title(key)
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(config.checkpoint_dir / f"{key}.png", dpi=200)
        plt.close()


def train_and_evaluate(config: TrainConfig) -> None:
    seed_everything(config.seed)
    device = get_device(config.gpu_id)
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, test_loader = build_loaders(config)
    model = build_model(device)
    temporal_loss_fn = TemporalLoss()
    waveform_loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        betas=(0.9, 0.99),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=2e-9,
    )

    history: Dict[str, List[float]] = {
        "train_total_loss": [],
        "train_hr_mae": [],
    }

    for epoch in range(config.epochs):
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            temporal_loss_fn=temporal_loss_fn,
            waveform_loss_fn=waveform_loss_fn,
            device=device,
            epoch=epoch,
            config=config,
        )

        history["train_total_loss"].append(train_metrics["total_loss"])
        history["train_hr_mae"].append(train_metrics["hr_mae"])

        print(
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"lr={train_metrics['lr']:.6e} | "
            f"train_loss={train_metrics['total_loss']:.6f} | "
            f"train_hr_mae={train_metrics['hr_mae']:.6f} | "
            f"train_hr_rmse={train_metrics['hr_rmse']:.6f}"
        )

    final_metrics = evaluate(
        model=model,
        data_loader=test_loader,
        temporal_loss_fn=temporal_loss_fn,
        device=device,
        epoch=config.epochs - 1,
        config=config,
    )
    torch.save(model.state_dict(), config.checkpoint_dir / "final_model.pt")

    print(
        "Final evaluation | "
        f"eval_loss={final_metrics['total_loss']:.6f} | "
        f"eval_hr_mae={final_metrics['hr_mae']:.6f} | "
        f"eval_hr_rmse={final_metrics['hr_rmse']:.6f} | "
        f"eval_rr_mae={final_metrics['rr_mae']:.6f} | "
        f"eval_rr_rmse={final_metrics['rr_rmse']:.6f}"
    )

    save_training_artifacts(history, config, final_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate the small VidFormer model.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of the prepared dataset.")
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=REPO_ROOT / "result" / "small_model_test",
        help="Directory used to save checkpoints and logs.",
    )
    parser.add_argument("--dataset-name", type=str, default="PURE", help="Dataset name used by Dataset_Base.")
    parser.add_argument("--data-key", type=str, default="video", help="Data key passed to Dataset_Base.")
    parser.add_argument("--label-key", type=str, default="GT_ppg", help="Label key passed to Dataset_Base.")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=8e-5, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay for AdamW.")
    parser.add_argument("--split-number", type=int, default=250, help="Sliding-window stride used by Dataset_Base.")
    parser.add_argument("--window-length", type=int, default=250, help="Sequence length used by Dataset_Base.")
    parser.add_argument("--test-ratio", type=float, default=0.25, help="Test split ratio.")
    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader workers.")
    parser.add_argument("--seed", type=int, default=1314520, help="Random seed.")
    parser.add_argument("--sample-rate", type=int, default=30, help="Signal sampling rate used for HR estimation.")
    parser.add_argument("--epoch-test", type=int, default=300, help="Extra epoch argument passed to the model forward method.")
    parser.add_argument("--gpu-id", type=int, default=0, help="Optional CUDA device id.")
    return parser


def parse_args() -> TrainConfig:
    args = build_parser().parse_args()
    return TrainConfig(
        dataset_root=args.dataset_root.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        dataset_name=args.dataset_name,
        data_key=args.data_key,
        label_key=args.label_key,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        split_number=args.split_number,
        window_length=args.window_length,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
        seed=args.seed,
        sample_rate=args.sample_rate,
        epoch_test=args.epoch_test,
        gpu_id=args.gpu_id,
    )


if __name__ == "__main__":
    train_and_evaluate(parse_args())
