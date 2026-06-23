#!/usr/bin/env python3
import argparse
import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


SC_ROOT = Path("/home/DM24/workspace/Time_Series_Forecasting/SC_FTMixer_SDE_Unified")
REPOS = {
    "SC-FSD": SC_ROOT,
    "XLinear": Path("/home/DM24/workspace/Time_Series_Forecasting/XLinear-main"),
    "TEFN": Path("/home/DM24/workspace/Time_Series_Forecasting/Time-Evidence-Fusion-Network-master/Time-Evidence-Fusion-Network-master"),
    "iTransformer": Path("/home/DM24/workspace/Time_Series_Forecasting/Time-Series-Library-main"),
    "PatchTST": Path("/home/DM24/workspace/Time_Series_Forecasting/Time-Series-Library-main"),
    "DLinear": Path("/home/DM24/workspace/Time_Series_Forecasting/Time-Series-Library-main"),
}


class ECLWindows(Dataset):
    def __init__(self, csv_path, seq_len=96, label_len=48, pred_len=96):
        frame = pd.read_csv(csv_path)
        values = frame.iloc[:, 1:].values.astype(np.float32)
        dates = pd.DatetimeIndex(pd.to_datetime(frame.iloc[:, 0]))
        train_end = int(len(frame) * 0.7)
        scaler = StandardScaler()
        scaler.fit(values[:train_end])
        self.values = scaler.transform(values[:train_end]).astype(np.float32)
        self.dates = dates[:train_end]
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

    @staticmethod
    def marks(dates):
        return np.stack([
            dates.hour.to_numpy() / 23.0 - 0.5,
            dates.dayofweek.to_numpy() / 6.0 - 0.5,
            (dates.day.to_numpy() - 1) / 30.0 - 0.5,
            (dates.dayofyear.to_numpy() - 1) / 365.0 - 0.5,
        ], axis=-1).astype(np.float32)

    def __len__(self):
        return len(self.values) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        s0 = index
        s1 = s0 + self.seq_len
        r0 = s1 - self.label_len
        r1 = r0 + self.label_len + self.pred_len
        x = self.values[s0:s1]
        y_all = self.values[r0:r1]
        x_mark = self.marks(self.dates[s0:s1])
        y_mark = self.marks(self.dates[r0:r1])
        target = y_all[-self.pred_len:]
        return x, target, x_mark, y_mark


def make_config(model_name):
    common = dict(
        task_name="long_term_forecast", seq_len=96, label_len=48,
        pred_len=96, enc_in=321, dec_in=321, c_out=321,
        features="M", output_attention=False,
    )
    if model_name == "SC-FSD":
        return SimpleNamespace(
            **common, sde_hidden=192, sde_cycle_len=168, freq="h",
            sde_phase_mode="hour_week", sde_slots_per_hour=1,
            use_sde=1, use_global_sde=1, use_calendar_sde=1,
            use_dynamic_filter=1, fix_calendar_gate=0,
            sde_calendar_gate_init=2.0,
        )
    if model_name == "XLinear":
        return SimpleNamespace(
            **common, d_model=2048, t_ff=512, c_ff=32, usenorm=1,
            embed_dropout=0.2, head_dropout=0.2,
            t_dropout=0.2, c_dropout=0.1,
        )
    if model_name == "TEFN":
        return SimpleNamespace(
            **common, use_norm=True, use_T_model=True, use_C_model=True,
            fusion_method="add", use_probabilistic_layer=False,
            e_layers=2, kernel_activation="linear",
            use_residual=True, dropout=0.1,
        )
    return SimpleNamespace(
        **common, d_model=512, n_heads=8,
        e_layers=3 if model_name == "iTransformer" else 2,
        d_ff=512 if model_name == "iTransformer" else 2048,
        factor=3, dropout=0.1, activation="gelu",
        embed="timeF", freq="h", moving_avg=25,
        num_class=2,
    )


def load_model(model_name):
    repo = REPOS[model_name]
    sys.path.insert(0, str(repo))
    module_names = {
        "SC-FSD": "models.SC_FTMixer_SDE_Unified",
        "XLinear": "models.XLinear",
        "TEFN": "models.TEFN",
        "iTransformer": "models.iTransformer",
        "PatchTST": "models.PatchTST",
        "DLinear": "models.DLinear",
    }
    module = importlib.import_module(module_names[model_name])
    return module.Model(make_config(model_name))


def forward_model(model_name, model, x, x_mark, y_mark):
    if model_name == "XLinear":
        return model(x)
    x_dec = torch.zeros(
        x.shape[0], 48 + 96, x.shape[2], device=x.device, dtype=x.dtype
    )
    return model(x, x_mark, x_dec, y_mark)


def synchronize():
    torch.cuda.synchronize()


def timed_iterations(fn, warmup, repeats):
    for _ in range(warmup):
        fn()
    synchronize()
    samples = []
    for _ in range(repeats):
        synchronize()
        start = time.perf_counter()
        fn()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples


def stats(samples):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std(ddof=1)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "samples": int(values.size),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(REPOS))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-epoch", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(2021)
    np.random.seed(2021)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0")
    dataset = ECLWindows(
        SC_ROOT / "dataset/electricity/electricity.csv",
        seq_len=96, label_len=48, pred_len=96,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, drop_last=True, pin_memory=True,
    )
    first = next(iter(loader))
    x, target, x_mark, y_mark = [
        item.float().to(device, non_blocking=True) for item in first
    ]

    model = load_model(args.model).float().to(device)
    params_total = sum(p.numel() for p in model.parameters())
    params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    learning_rates = {
        "SC-FSD": 0.005, "XLinear": 0.0002, "TEFN": 0.0001,
        "iTransformer": 0.0005, "PatchTST": 0.0001, "DLinear": 0.0001,
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates[args.model])
    criterion = torch.nn.MSELoss()

    epoch_seconds = None
    epoch_iterations = 0
    if not args.skip_epoch:
        model.train()
        synchronize()
        epoch_start = time.perf_counter()
        for batch in loader:
            bx, by, bxm, bym = [
                item.float().to(device, non_blocking=True) for item in batch
            ]
            optimizer.zero_grad(set_to_none=True)
            output = forward_model(args.model, model, bx, bxm, bym)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()
            epoch_iterations += 1
        synchronize()
        epoch_seconds = time.perf_counter() - epoch_start

    model.train()
    def train_step():
        optimizer.zero_grad(set_to_none=True)
        output = forward_model(args.model, model, x, x_mark, y_mark)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    timed_iterations(train_step, args.warmup, 2)
    torch.cuda.reset_peak_memory_stats()
    train_samples = timed_iterations(train_step, args.warmup, args.repeats)
    train_peak = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    model.eval()
    @torch.no_grad()
    def infer_batch():
        forward_model(args.model, model, x, x_mark, y_mark)

    timed_iterations(infer_batch, args.warmup, 2)
    torch.cuda.reset_peak_memory_stats()
    inference_samples = timed_iterations(
        infer_batch, args.warmup, args.repeats
    )
    inference_peak = torch.cuda.max_memory_allocated() / (1024.0 ** 2)

    x1, xm1, ym1 = x[:1], x_mark[:1], y_mark[:1]
    @torch.no_grad()
    def infer_single():
        forward_model(args.model, model, x1, xm1, ym1)

    single_samples = timed_iterations(
        infer_single, args.warmup, args.repeats
    )
    gpu_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"], text=True
        ).strip().splitlines()[0]
    except Exception:
        driver = None

    result = {
        "model": args.model,
        "dataset": "ECL",
        "input_len": 96,
        "pred_len": 96,
        "batch_size": args.batch_size,
        "precision": "FP32",
        "seed": 2021,
        "warmup_iterations": args.warmup,
        "measurement_iterations": args.repeats,
        "parameters_total": params_total,
        "parameters_trainable": params_trainable,
        "one_epoch_seconds": epoch_seconds,
        "one_epoch_iterations": epoch_iterations,
        "one_epoch_mean_ms_per_iter": (
            epoch_seconds * 1000.0 / epoch_iterations
            if epoch_seconds is not None and epoch_iterations else None
        ),
        "stable_train": stats(train_samples),
        "inference_batch": stats(inference_samples),
        "single_sample_latency": stats(single_samples),
        "peak_train_memory_mib": train_peak,
        "peak_inference_memory_mib": inference_peak,
        "gpu": gpu_name,
        "gpu_compute_capability": list(capability),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "driver_version": driver,
        "repo_path": str(REPOS[args.model]),
        "config": vars(make_config(args.model)),
        "learning_rate": learning_rates[args.model],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
