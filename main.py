"""
main.py — GNSS Pseudorange Error Correction Experiment Runner

Usage examples:
    python main.py --model PrNet               --dataset v1
    python main.py --model transformer         --dataset v2
    python main.py --model Fcnn_Lstm           --dataset v1
    python main.py --model LSTM                --dataset v1 --window_size 5
    python main.py --model MAMBA               --dataset v2 --epochs 200
    python main.py --model 1DUnet              --dataset v1
    python main.py --model transformer_enhanced --dataset v2
    python main.py --model DiffGNSS            --dataset v1 --batch_size 256

Any config.json value can be overridden on the command line.

Data split:
    Train: sat_data_V1A1_[1-4]Jan_[highway|suburban]_multipath_processed.csv
    Test:  sat_data_V1A1_5Jan_[highway|suburban]_multipath_processed.csv
"""

import argparse
import importlib
import json
import os

import numpy as np
import pandas as pd
import torch

# ── Model imports ─────────────────────────────────────────────────────────────
from models.PrNet import MlpFeatureExtractor, PrNet
from models.transformer import TransformerGNSS
from models.Fcnn_Lstm import Fcnn_Lstm, Fcnn_LSTM_encoder
from models.LSTM import IndependentSatelliteLSTM
from models.transformer_enhanced import TransformerEnhancedLSTM
from models.DiffGNSS import DiffGNSS

# MAMBA requires the optional mamba_ssm package
try:
    from models.MAMBA import IndependentSatelliteMamba
    _MAMBA_AVAILABLE = True
except ImportError:
    _MAMBA_AVAILABLE = False

# 1DUnet filename starts with a digit — must use importlib
_unet_mod = importlib.import_module("models.1DUnet")
GNSS_UNet1D = _unet_mod.GNSS_UNet1D

# ── Pipeline imports ──────────────────────────────────────────────────────────
from utils import (
    create_dataloaders as create_dataloaders_epoch,
    evaluate_gnss_fcnn_lstm,
    evaluate_gnss_net,
    train_gnss_fcnn_lstm,
    train_gnss_net,
)
from temporal_utils import (
    create_dataloaders as create_dataloaders_windowed,
    evaluate_diff_gnss_windowed,
    evaluate_gnss_sequential,
    train_diff_gnss,
    train_gnss_sequential,
)

# ── Model groups ──────────────────────────────────────────────────────────────
NON_SEQUENTIAL = {"PrNet", "transformer", "Fcnn_Lstm"}
SEQUENTIAL      = {"LSTM", "MAMBA", "transformer_enhanced", "1DUnet", "DiffGNSS"}
ALL_MODELS      = NON_SEQUENTIAL | SEQUENTIAL


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


def build_model(model_name: str, cfg: dict) -> torch.nn.Module:
    """Instantiate the requested model from config parameters."""
    m = cfg[model_name]

    if model_name == "PrNet":
        extractor = MlpFeatureExtractor(
            input_size_debiasing=m["input_features"],
            num_hiddens_debiasing=m["num_hiddens"],
            num_debiasing_layers=m["num_layers"],
            dropout=m["dropout"],
        )
        return PrNet(extractor)

    elif model_name == "transformer":
        return TransformerGNSS(
            input_features=m["input_features"],
            d_model=m["d_model"],
            n_layers=m["n_layers"],
            nhead=m["nhead"],
            dropout=m["dropout"],
        )

    elif model_name == "Fcnn_Lstm":
        encoder = Fcnn_LSTM_encoder(
            input_size_debiasing=m["input_features"],
            num_hiddens_debiasing_fcnn=m["num_hiddens_fcnn"],
            num_hiddens_debiasing_lstm=m["num_hiddens_lstm"],
            dropout=m["dropout"],
        )
        return Fcnn_Lstm(encoder)

    elif model_name == "LSTM":
        return IndependentSatelliteLSTM(
            input_features=m["input_features"],
            hidden_dim=m["hidden_dim"],
            lstm_layers=m["lstm_layers"],
            dropout=m["dropout"],
        )

    elif model_name == "MAMBA":
        if not _MAMBA_AVAILABLE:
            raise RuntimeError(
                "MAMBA requires the 'mamba_ssm' package. "
                "Install it with: pip install mamba-ssm"
            )
        return IndependentSatelliteMamba(
            input_features=m["input_features"],
            hidden_dim=m["hidden_dim"],
            d_state=m["d_state"],
            d_conv=m["d_conv"],
            expand=m["expand"],
            num_layers=m["num_layers"],
            dropout=m["dropout"],
        )

    elif model_name == "1DUnet":
        return GNSS_UNet1D(in_channels=m["input_features"])

    elif model_name == "transformer_enhanced":
        return TransformerEnhancedLSTM(
            input_features=m["input_features"],
            hidden_dim=m["hidden_dim"],
            lstm_layers=m["lstm_layers"],
            mlp_layers=m["mlp_layers"],
            n_heads=m["n_heads"],
            dropout=m["dropout"],
        )

    elif model_name == "DiffGNSS":
        return DiffGNSS(
            input_dim=m["input_dim"],
            hidden_dim=m["hidden_dim"],
            num_timesteps=m["num_timesteps"],
            beta_schedule=m["beta_schedule"],
        )

    else:
        raise ValueError(f"Unknown model: '{model_name}'")


def parse_args():
    parser = argparse.ArgumentParser(
        description="GNSS PR Correction — train and evaluate a model on simGNSS data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--model", required=True, choices=sorted(ALL_MODELS),
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--dataset", required=True, choices=["v1", "v2"],
        help="Dataset version: v1 (simGNSS V1) or v2 (simGNSS V2).",
    )

    # General hyperparameter overrides (all optional; fall back to config.json)
    parser.add_argument("--epochs",      type=int,   default=None, help="Override num_epochs.")
    parser.add_argument("--lr",          type=float, default=None, help="Override learning rate.")
    parser.add_argument("--batch_size",  type=int,   default=None, help="Override batch size.")
    parser.add_argument("--label_index", type=int,   default=None,
                        help="Override label column index (14=Raw, 15=Smoothed).")
    parser.add_argument("--window_size", type=int,   default=None,
                        help="Temporal window size (sequential models only).")
    parser.add_argument("--test_files",  nargs="+",  default=None,
                        help="Override test file name patterns, e.g. '5Jan_highway 5Jan_suburban'.")

    # Misc
    parser.add_argument("--config",    type=str, default="config.json",
                        help="Path to config JSON.")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save the trained model weights (.pt).")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save per-file results CSVs (default: current directory).")

    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    gcfg   = cfg["general"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resolve hyperparameters (CLI > config.json) ────────────────────────
    num_epochs  = args.epochs     if args.epochs     is not None else gcfg["num_epochs"]
    lr          = args.lr         if args.lr         is not None else gcfg["lr"]
    batch_size  = args.batch_size if args.batch_size is not None else gcfg["batch_size"]
    label_index = args.label_index if args.label_index is not None else gcfg["label_index"]
    test_files  = args.test_files  if args.test_files  is not None else gcfg["test_files"]

    model_cfg   = cfg[args.model]
    window_size = args.window_size if args.window_size is not None \
                  else model_cfg.get("window_size", 5)

    data_dir = os.path.join("data", f"data_{args.dataset}")

    print(f"\n{'=' * 60}")
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    simGNSS {args.dataset}  ({data_dir})")
    print(f"  Device:     {device}")
    print(f"  Epochs:     {num_epochs}  |  LR: {lr}  |  Batch: {batch_size}")
    if args.model in SEQUENTIAL:
        print(f"  Window:     {window_size}")
    print(f"  Label idx:  {label_index}")
    print(f"  Test files: {test_files}")
    print(f"{'=' * 60}\n")

    # ── Build model ────────────────────────────────────────────────────────
    model = build_model(args.model, cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}\n")

    # ── Create dataloaders ─────────────────────────────────────────────────
    if args.model in NON_SEQUENTIAL:
        train_loader, test_loaders, norm_stats = create_dataloaders_epoch(
            data_directory=data_dir,
            input_size=gcfg["input_size"],
            PRN_size=gcfg["PRN_size"],
            batch_size=batch_size,
            label_index=label_index,
            device=device,
            test_files=test_files,
        )
    else:
        train_loader, test_loaders, norm_stats = create_dataloaders_windowed(
            data_directory=data_dir,
            input_size=gcfg["input_size"],
            PRN_size=gcfg["PRN_size"],
            batch_size=batch_size,
            label_index=label_index,
            window_size=window_size,
            device=device,
            test_files=test_files,
        )

    # test_loaders is a dict: {pattern_name: DataLoader}
    # Use the first test loader as validation during DiffGNSS training.
    first_test_loader = next(iter(test_loaders.values()))

    # ── Train ──────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"TRAINING  {args.model}")
    print(f"{'=' * 60}")

    if args.model in {"PrNet", "transformer"}:
        train_gnss_net(model, train_loader, lr, num_epochs, 1, device)

    elif args.model == "Fcnn_Lstm":
        train_gnss_fcnn_lstm(model, train_loader, lr, num_epochs, 1, device)

    elif args.model == "DiffGNSS":
        _, model = train_diff_gnss(
            model=model,
            train_loader=train_loader,
            val_loader=first_test_loader,
            device=device,
            num_epochs=num_epochs,
            lr=lr,
            lambda_pri=model_cfg["lambda_pri"],
            lambda_res=model_cfg["lambda_res"],
            lambda_prr=model_cfg["lambda_prr"],
            gradient_clip=model_cfg["gradient_clip"],
            save_path=args.save_path,
        )

    else:
        # LSTM, MAMBA, transformer_enhanced, 1DUnet
        train_gnss_sequential(
            net=model,
            train_loader=train_loader,
            test_loader=first_test_loader,
            lr=lr,
            total_epochs=num_epochs,
            device=device,
        )

    # ── Evaluate on each test file, save one CSV per file ─────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {}  # pattern -> (output_tensor, losses)

    for pattern, test_loader in test_loaders.items():
        print(f"\n{'=' * 60}")
        print(f"EVALUATING  {args.model}  on  {pattern}")
        print(f"{'=' * 60}")

        if args.model in {"PrNet", "transformer"}:
            output_tensor, losses = evaluate_gnss_net(model, test_loader, device, label_index)
        elif args.model == "Fcnn_Lstm":
            output_tensor, losses = evaluate_gnss_fcnn_lstm(model, test_loader, device, label_index)
        elif args.model == "DiffGNSS":
            output_tensor, losses = evaluate_diff_gnss_windowed(model, test_loader, device)
        else:
            output_tensor, losses = evaluate_gnss_sequential(model, test_loader, device)

        all_results[pattern] = (output_tensor, losses)

        # Per-file metrics
        mae  = float(np.mean([l.item() for l in losses]))
        errors = output_tensor[:, 2] - output_tensor[:, 3]
        rmse = float(torch.sqrt((errors ** 2).mean()).item())

        print(f"\n{'=' * 60}")
        print(f"  RESULTS  —  {args.model}  ({pattern})  [simGNSS {args.dataset}]")
        print(f"  MAE:  {mae:.4f} m")
        print(f"  RMSE: {rmse:.4f} m")
        print(f"{'=' * 60}")

        # Save CSV
        safe_pattern = pattern.replace("/", "_").replace("\\", "_")
        csv_path = os.path.join(
            args.output_dir,
            f"results_{args.model}_{args.dataset}_{safe_pattern}.csv",
        )
        df = pd.DataFrame({
            "time_epoch":          output_tensor[:, 0].numpy(),
            "sat_id":              output_tensor[:, 1].numpy(),
            "actual_target_value": output_tensor[:, 2].numpy(),
            "model_output":        output_tensor[:, 3].numpy(),
        })
        df.index.name = "index"
        df.to_csv(csv_path, header=False)
        print(f"Results saved to: {csv_path}  ({len(df):,} rows)")

    # ── Save model weights ─────────────────────────────────────────────────
    if args.save_path and args.model != "DiffGNSS":
        # DiffGNSS saves checkpoints internally during training
        torch.save(model.state_dict(), args.save_path)
        print(f"Model weights saved to: {args.save_path}")

    return all_results


if __name__ == "__main__":
    main()
