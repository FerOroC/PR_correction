"""
temporal_utils.py — Data pipeline and training utilities for sequential (windowed) GNSS models.

Supported models:
    - IndependentSatelliteLSTM       (LSTM)
    - IndependentSatelliteMamba      (MAMBA)
    - TransformerEnhancedLSTM        (transformer_enhanced)
    - GNSS_UNet1D                    (1DUnet)
    - DiffGNSS                       (DiffGNSS)
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Optional, Tuple

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_data_from_filelist(
        files: list,
        input_size: int,
        PRN_size: int,
) -> torch.Tensor:
    """
    Load a list of CSV files and return a stacked tensor of epochs.

    Returns:
        Shape (num_epochs, PRN_size, input_size)
    """
    print(f"\n{'=' * 60}")
    print(f"Loading {len(files)} file(s)")
    print(f"{'=' * 60}")

    all_epochs = []

    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)

        # Convert to numpy (no delta computation)
        base_data = df.iloc[:, :input_size].values

        # To tensor
        raw_tensor = torch.tensor(base_data, dtype=torch.float64)

        # Organize into epochs (PRN_size, input_size) per timestep
        epochs = organize_into_epochs(raw_tensor, input_size, PRN_size)
        all_epochs.extend(epochs)

    print(f"\nTotal epochs loaded: {len(all_epochs)}")
    if all_epochs:
        return torch.stack(all_epochs)
    else:
        return torch.empty(0, PRN_size, input_size, dtype=torch.float64)


def organize_into_epochs(
        data_tensor: torch.Tensor,
        feature_size: int,
        PRN_size: int
) -> list:
    """
    Convert flat tensor to list of (PRN_size, feature_size) tensors per epoch.
    """
    epochs = []

    if data_tensor.shape[0] == 0:
        return epochs

    current_epoch = torch.zeros(PRN_size, feature_size, dtype=torch.float64)
    current_time_ms = data_tensor[0, 0]

    for row in data_tensor:
        time_ms = row[0]
        sat_id = row[1]

        if time_ms != current_time_ms:
            epochs.append(current_epoch.clone())
            current_epoch = torch.zeros(PRN_size, feature_size, dtype=torch.float64)
            current_time_ms = time_ms

        if sat_id != 0:
            prn_idx = int(sat_id.item()) - 1
            if 0 <= prn_idx < PRN_size:
                current_epoch[prn_idx, :] = row[:feature_size]

    # Append last epoch
    epochs.append(current_epoch)

    return epochs


# ============================================================================
# STEP 2: CREATE SLIDING WINDOWS
# ============================================================================

def create_windows(
        epoch_tensor: torch.Tensor,
        window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create sliding windows from epoch tensor with full-visibility masking.

    Args:
        epoch_tensor: Shape (num_epochs, PRN_size, input_size)
        window_size: Number of consecutive epochs per window

    Returns:
        windows: Shape (num_windows, window_size, PRN_size, input_size)
        window_masks: Shape (num_windows, PRN_size) - True only if satellite
                      visible in ALL timesteps of the window
    """
    num_epochs, PRN_size, input_size = epoch_tensor.shape
    num_windows = num_epochs - window_size + 1

    if num_windows <= 0:
        raise ValueError(f"Not enough epochs ({num_epochs}) for window size {window_size}")

    print(f"\n{'=' * 60}")
    print("CREATING SLIDING WINDOWS")
    print(f"{'=' * 60}")
    print(f"Epochs: {num_epochs}, Window size: {window_size}, Windows: {num_windows}")

    # Create windows using unfold: efficient sliding window operation
    windows = epoch_tensor.unfold(0, window_size, 1)  # (num_windows, PRN_size, input_size, window_size)
    windows = windows.permute(0, 3, 1, 2)             # (num_windows, window_size, PRN_size, input_size)

    # Create per-epoch masks (satellite visible if sat_id != 0)
    SAT_ID_COL = 1
    sat_ids = epoch_tensor[:, :, SAT_ID_COL]  # (num_epochs, PRN_size)
    per_epoch_mask = (sat_ids != 0)            # (num_epochs, PRN_size)

    # Create window masks - satellite valid ONLY if visible in ALL timesteps
    window_masks_unfolded = per_epoch_mask.unfold(0, window_size, 1)  # (num_windows, PRN_size, window_size)
    window_masks = window_masks_unfolded.all(dim=-1)                   # (num_windows, PRN_size)

    # Statistics on data retention
    total_possible = num_windows * PRN_size
    total_valid = window_masks.sum().item()
    avg_sats_per_window = window_masks.sum(dim=1).float().mean().item()

    final_timestep_visible = per_epoch_mask[window_size - 1:]
    possible_from_visible = final_timestep_visible.sum().item()
    retention_vs_visible = total_valid / possible_from_visible * 100 if possible_from_visible > 0 else 0
    avg_visible_final = final_timestep_visible.sum(dim=1).float().mean().item()

    print(f"Valid satellite-window pairs: {int(total_valid)}/{total_possible} "
          f"({100 * total_valid / total_possible:.1f}%) [vs all PRN slots]")
    print(f"Retention vs final-timestep visible: {int(total_valid)}/{int(possible_from_visible)} "
          f"({retention_vs_visible:.1f}%)")
    print(f"Average satellites visible at final timestep: {avg_visible_final:.1f}")
    print(f"Average satellites with full history: {avg_sats_per_window:.1f}")

    return windows, window_masks


# ============================================================================
# STEP 3: FEATURE PREPROCESSING
# ============================================================================

def preprocess_features_batch(
        data_tensor: torch.Tensor,
        window_mask: torch.Tensor,
        device: torch.device,
        label_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess windowed data into model-ready features.

    Args:
        data_tensor: Shape (num_windows, window_size, PRN_size, input_size)
        window_mask: Shape (num_windows, PRN_size) - full-window validity
        device: torch device
        label_index: Column index for labels in raw data

    Returns:
        features: (num_windows, window_size, PRN_size, num_features=22)
        mask: (num_windows, PRN_size) - full-window validity mask
        labels: (num_windows, window_size, PRN_size, 1)
        clock_corrections: (num_windows, PRN_size, 1)
        time_ms: (num_windows, PRN_size, 1)
        sat_id: (num_windows, PRN_size, 1)
    """
    print(f"\n{'=' * 60}")
    print("PREPROCESSING FEATURES (WINDOWED)")
    print(f"{'=' * 60}")
    print(f"Input shape: {data_tensor.shape}")

    N, W, S, F = data_tensor.shape  # Windows, window_size, Satellites, Features

    # Column indices
    IDX_TIME = 0
    IDX_SAT_ID = 1
    IDX_SAT_XYZ = slice(2, 5)
    IDX_ELEVATION = 5
    IDX_AZIMUTH = 7
    IDX_NED = slice(8, 11)
    IDX_PR_RATE = 11
    IDX_P_RANGE = 13
    IDX_LABEL = label_index
    IDX_ATMOSPHERIC = 18
    IDX_POWER = 19
    IDX_HEADING = slice(21, 24)
    IDX_WLS_LON_DEG = 27
    IDX_WLS_LON_MIN = 28
    IDX_WLS_LON_SEC = 29
    IDX_WLS_LAT_DEG = 30
    IDX_WLS_LAT_MIN = 31
    IDX_WLS_LAT_SEC = 32
    IDX_CLOCK_BIAS = 33
    IDX_WLS_XYZ = slice(34, 37)
    IDX_H_ITEM_LAST = 17

    # Expand mask for 4D broadcasting: (N, S) -> (N, W, S, 1)
    mask_4d = window_mask.unsqueeze(1).unsqueeze(-1).expand(N, W, S, 1).float()

    # ==================== EXTRACT RAW FEATURES ====================
    elev = data_tensor[:, :, :, IDX_ELEVATION]
    azim = data_tensor[:, :, :, IDX_AZIMUTH]
    pr_rate = data_tensor[:, :, :, IDX_PR_RATE]
    power = data_tensor[:, :, :, IDX_POWER].unsqueeze(-1)

    wls_lon_deg = data_tensor[:, :, :, IDX_WLS_LON_DEG]
    wls_lon_min = data_tensor[:, :, :, IDX_WLS_LON_MIN]
    wls_lon_sec = data_tensor[:, :, :, IDX_WLS_LON_SEC]
    wls_lat_deg = data_tensor[:, :, :, IDX_WLS_LAT_DEG]
    wls_lat_min = data_tensor[:, :, :, IDX_WLS_LAT_MIN]
    wls_lat_sec = data_tensor[:, :, :, IDX_WLS_LAT_SEC]

    ugv = data_tensor[:, :, :, IDX_NED]
    heading = data_tensor[:, :, :, IDX_HEADING]

    Pr = data_tensor[:, :, :, IDX_P_RANGE:IDX_P_RANGE + 1]
    AtmDelays = data_tensor[:, :, :, IDX_ATMOSPHERIC:IDX_ATMOSPHERIC + 1]
    svXyz = data_tensor[:, :, :, IDX_SAT_XYZ]
    wlsXyz = data_tensor[:, :, :, IDX_WLS_XYZ]
    wlsDtu = data_tensor[:, :, :, IDX_CLOCK_BIAS:IDX_CLOCK_BIAS + 1]

    sat_prn = data_tensor[:, :, :, IDX_SAT_ID:IDX_SAT_ID + 1]

    # ==================== TRIGONOMETRIC FEATURES ====================
    sinE = torch.sin(elev).unsqueeze(-1)
    cosE = torch.cos(elev).unsqueeze(-1)
    sinA = torch.sin(azim).unsqueeze(-1)
    cosA = torch.cos(azim).unsqueeze(-1)

    # ==================== PR RESIDUAL ====================
    geom_range = torch.linalg.norm(svXyz - wlsXyz, dim=-1, keepdim=True)
    pr_resi = Pr - AtmDelays - geom_range - wlsDtu

    # ==================== RSS (per timestep in window) ====================
    mask_3d = window_mask.unsqueeze(1).expand(N, W, S).float()
    pr_resi_sq = (pr_resi.squeeze(-1) ** 2) * mask_3d
    rss_per_timestep = torch.sqrt(torch.clamp(pr_resi_sq.sum(dim=2), min=0.0))  # (N, W)
    rss_feat = rss_per_timestep.unsqueeze(2).unsqueeze(-1).expand(N, W, S, 1)

    # ==================== VISIBLE SATELLITES COUNT (per timestep) ====================
    visible_count = window_mask.sum(dim=1, keepdim=True).float()  # (N, 1)
    visible_sats_feat = visible_count.unsqueeze(1).unsqueeze(-1).expand(N, W, S, 1)

    # ==================== NORMALIZE COORDINATES ====================
    wls_lon_deg_n = (wls_lon_deg / 180.0).unsqueeze(-1)
    wls_lon_min_n = (wls_lon_min / 60.0).unsqueeze(-1)
    lon_sec_normalized = (wls_lon_sec / 60.0).unsqueeze(-1)
    wls_lon_sec_n = torch.round(lon_sec_normalized * 1000) / 1000
    wls_lat_deg_n = (wls_lat_deg / 90.0).unsqueeze(-1)
    wls_lat_min_n = (wls_lat_min / 60.0).unsqueeze(-1)
    lat_sec_normalized = (wls_lat_sec / 60.0).unsqueeze(-1)
    wls_lat_sec_n = torch.round(lat_sec_normalized * 1000) / 1000

    # ==================== CONCATENATE ALL FEATURES ====================
    features = torch.cat([
        sinE, cosE, sinA, cosA,              # 0-3:  Trig features
        power,                               # 4:    Power/CN0
        pr_rate.unsqueeze(-1),               # 5:    PR rate
        wls_lon_deg_n, wls_lon_min_n, wls_lon_sec_n,  # 6-8:  Longitude
        wls_lat_deg_n, wls_lat_min_n, wls_lat_sec_n,  # 9-11: Latitude
        ugv,                                 # 12-14: Unit geometry vector (NED)
        heading,                             # 15-17: Heading (NED)
        pr_resi,                             # 18:   PR residual
        rss_feat,                            # 19:   RSS
        visible_sats_feat,                   # 20:   Visible satellites count
        sat_prn,                             # 21:   Satellite PRN
    ], dim=-1)  # Shape: (N, W, S, 22)

    # ==================== EXTRACT LABELS (MANY-TO-MANY) ====================
    labels = data_tensor[:, :, :, IDX_LABEL:IDX_LABEL + 1]  # (N, W, S, 1)

    # Metadata from the final timestep
    final_timestep = data_tensor[:, -1, :, :]
    clock_corrections = final_timestep[:, :, IDX_H_ITEM_LAST:IDX_H_ITEM_LAST + 1]
    time_ms = final_timestep[:, :, IDX_TIME:IDX_TIME + 1]
    sat_id_data = final_timestep[:, :, IDX_SAT_ID:IDX_SAT_ID + 1]

    # ==================== APPLY MASK AND CLEAN ====================
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) * mask_4d
    labels = labels * window_mask.unsqueeze(1).unsqueeze(-1).float()

    print(f"Output features shape: {features.shape}")
    print(f"Mask shape: {window_mask.shape}")
    print(f"Labels shape: {labels.shape}")

    return features, window_mask, labels, clock_corrections, time_ms, sat_id_data


# ============================================================================
# STEP 4: NORMALIZATION (GLOBAL ACROSS DATASET)
# ============================================================================

def compute_normalization_stats(features: torch.Tensor, mask: torch.Tensor) -> Dict:
    """
    Compute mean/std for z-score normalization across entire dataset.
    Only computes over valid (masked) satellite entries.

    Args:
        features: Shape (N, W, S, num_features)
        mask: Shape (N, S) - full-window validity
    """
    print(f"\n{'=' * 60}")
    print("COMPUTING NORMALIZATION STATISTICS (GLOBAL)")
    print(f"{'=' * 60}")

    stats = {}
    N, W, S, F = features.shape

    # Expand mask: (N, S) -> (N, W, S) for indexing
    mask_expanded = mask.unsqueeze(1).expand(N, W, S)

    # Z-score normalization indices
    z_score_features = {
        'power': 4,
        'pr_rate': 5,
        'pr_resi': 18,
        'rss': 19,
    }

    for name, idx in z_score_features.items():
        values = features[:, :, :, idx][mask_expanded]
        mean_val = values.mean().item()
        std_val = values.std().item()
        stats[name] = {'mean': mean_val, 'std': std_val}
        print(f"  {name:15s}: mean={mean_val:12.4f}, std={std_val:12.4f}")

    # Min-max normalization indices
    minmax_features = {
        'visible_sats': 20
    }

    for name, idx in minmax_features.items():
        values = features[:, :, :, idx][mask_expanded]
        min_val = values.min().item()
        max_val = values.max().item()
        stats[name] = {'min': min_val, 'max': max_val}
        print(f"  {name:15s}: min={min_val:12.4f}, max={max_val:12.4f}")

    return stats


def apply_normalization(
        features: torch.Tensor,
        mask: torch.Tensor,
        stats: Dict
) -> torch.Tensor:
    """
    Apply normalization using precomputed statistics.

    Args:
        features: Shape (N, W, S, num_features)
        mask: Shape (N, S)
    """
    print(f"\n{'=' * 60}")
    print("APPLYING NORMALIZATION")
    print(f"{'=' * 60}")

    features_norm = features.clone()
    N, W, S, F = features.shape
    mask_4d = mask.unsqueeze(1).unsqueeze(-1).expand(N, W, S, 1).float()

    # Z-score normalization
    z_score_features = {
        'power': 4,
        'pr_rate': 5,
        'pr_resi': 18,
        'rss': 19,
    }

    for name, idx in z_score_features.items():
        mean = stats[name]['mean']
        std = stats[name]['std']
        if std > 1e-8:
            features_norm[:, :, :, idx] = (features[:, :, :, idx] - mean) / std
        features_norm[:, :, :, idx:idx + 1] = features_norm[:, :, :, idx:idx + 1] * mask_4d
        print(f"  Normalized {name} (z-score)")

    # Min-max normalization
    minmax_features = {
        'visible_sats': 20
    }

    for name, idx in minmax_features.items():
        min_val = stats[name]['min']
        max_val = stats[name]['max']
        if max_val - min_val > 1e-8:
            features_norm[:, :, :, idx] = (features[:, :, :, idx] - min_val) / (max_val - min_val)
        features_norm[:, :, :, idx:idx + 1] = features_norm[:, :, :, idx:idx + 1] * mask_4d
        print(f"  Normalized {name} (min-max)")

    return features_norm


# ============================================================================
# STEP 5: CREATE DATALOADERS
# ============================================================================

def create_dataloaders(
        data_directory: str,
        input_size: int = 37,
        PRN_size: int = 32,
        batch_size: int = 64,
        label_index: int = 15,
        window_size: int = 5,
        device: torch.device = None,
        test_files: list = None,
):
    """
    Create dataloaders with temporal windowing and full-visibility masking.

    Args:
        data_directory: Path to CSV files
        input_size: Number of columns in raw CSV
        PRN_size: Maximum number of satellites (typically 32)
        batch_size: Batch size for dataloaders
        label_index: Column index for target labels
        window_size: Number of consecutive epochs per sample
        device: torch device
        test_files: List of filename patterns for test set

    Returns:
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        norm_stats: Normalization statistics (for inference)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("CREATING DATALOADERS (WINDOWED)")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Window size: {window_size}")

    # Gather all CSV files
    all_files = sorted(
        os.path.join(data_directory, f)
        for f in os.listdir(data_directory)
        if f.endswith(".csv")
    )
    assert len(all_files) >= 2, "Need at least 2 files"

    if test_files is None or len(test_files) == 0:
        raise ValueError("You must specify one or more test files.")

    # Match test files
    test_file_paths = []
    for name in test_files:
        matches = [f for f in all_files if name in os.path.basename(f)]
        assert len(matches) == 1, f"Could not uniquely match test file pattern '{name}'"
        test_file_paths.append(matches[0])

    train_files = [f for f in all_files if f not in test_file_paths]

    print("\nTRAIN FILES:")
    for f in train_files:
        print("  -", os.path.basename(f))
    print("\nTEST FILES:")
    for f in test_file_paths:
        print("  -", os.path.basename(f))
    print("=" * 60)

    # ==================== LOAD RAW EPOCHS ====================
    train_epochs = load_data_from_filelist(train_files, input_size, PRN_size)

    # ==================== CREATE WINDOWS ====================
    train_windows, train_window_mask = create_windows(train_epochs, window_size)

    # ==================== PREPROCESS FEATURES ====================
    train_features, train_mask, train_labels, train_clock, train_time, train_satid = \
        preprocess_features_batch(train_windows, train_window_mask, device, label_index)

    norm_stats = compute_normalization_stats(train_features, train_mask)
    train_features = apply_normalization(train_features, train_mask, norm_stats)

    # ==================== BUILD TRAIN DATASET ====================
    train_dataset = TensorDataset(
        train_features, train_mask, train_labels, train_clock, train_time, train_satid
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # ==================== BUILD ONE TEST LOADER PER FILE ====================
    test_loaders = {}
    for name, path in zip(test_files, test_file_paths):
        epochs = load_data_from_filelist([path], input_size, PRN_size)
        windows, window_mask = create_windows(epochs, window_size)
        feat, mask_, lbl, clk, t, sid = preprocess_features_batch(
            windows, window_mask, device, label_index
        )

        print("\n" + "=" * 60)
        print(f"NORMALIZING TEST DATA ({name}) USING TRAIN STATS")
        print("=" * 60)
        feat = apply_normalization(feat, mask_, norm_stats)

        ds = TensorDataset(feat, mask_, lbl, clk, t, sid)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("DATALOADER SUMMARY")
    print("=" * 60)
    print(f"TRAIN WINDOWS: {len(train_dataset)}")
    for name, loader in test_loaders.items():
        print(f"TEST  WINDOWS ({name}): {len(loader.dataset)}")
    print(f"Feature shape per sample: (W={window_size}, S={PRN_size}, F={train_features.shape[-1]})")

    return train_loader, test_loaders, norm_stats


# ============================================================================
# FEATURE SELECTION
# ============================================================================

def select_features_windowed(features: torch.Tensor) -> torch.Tensor:
    """
    Select 18 feature channels from the 22-feature windowed tensor.

    Input shape:  (B, W, S, 22)
    Output shape: (B, W, S, 18)

    Feature index map (input):
        0:    sinE        1:    cosE        2:    sinA        3:    cosA
        4:    power/CN0   5:    pr_rate
        6-8:  longitude (deg, min, sec)     9-11: latitude (deg, min, sec)
        12-14: ugv (NED)                   15-17: heading (NED)
        18:   pr_resi     19:   rss         20:   visible_sats  21: sat_prn

    Selected (18): sinE, sinA, power→sat_prn(skip cosE/cosA), pr_resi, rss, sat_prn
    """
    selected = torch.cat([
        features[:, :, :, 0:2],    # sinE, cosE
        features[:, :, :, 4:18],   # power, pr_rate, lon[3], lat[3], ugv[3], heading[3]
        features[:, :, :, 18:20],  # pr_resi, rss
        features[:, :, :, 21:22],  # sat_prn
    ], dim=-1)

    return selected  # (B, W, S, 19)


# ============================================================================
# TRAINING: SEQUENTIAL MODELS
# (LSTM, MAMBA, transformer_enhanced, 1DUnet)
# ============================================================================

def train_gnss_sequential(
        net: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        lr: float,
        total_epochs: int,
        device: torch.device,
        print_every: int = 10,
        use_scheduler: bool = True,
) -> Dict:
    """
    Train any sequential (windowed) GNSS model.

    Handles both output shapes:
        - (B, W, S, 1)  e.g. GNSS_UNet1D  → loss on last timestep
        - (B, S, 1)     e.g. LSTM / MAMBA / TransformerEnhancedLSTM → direct

    Labels are always evaluated at the final timestep of each window.

    Args:
        net: Any model with forward(features, mask) signature
        train_loader: Windowed training DataLoader
        test_loader: Windowed test DataLoader (accepted for API compatibility)
        lr: Initial learning rate
        total_epochs: Number of training epochs
        device: torch device
        print_every: Print progress every N epochs
        use_scheduler: Apply StepLR(step_size=100, gamma=0.5)

    Returns:
        history: Dict with 'train_loss' list
    """
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.L1Loss(reduction='none')
    history = {'train_loss': []}

    for epoch in range(total_epochs):
        net.train()
        epoch_losses = []

        for features, mask, labels, clock, time_ms, sat_id in train_loader:
            features = select_features_windowed(features).to(device)  # (B, W, S, 18)
            mask = mask.to(device)                                     # (B, S)
            labels = labels.to(device)                                 # (B, W, S, 1)

            optimizer.zero_grad()

            preds = net(features, mask)

            # Normalize output shape to (B, S, 1)
            if preds.dim() == 4:
                preds_last = preds[:, -1, :, :]   # UNet: take last timestep
            else:
                preds_last = preds                 # LSTM/MAMBA/TransformerEnhanced

            labels_last = labels[:, -1, :, :]     # (B, S, 1)

            loss_per_sat = criterion(preds_last, labels_last)  # (B, S, 1)
            J = (loss_per_sat * mask.unsqueeze(-1).float()).sum() / mask.sum().float()

            J.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(J.detach().item())

        mean_train = float(np.mean(epoch_losses))
        history['train_loss'].append(mean_train)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch + 1:4d}/{total_epochs} | Train Loss: {mean_train:.4f}")

        if use_scheduler:
            scheduler.step()

    return history


# ============================================================================
# EVALUATION: SEQUENTIAL MODELS
# ============================================================================

def evaluate_gnss_sequential(
        net: nn.Module,
        data_iter: DataLoader,
        device: torch.device,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate any sequential GNSS model.

    Handles both output shapes:
        - (B, W, S, 1)  → evaluated at last timestep
        - (B, S, 1)     → used directly

    Returns:
        output_tensor: (num_valid_sats_total, 4)
                       columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]
        losses: List of per-batch L1 losses
    """
    net.to(device)
    net.eval()

    loss_fn = nn.L1Loss(reduction='none')

    time_step = 0
    time_sum = []
    output_seq = []
    losses = []

    print(f"\n{'=' * 60}")
    print("EVALUATING SEQUENTIAL MODEL")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for features, mask, labels, clock, time_ms, sat_id in data_iter:
            time_step += 1

            features = select_features_windowed(features).to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            time_ms = time_ms.to(device)
            sat_id = sat_id.to(device)

            start_time = time.time()
            preds = net(features, mask)
            elapsed = time.time() - start_time
            time_sum.append(elapsed)

            # Normalize output shape to (B, S, 1)
            if preds.dim() == 4:
                preds_last = preds[:, -1, :, :]
            else:
                preds_last = preds

            labels_last = labels[:, -1, :, :]  # (B, S, 1)

            # Loss
            loss_per_sat = loss_fn(preds_last, labels_last)
            J = (loss_per_sat * mask.unsqueeze(-1).float()).sum() / mask.sum().float()
            losses.append(J.cpu())

            # Collect output
            for i in range(features.shape[0]):
                valid_mask = mask[i]  # (S,)

                epoch_time  = time_ms[i, valid_mask, :]      # (num_valid, 1)
                epoch_satid = sat_id[i, valid_mask, :]       # (num_valid, 1)
                epoch_real  = labels_last[i, valid_mask, :]  # (num_valid, 1)
                epoch_pred  = preds_last[i, valid_mask, :]   # (num_valid, 1)

                output_per_epoch = torch.cat([
                    epoch_time, epoch_satid, epoch_real, epoch_pred
                ], dim=1)  # (num_valid, 4)

                output_seq.append(output_per_epoch.cpu())

    elapsed_per_sample = sum(time_sum) / len(time_sum)
    print(f'\nInference time per sample: {elapsed_per_sample:.6f}s')
    print(f'Average loss: {np.mean([l.item() for l in losses]):.6f}')

    output_tensor = torch.cat(output_seq, dim=0)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]")

    return output_tensor, losses


# ============================================================================
# DIFF-GNSS LOSS
# ============================================================================

class DiffGNSSLoss(nn.Module):
    """
    Multi-task loss for Diff-GNSS.

    L = λ_pri * L_pri + λ_res * L_res + λ_prr * L_prr
    """

    def __init__(
            self,
            lambda_pri: float = 0.5,
            lambda_res: float = 0.5,
            lambda_prr: float = 1.0,
    ):
        super().__init__()
        self.lambda_pri = lambda_pri
        self.lambda_res = lambda_res
        self.lambda_prr = lambda_prr
        self.mse = nn.MSELoss(reduction='none')

    def forward(
            self,
            outputs: Dict[str, torch.Tensor],
            labels_gt: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            outputs: Dict from DiffGNSS forward pass
            labels_gt: (B, N, 1) ground truth pseudorange errors
            mask: (B, N) valid satellite mask

        Returns:
            total_loss: Scalar tensor
            loss_dict: Individual loss values for logging
        """
        mask_float = mask.unsqueeze(-1).float()
        num_valid = mask_float.sum() + 1e-8

        # L_pri: Coarse prediction loss
        L_pri = self.mse(outputs['delta_rho_init'], labels_gt)
        L_pri = (L_pri * mask_float).sum() / num_valid

        # L_res: Residual prediction loss
        L_res = self.mse(outputs['epsilon_hat_0'], outputs['epsilon_gt'])
        L_res = (L_res * mask_float).sum() / num_valid

        # L_prr: Final prediction loss
        L_prr = self.mse(outputs['delta_rho_fine'], labels_gt)
        L_prr = (L_prr * mask_float).sum() / num_valid

        total_loss = (
                self.lambda_pri * L_pri
                + self.lambda_res * L_res
                + self.lambda_prr * L_prr
        )

        loss_dict = {
            'L_pri': L_pri.item(),
            'L_res': L_res.item(),
            'L_prr': L_prr.item(),
            'total': total_loss.item(),
        }

        return total_loss, loss_dict


# ============================================================================
# TRAINING: DIFF-GNSS
# ============================================================================

def train_diff_gnss(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        num_epochs: int = 200,
        lr: float = 1e-4,
        lambda_pri: float = 0.5,
        lambda_res: float = 0.5,
        lambda_prr: float = 1.0,
        gradient_clip: float = 1.0,
        print_every: int = 10,
        save_path: Optional[str] = None,
) -> Tuple[Dict[str, List[float]], nn.Module]:
    """
    Train Diff-GNSS model with joint optimization.

    Args:
        model: DiffGNSS model
        train_loader: Training DataLoader (windowed)
        val_loader: Validation DataLoader (windowed)
        device: Device to train on
        num_epochs: Number of training epochs
        lr: Initial learning rate
        lambda_pri/res/prr: Loss component weights
        gradient_clip: Max gradient norm for clipping
        print_every: Print progress every N epochs
        save_path: Path to save best model checkpoint

    Returns:
        history: Dictionary of training history
        model: Trained model
    """
    model = model.to(device)

    criterion = DiffGNSSLoss(lambda_pri, lambda_res, lambda_prr)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    history = {
        'train_loss': [],
        'train_L_pri': [],
        'train_L_res': [],
        'train_L_prr': [],
        'val_loss': [],
        'val_mae': [],
        'val_rmse': [],
    }

    best_val_loss = float('inf')

    print("=" * 70)
    print(f"Training Diff-GNSS")
    print(f"  Device: {device}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Loss weights: λ_pri={lambda_pri}, λ_res={lambda_res}, λ_prr={lambda_prr}")
    print("=" * 70)

    for epoch in range(num_epochs):
        epoch_start = time.time()

        # ==================== TRAINING ====================
        model.train()
        train_losses = {k: 0.0 for k in ['total', 'L_pri', 'L_res', 'L_prr']}
        num_batches = 0

        for batch in train_loader:
            features = batch[0]
            features = select_features_windowed(features).to(device)
            mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Windowed labels: (B, W, S, 1) → take final timestep (B, S, 1)
            if labels.dim() == 4:
                labels = labels[:, -1, :, :]
            elif labels.dim() == 2:
                labels = labels.unsqueeze(-1)

            optimizer.zero_grad()
            outputs = model(features, mask, labels)

            loss, loss_dict = criterion(outputs, labels, mask)

            loss.backward()
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

            for k, v in loss_dict.items():
                train_losses[k] += v
            num_batches += 1

        for k in train_losses:
            train_losses[k] /= num_batches

        history['train_loss'].append(train_losses['total'])
        history['train_L_pri'].append(train_losses['L_pri'])
        history['train_L_res'].append(train_losses['L_res'])
        history['train_L_prr'].append(train_losses['L_prr'])

        # ==================== VALIDATION ====================
        val_loss, val_mae, val_rmse = validate_diff_gnss(model, val_loader, criterion, device)

        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_rmse'].append(val_rmse)

        scheduler.step()

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'val_loss': val_loss,
                }, save_path)

        epoch_time = time.time() - epoch_start
        if (epoch + 1) % print_every == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{num_epochs} ({epoch_time:.1f}s) | "
                  f"Train: {train_losses['total']:.4f} "
                  f"(pri={train_losses['L_pri']:.4f}, res={train_losses['L_res']:.4f}, "
                  f"prr={train_losses['L_prr']:.4f}) | "
                  f"Val: {val_loss:.4f} | MAE: {val_mae:.3f}m | RMSE: {val_rmse:.3f}m")

    print("=" * 70)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    print("=" * 70)

    return history, model


# ============================================================================
# VALIDATION: DIFF-GNSS
# ============================================================================

@torch.no_grad()
def validate_diff_gnss(
        model: nn.Module,
        val_loader: DataLoader,
        criterion: DiffGNSSLoss,
        device: torch.device,
        use_inference: bool = False,
        num_inference_steps: int = 50,
) -> Tuple[float, float, float]:
    """
    Validate Diff-GNSS model.

    Returns:
        avg_loss: Average validation loss
        mae: Mean absolute error (m)
        rmse: Root mean squared error (m)
    """
    model.eval()

    total_loss = 0.0
    all_errors = []
    num_batches = 0

    for batch in val_loader:
        features = batch[0]
        features = select_features_windowed(features).to(device)
        mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Windowed labels: take final timestep
        if labels.dim() == 4:
            labels = labels[:, -1, :, :]
        elif labels.dim() == 2:
            labels = labels.unsqueeze(-1)

        if use_inference:
            predictions, _ = model.inference(features, mask, num_inference_steps=num_inference_steps)
        else:
            outputs = model(features, mask, labels)
            predictions = outputs['delta_rho_fine']
            loss, _ = criterion(outputs, labels, mask)
            total_loss += loss.item()

        mask_float = mask.unsqueeze(-1).float()
        errors = (predictions - labels) * mask_float
        all_errors.append(errors.cpu())
        num_batches += 1

    all_errors = torch.cat(all_errors, dim=0)  # (total_samples, N, 1)

    valid_mask = (all_errors != 0)
    valid_errors = all_errors[valid_mask]

    mae = torch.abs(valid_errors).mean().item()
    rmse = torch.sqrt((valid_errors ** 2).mean()).item()
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

    return avg_loss, mae, rmse


# ============================================================================
# EVALUATION: DIFF-GNSS
# ============================================================================

def evaluate_diff_gnss_windowed(
        model: nn.Module,
        data_iter: DataLoader,
        device: torch.device,
        num_inference_steps: int = 50,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Evaluate Diff-GNSS model on windowed test data.

    Uses iterative DDIM denoising for refined predictions.

    Returns:
        output_tensor: (num_valid_sats_total, 4)
                       columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]
        losses: List of per-batch L1 losses (refined predictions)
    """
    model.to(device)
    model.eval()

    loss_fn = nn.L1Loss(reduction='none')

    time_step = 0
    time_sum = []
    output_seq = []
    losses = []
    coarse_losses = []
    refined_losses = []

    print(f"\n{'=' * 60}")
    print("EVALUATING DIFF-GNSS MODEL (WINDOWED)")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for batch in data_iter:
            time_step += 1

            features = batch[0]
            mask = batch[1]
            labels = batch[2]
            time_ms = batch[4] if len(batch) > 4 else None
            sat_id = batch[5] if len(batch) > 5 else None

            features = select_features_windowed(features).to(device)
            mask = mask.to(device)
            labels = labels.to(device)

            # Windowed labels: take final timestep (B, S, 1)
            if labels.dim() == 4:
                labels = labels[:, -1, :, :]
            elif labels.dim() == 2:
                labels = labels.unsqueeze(-1)

            if time_ms is not None:
                time_ms = time_ms.to(device)
            if sat_id is not None:
                sat_id = sat_id.to(device)

            # ==================== INFERENCE ====================
            start_time = time.time()

            # Coarse prediction (single forward pass, no ground truth)
            outputs = model(features, mask, labels)
            coarse_predictions = outputs['delta_rho_init']  # (B, S, 1)

            # Refined prediction (iterative denoising)
            refined_predictions = model.inference(
                features, mask, num_inference_steps=num_inference_steps
            )  # (B, S, 1)

            elapsed = time.time() - start_time
            time_sum.append(elapsed)

            # ==================== LOSS ====================
            coarse_loss = loss_fn(coarse_predictions, labels)
            coarse_J = (coarse_loss * mask.unsqueeze(-1).float()).sum() / mask.sum().float()
            coarse_losses.append(coarse_J.cpu())

            refined_loss = loss_fn(refined_predictions, labels)
            refined_J = (refined_loss * mask.unsqueeze(-1).float()).sum() / mask.sum().float()
            refined_losses.append(refined_J.cpu())
            losses.append(refined_J.cpu())

            # ==================== FORMAT OUTPUT ====================
            for i in range(features.shape[0]):
                valid_mask = mask[i]  # (S,)

                if time_ms is not None and sat_id is not None:
                    epoch_time  = time_ms[i, valid_mask, :] if time_ms.dim() == 3 \
                                  else time_ms[i, valid_mask].unsqueeze(-1)
                    epoch_satid = sat_id[i, valid_mask, :] if sat_id.dim() == 3 \
                                  else sat_id[i, valid_mask].unsqueeze(-1)
                else:
                    num_valid = valid_mask.sum().item()
                    epoch_time  = torch.full((num_valid, 1), float(time_step), device=device)
                    epoch_satid = torch.arange(num_valid, device=device).unsqueeze(-1).float()

                epoch_real = labels[i, valid_mask, :]              # (num_valid, 1)
                epoch_pred = refined_predictions[i, valid_mask, :] # (num_valid, 1)

                output_per_epoch = torch.cat([
                    epoch_time.float(), epoch_satid.float(),
                    epoch_real, epoch_pred
                ], dim=1)

                output_seq.append(output_per_epoch.cpu())

    # ==================== SUMMARY ====================
    elapsed_per_batch = sum(time_sum) / len(time_sum)
    coarse_mae  = np.mean([l.item() for l in coarse_losses])
    refined_mae = np.mean([l.item() for l in refined_losses])
    improvement = (coarse_mae - refined_mae) / coarse_mae * 100 if coarse_mae > 0 else 0.0

    output_tensor = torch.cat(output_seq, dim=0)
    errors = output_tensor[:, 2] - output_tensor[:, 3]
    rmse = torch.sqrt((errors ** 2).mean()).item()

    print(f"\n{'=' * 60}")
    print("EVALUATION RESULTS")
    print(f"{'=' * 60}")
    print(f"Inference time per batch: {elapsed_per_batch:.4f}s")
    print(f"Coarse Estimator  MAE:  {coarse_mae:.4f} m")
    print(f"Diffusion Refined MAE:  {refined_mae:.4f} m")
    print(f"Diffusion Refined RMSE: {rmse:.4f} m")
    print(f"Improvement: {improvement:.1f}%")
    print(f"Output shape: {output_tensor.shape}")
    print(f"Columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]")
    print(f"{'=' * 60}")

    return output_tensor, losses
