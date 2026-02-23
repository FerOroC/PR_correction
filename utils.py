import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, Dict
import torch.nn as nn
import time
from models.Fcnn_Lstm import data_random_pack

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_deltas_per_file(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

    TIME_COL = 0
    SAT_ID_COL = 1
    ELEVATION_COL = 5
    POW_COL = 19

    df = df.copy()
    df['Delta_Elevation'] = 0.0
    df['Delta_Power'] = 0.0

    # Group by satellite and compute differences
    for sat_id in df.iloc[:, SAT_ID_COL].unique():
        if sat_id == 0:  # Skip invalid satellites
            continue

        sat_mask = df.iloc[:, SAT_ID_COL] == sat_id
        sat_indices = df[sat_mask].index

        # Compute deltas (current - previous)
        df.loc[sat_indices, 'Delta_Elevation'] = df.loc[sat_indices].iloc[:, ELEVATION_COL].diff().fillna(0)
        df.loc[sat_indices, 'Delta_Power'] = df.loc[sat_indices].iloc[:, POW_COL].diff().fillna(0)

    return df['Delta_Elevation'].values, df['Delta_Power'].values


def load_data_with_deltas_from_filelist(
        files: list,
        input_size: int,
        PRN_size: int,
) -> torch.Tensor:

    print(f"\n{'=' * 60}")
    print(f"Loading {len(files)} file(s)")
    print(f"{'=' * 60}")

    all_epochs = []

    for file_path in files:
        print(f"Processing: {os.path.basename(file_path)}")
        df = pd.read_csv(file_path)

        # Compute deltas
        delta_elev, delta_power = compute_deltas_per_file(df)

        # Convert to numpy, append deltas
        base_data = df.iloc[:, :input_size].values
        data_with_deltas = np.concatenate([
            base_data,
            delta_elev.reshape(-1, 1),
            delta_power.reshape(-1, 1)
        ], axis=1)

        # To tensor
        raw_tensor = torch.tensor(data_with_deltas, dtype=torch.float64)

        # Organize into epochs (PRN_size, input_size+2) per timestep
        epochs = organize_into_epochs(raw_tensor, input_size + 2, PRN_size)
        all_epochs.extend(epochs)

    print(f"\nTotal epochs loaded: {len(all_epochs)}")
    if all_epochs:
        return torch.stack(all_epochs)
    else:
        return torch.empty(0, PRN_size, input_size + 2, dtype=torch.float64)


def organize_into_epochs(data_tensor: torch.Tensor, feature_size: int, PRN_size: int) -> list:

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

def preprocess_features_batch(
        data_tensor: torch.Tensor,
        device: torch.device,
        label_index: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    print(f"\n{'=' * 60}")
    print("PREPROCESSING FEATURES")
    print(f"{'=' * 60}")
    print(f"Input shape: {data_tensor.shape}")
    print(f"Data dtype: {data_tensor.dtype}")

    data_tensor = data_tensor.to(device)
    B, S, F = data_tensor.shape  # Batch, Satellites, Features

    # Indices (adjust based on your data structure)
    IDX_TIME = 0
    IDX_SAT_ID = 1
    IDX_SAT_XYZ = slice(2, 5)
    IDX_ELEVATION = 5
    IDX_DELTA_ELEV = 6
    IDX_AZIMUTH = 7
    IDX_NED = slice(8, 11)
    IDX_PR_RATE = 11
    IDX_RANGE = 12
    IDX_P_RANGE = 13
    IDX_LABEL = label_index  # Raw_Residual or Smoothed_Projected_Residual
    IDX_ATMOSPHERIC = 18
    IDX_POWER = 19
    IDX_HEADING = slice(20, 23)
    IDX_WLS_LON_DEG = 26
    IDX_WLS_LON_MIN = 27
    IDX_WLS_LON_SEC = 28
    IDX_WLS_LAT_DEG = 29
    IDX_WLS_LAT_MIN = 30
    IDX_WLS_LAT_SEC = 31
    IDX_CLOCK_BIAS = 32
    IDX_WLS_XYZ = slice(33, 36)
    IDX_DELTA_ELEV_COMPUTED = F - 2  # Second to last column
    IDX_DELTA_CN0 = F - 1  # Last column
    IDX_H_ITEM_LAST = 17

    # ==================== MASK ====================
    sat_id = data_tensor[:, :, IDX_SAT_ID]
    mask = (sat_id != 0)
    mask_float = mask.unsqueeze(-1).float()

    # ==================== EXTRACT FEATURES ====================
    elev = data_tensor[:, :, IDX_ELEVATION]
    azim = data_tensor[:, :, IDX_AZIMUTH]
    pr_rate = data_tensor[:, :, IDX_PR_RATE]
    power = data_tensor[:, :, IDX_POWER].unsqueeze(-1)
    delta_elev = data_tensor[:, :, IDX_DELTA_ELEV_COMPUTED]
    delta_cn0 = data_tensor[:, :, IDX_DELTA_CN0]

    # Coordinates
    wls_lon_deg = data_tensor[:, :, IDX_WLS_LON_DEG]
    wls_lon_min = data_tensor[:, :, IDX_WLS_LON_MIN]
    wls_lon_sec = data_tensor[:, :, IDX_WLS_LON_SEC]
    wls_lat_deg = data_tensor[:, :, IDX_WLS_LAT_DEG]
    wls_lat_min = data_tensor[:, :, IDX_WLS_LAT_MIN]
    wls_lat_sec = data_tensor[:, :, IDX_WLS_LAT_SEC]

    # Geometry
    ugv = data_tensor[:, :, IDX_NED]
    heading = data_tensor[:, :, IDX_HEADING]

    # For PR residual computation
    Pr = data_tensor[:, :, IDX_P_RANGE:IDX_P_RANGE + 1]
    AtmDelays = data_tensor[:, :, IDX_ATMOSPHERIC:IDX_ATMOSPHERIC + 1]
    svXyz = data_tensor[:, :, IDX_SAT_XYZ]
    wlsXyz = data_tensor[:, :, IDX_WLS_XYZ]
    wlsDtu = data_tensor[:, :, IDX_CLOCK_BIAS:IDX_CLOCK_BIAS + 1]

    sat_prn_norm = data_tensor[:, :, IDX_SAT_ID:IDX_SAT_ID + 1]  # / 32.0

    # ==================== TRIGONOMETRIC FEATURES ====================
    sinE = torch.sin(elev).unsqueeze(-1)
    cosE = torch.cos(elev).unsqueeze(-1)
    sinA = torch.sin(azim).unsqueeze(-1)
    cosA = torch.cos(azim).unsqueeze(-1)

    # ==================== PR RESIDUAL ====================
    geom_range = torch.linalg.norm(svXyz - wlsXyz, dim=-1, keepdim=True)
    pr_resi = Pr - AtmDelays - geom_range - wlsDtu

    # ==================== RSS (Root Sum Square per epoch) ====================
    pr_resi_sq = (pr_resi.squeeze(-1) ** 2) * mask.float()
    rss_per_epoch = torch.sqrt(torch.clamp(pr_resi_sq.sum(dim=1), min=0.0))
    rss_feat = rss_per_epoch.view(B, 1, 1).repeat(1, S, 1)

    # ==================== VISIBLE SATELLITES COUNT ====================
    visible_sats_count = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    visible_sats_feat = visible_sats_count.repeat(1, S, 1)

    # ==================== NORMALIZE COORDINATES ====================
    wls_lon_deg_n = (wls_lon_deg / 180.0).unsqueeze(-1)
    wls_lon_min_n = (wls_lon_min / 60.0).unsqueeze(-1)
    lon_sec_normalized = (wls_lon_sec / 60.0).unsqueeze(-1)
    wls_lon_sec_n = torch.round(lon_sec_normalized * 1000) / 1000
    wls_lat_deg_n = (wls_lat_deg / 90.0).unsqueeze(-1)
    wls_lat_min_n = (wls_lat_min / 60.0).unsqueeze(-1)
    lat_sec_normalized = (wls_lat_sec / 60.0).unsqueeze(-1)
    wls_lat_sec_n = torch.round(lat_sec_normalized * 1000) / 1000

    # ==================== NEW: WLS CLOCK DELTA (Feature 2) ====================
    # Get clock bias (B, S, 1)
    wls_clock_epoch = wlsDtu
    # Get previous clock bias (B, S, 1)
    prev_wls_clock_epoch = torch.roll(wls_clock_epoch, shifts=1, dims=0)
    # Calculate delta
    delta_clock = wls_clock_epoch - prev_wls_clock_epoch
    # Zero out the delta for the first item in the batch
    delta_clock[0, :, :] = 0.0
    # This is an epoch-wide feature, so repeat it
    delta_clock_feat = delta_clock[:, 0:1, :]  # (B, 1, 1)
    delta_clock_feat = delta_clock_feat.repeat(1, S, 1)  # (B, S, 1)

    # ==================== CONCATENATE ALL FEATURES ====================
    features = torch.cat([
        sinE, cosE, sinA, cosA,  # 0-3: Trig features
        power,  # 4: Power
        pr_rate.unsqueeze(-1),  # 5: PR rate
        delta_elev.unsqueeze(-1),  # 6: Delta elevation
        delta_cn0.unsqueeze(-1),  # 7: Delta CN0
        wls_lon_deg_n, wls_lon_min_n, wls_lon_sec_n,  # 8-10: Longitude
        wls_lat_deg_n, wls_lat_min_n, wls_lat_sec_n,  # 11-13: Latitude
        ugv,  # 14-16: Unit geometry vector (NED)
        heading,  # 17-19: Heading (NED)
        pr_resi,  # 20: PR residual
        rss_feat,  # 21: RSS
        visible_sats_feat,  # 22: Visible satellites count
        sat_prn_norm,  # 23: Normalized satellite PRN
    ], dim=-1)  # Shape: (B, S, 24)

    # ==================== EXTRACT LABELS ====================
    labels = data_tensor[:, :, IDX_LABEL:IDX_LABEL + 1]
    clock_corrections = data_tensor[:, :, IDX_H_ITEM_LAST:IDX_H_ITEM_LAST + 1]

    # ==================== CLEAN NANS ====================
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0) * mask_float
    labels = labels * mask_float

    time_ms = data_tensor[:, :, IDX_TIME:IDX_TIME + 1]  # (B, S, 1)
    sat_id_data = data_tensor[:, :, IDX_SAT_ID:IDX_SAT_ID + 1]  # (B, S, 1)

    print(f"Output features shape: {features.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Labels shape: {labels.shape}")

    return features, mask, labels, clock_corrections, time_ms, sat_id_data


# ============================================================================
# STEP 4: NORMALIZATION
# ============================================================================

def compute_normalization_stats(features: torch.Tensor, mask: torch.Tensor) -> Dict:

    print(f"\n{'=' * 60}")
    print("COMPUTING NORMALIZATION STATISTICS")
    print(f"{'=' * 60}")

    stats = {}

    # Z-score normalization indices
    z_score_features = {
        'power': 4,
        'pr_rate': 5,
        'delta_elev': 6,
        'delta_cn0': 7,
        'pr_resi': 20,
        'rss': 21,
    }

    for name, idx in z_score_features.items():
        values = features[:, :, idx][mask]
        mean_val = values.mean().item()
        std_val = values.std().item()
        stats[name] = {'mean': mean_val, 'std': std_val}
        print(f"  {name:15s}: mean={mean_val:10.4f}, std={std_val:10.4f}")

    # Min-max normalization indices
    minmax_features = {
        'visible_sats': 22
    }

    for name, idx in minmax_features.items():
        values = features[:, :, idx][mask]
        min_val = values.min().item()
        max_val = values.max().item()
        stats[name] = {'min': min_val, 'max': max_val}
        print(f"  {name:15s}: min={min_val:10.4f}, max={max_val:10.4f}")

    return stats


def apply_normalization(
        features: torch.Tensor,
        mask: torch.Tensor,
        stats: Dict
) -> torch.Tensor:

    print(f"\n{'=' * 60}")
    print("APPLYING NORMALIZATION")
    print(f"{'=' * 60}")

    features_norm = features.clone()
    mask_float = mask.unsqueeze(-1).float()

    # Z-score normalization
    z_score_features = {
        'power': 4,
        'pr_rate': 5,
        'delta_elev': 6,
        'delta_cn0': 7,
        'pr_resi': 20,
        'rss': 21,
    }

    for name, idx in z_score_features.items():
        mean = stats[name]['mean']
        std = stats[name]['std']
        if std > 1e-8:
            features_norm[:, :, idx] = (features[:, :, idx] - mean) / std
        features_norm[:, :, idx] = features_norm[:, :, idx] * mask_float.squeeze(-1)
        print(f"  Normalized {name} (z-score)")

    # Min-max normalization
    minmax_features = {
        'visible_sats': 22
    }

    for name, idx in minmax_features.items():
        min_val = stats[name]['min']
        max_val = stats[name]['max']
        if max_val - min_val > 1e-8:
            features_norm[:, :, idx] = (features[:, :, idx] - min_val) / (max_val - min_val)
        features_norm[:, :, idx] = features_norm[:, :, idx] * mask_float.squeeze(-1)
        print(f"  Normalized {name} (min-max)")

    return features_norm

def create_dataloaders(
        data_directory: str,
        input_size: int = 34,
        PRN_size: int = 32,
        batch_size: int = 64,
        label_index: int = 14,
        device: torch.device = None,
        test_files: list = None,
):
    """
    If test_files is provided, those files are used for testing.
    All remaining files are used for training.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("CREATING DATALOADERS (multi-test file support)")
    print("=" * 60)
    print(f"Device: {device}")

    # Gather files
    all_files = sorted(
        os.path.join(data_directory, f)
        for f in os.listdir(data_directory)
        if f.endswith(".csv")
    )
    assert len(all_files) >= 2, "Need at least 2 files"

    # --- Match multiple test files ---
    if test_files is None or len(test_files) == 0:
        raise ValueError("You must specify one or more test files.")

    test_file_paths = []
    for name in test_files:
        matches = [f for f in all_files if name in os.path.basename(f)]
        assert len(matches) == 1, f"Could not uniquely match test file pattern '{name}'"
        test_file_paths.append(matches[0])

    # Remaining files → TRAIN
    train_files = [f for f in all_files if f not in test_file_paths]

    print("\nTRAIN FILES:")
    for f in train_files: print("  -", os.path.basename(f))
    print("\nTEST FILES:")
    for f in test_file_paths: print("  -", os.path.basename(f))
    print("=" * 60)

    # Load data tensors
    train_data_raw = load_data_with_deltas_from_filelist(train_files, input_size, PRN_size)

    train_features, train_mask, train_labels, train_clock, train_time, train_satid = preprocess_features_batch(
        train_data_raw, device, label_index)
    norm_stats = compute_normalization_stats(train_features, train_mask)
    train_features = apply_normalization(train_features, train_mask, norm_stats)

    # Build train dataset
    train_dataset = TensorDataset(train_features, train_mask, train_labels, train_clock, train_time, train_satid)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build one test loader per test file
    test_loaders = {}
    for name, path in zip(test_files, test_file_paths):
        raw = load_data_with_deltas_from_filelist([path], input_size, PRN_size)
        feat, mask_, lbl, clk, t, sid = preprocess_features_batch(raw, device, label_index)

        print("\n" + "=" * 60)
        print(f"NORMALIZING TEST DATA ({name}) USING TRAIN STATS")
        print("=" * 60)
        feat = apply_normalization(feat, mask_, norm_stats)

        ds = TensorDataset(feat, mask_, lbl, clk, t, sid)
        test_loaders[name] = DataLoader(ds, batch_size=batch_size, shuffle=False)

    print("\n" + "=" * 60)
    print("DATALOADER SUMMARY")
    print("=" * 60)
    print(f"TRAIN EPOCHS: {len(train_dataset)}")
    for name, loader in test_loaders.items():
        print(f"TEST  EPOCHS ({name}): {len(loader.dataset)}")

    return train_loader, test_loaders, norm_stats


def train_gnss_net(net, train_loader, lr, num_epochs, num_iterations, device):

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.L1Loss(reduction='none')

    net.train()

    time_step = 0
    time_sum = []

    num_batches_train = len(train_loader)

    for epoch in range(num_epochs):
        epoch_losses = []
        for features, mask, labels, clock, time_ms, sat_id in train_loader:
            # Select 19 features
            features = torch.cat([features[:, :, 0:1], features[:, :, 2:3], features[:, :, 4:6],
                                  features[:, :, 8:20], features[:, :, 20:22], features[:, :, 23:24]], dim=-1)

            optimizer.zero_grad()
            start_time = time.time()

            time_step = time_step + 1

            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            total_prm_error = net(features, mask)

            loss_per_satellite = criterion(total_prm_error, labels)  # (B, S, 1)
            J = (loss_per_satellite * mask.unsqueeze(-1).float()).sum() / mask.sum().float()

            J.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()
            epoch_losses.append(J.detach().item())

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_sum.append(elapsed_time)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Loss: {float(np.mean(epoch_losses)):.4f}")
        scheduler.step()

    elapsed_time_per_batch = sum(time_sum) / len(time_sum)
    print('Training time per batch: ', elapsed_time_per_batch)
    return optimizer


def evaluate_gnss_net(net, data_iter, device, label_index):
    net.to(device)
    net.eval()

    loss_fn = nn.L1Loss(reduction='none')

    time_step = 0
    time_sum = []
    output_seq = []
    losses = []

    print(f"\n{'=' * 60}")
    print("EVALUATING MODEL")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for features, mask, labels, clock, time_ms, sat_id in data_iter:
            time_step += 1
            # Select 19 features
            features = torch.cat([features[:, :, 0:1], features[:, :, 2:3], features[:, :, 4:6],
                                  features[:, :, 8:20], features[:, :, 20:22], features[:, :, 23:24]], dim=-1)

            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            time_ms = time_ms.to(device)
            sat_id = sat_id.to(device)

            start_time = time.time()
            predictions = net(features, mask)
            elapsed_time = time.time() - start_time
            time_sum.append(elapsed_time)

            # Calculate loss
            loss_per_satellite = loss_fn(predictions, labels)
            J = (loss_per_satellite * mask.unsqueeze(-1).float()).sum() / mask.sum().float()
            losses.append(J.cpu())

            # ==================== FORMAT OUTPUT ====================
            for i in range(features.shape[0]):
                valid_mask = mask[i]  # (PRN_size,)

                epoch_time = time_ms[i, valid_mask, :]   # (num_valid, 1)
                epoch_satid = sat_id[i, valid_mask, :]   # (num_valid, 1)
                epoch_real = labels[i, valid_mask, :]    # (num_valid, 1)
                epoch_pred = predictions[i, valid_mask, :]  # (num_valid, 1)

                output_per_epoch = torch.cat([
                    epoch_time,
                    epoch_satid,
                    epoch_real,
                    epoch_pred
                ], dim=1)  # Shape: (num_valid, 4)

                output_seq.append(output_per_epoch.cpu())

    elapsed_time_per_sample = sum(time_sum) / len(time_sum)
    print(f'\nInference time per sample: {elapsed_time_per_sample:.6f}s')
    print(f'Average loss: {np.mean([l.item() for l in losses]):.6f}')

    output_tensor = torch.cat(output_seq, dim=0)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]")

    return output_tensor, losses


def train_gnss_fcnn_lstm(net, train_loader, lr, num_epochs, num_iterations, device):
    """Train the Fcnn_Lstm model.
    Feature selection: 18 features (indices 0:1, 2:3, 4:6, 8:19, 20:22, 23:24).
    """

    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    criterion = nn.L1Loss(reduction='none')

    net.train()

    time_step = 0
    time_sum = []

    num_batches_train = len(train_loader)

    for epoch in range(num_epochs):
        for features, mask, labels, clock, time_ms, sat_id in train_loader:
            # Select 19 features
            features = torch.cat([features[:, :, 0:1], features[:, :, 2:3], features[:, :, 4:6],
                                  features[:, :, 8:20], features[:, :, 20:22], features[:, :, 23:24]], dim=-1)
            optimizer.zero_grad()
            start_time = time.time()

            time_step = time_step + 1

            features = features.to(device)
            # shuffling the features for Fcnn Lstm
            features_random_pack = data_random_pack(features, mask)
            total_prm_error = net(features, features_random_pack)  # (batch_size, PRN_size, 1)

            # L1 loss
            batch_loss = criterion(total_prm_error, labels)  # (batch_size, PRN_size, 1)
            batch_loss = batch_loss.squeeze(-1)              # (batch_size, PRN_size)

            # Apply mask and compute mean
            masked_loss = batch_loss * mask.float()
            valid_samples = mask.sum()

            J = masked_loss.sum() / valid_samples

            J.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
            optimizer.step()

            end_time = time.time()
            elapsed_time = end_time - start_time
            time_sum.append(elapsed_time)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1} | Loss: {J.cpu().detach().item():.4f}")
        scheduler.step()

    elapsed_time_per_batch = sum(time_sum) / len(time_sum)
    print('Training time per batch: ', elapsed_time_per_batch)
    return optimizer


def evaluate_gnss_fcnn_lstm(net, data_iter, device, label_index=15):
    """
    Evaluate Fcnn_Lstm model with proper output format.

    Returns:
        output_seq: Tensor (num_satellites_total, 4)
                    [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]
        losses: List of L1 losses per batch
    """
    net.to(device)
    net.eval()

    loss_fn = nn.L1Loss(reduction='none')

    time_step = 0
    time_sum = []
    output_seq = []
    losses = []

    print(f"\n{'=' * 60}")
    print("EVALUATING MODEL")
    print(f"{'=' * 60}")

    with torch.no_grad():
        for features, mask, labels, clock, time_ms, sat_id in data_iter:
            time_step += 1
            # Select 19 features
            features = torch.cat([features[:, :, 0:1], features[:, :, 2:3], features[:, :, 4:6],
                                  features[:, :, 8:20], features[:, :, 20:22], features[:, :, 23:24]], dim=-1)

            features = features.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            time_ms = time_ms.to(device)
            sat_id = sat_id.to(device)

            start_time = time.time()
            features_random_pack = data_random_pack(features, mask)
            predictions = net(features, features_random_pack)  # (batch_size, PRN_size, 1)
            elapsed_time = time.time() - start_time
            time_sum.append(elapsed_time)

            # Calculate loss
            loss_per_satellite = loss_fn(predictions, labels)
            J = (loss_per_satellite * mask.unsqueeze(-1).float()).sum() / mask.sum().float()
            losses.append(J.cpu())

            # ==================== FORMAT OUTPUT ====================
            for i in range(features.shape[0]):
                valid_mask = mask[i]  # (PRN_size,)

                epoch_time = time_ms[i, valid_mask, :]      # (num_valid, 1)
                epoch_satid = sat_id[i, valid_mask, :]      # (num_valid, 1)
                epoch_real = labels[i, valid_mask, :]       # (num_valid, 1)
                epoch_pred = predictions[i, valid_mask, :]  # (num_valid, 1)

                output_per_epoch = torch.cat([
                    epoch_time,
                    epoch_satid,
                    epoch_real,
                    epoch_pred
                ], dim=1)  # Shape: (num_valid, 4)

                output_seq.append(output_per_epoch.cpu())

    elapsed_time_per_sample = sum(time_sum) / len(time_sum)
    print(f'\nInference time per sample: {elapsed_time_per_sample:.6f}s')
    print(f'Average loss: {np.mean([l.item() for l in losses]):.6f}')

    output_tensor = torch.cat(output_seq, dim=0)
    print(f"Output shape: {output_tensor.shape}")
    print(f"Columns: [Time_ms, Sat_ID, Real_PR_res, Predicted_PR_res]")

    return output_tensor, losses
