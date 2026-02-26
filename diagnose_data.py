#!/usr/bin/env python
"""
Diagnostic script to check data and feature computation
"""
import numpy as np
from features import FeatureAdapter, load_preset, get_enabled_features

# Load data
DEPTH_PATH = "data/ETHBTC-depth5-combined.npy"
TRADES_PATH = "data/ETHBTC-trades-combined.npy"

print("Loading data...")
depth_data = np.load(DEPTH_PATH)
trade_data = np.load(TRADES_PATH)

print(f"Depth data shape: {depth_data.shape}")
print(f"Trade data shape: {trade_data.shape}")
print()

# Check first few rows
print("First 3 rows of depth data:")
for i in range(min(3, len(depth_data))):
    print(f"Row {i}: {depth_data[i][:10]}... (showing first 10 cols)")
print()

# Check for NaN/inf in raw data
print("Checking for NaN/inf in raw data...")
depth_nan = np.isnan(depth_data).any()
depth_inf = np.isinf(depth_data).any()
print(f"Depth data has NaN: {depth_nan}")
print(f"Depth data has inf: {depth_inf}")

if depth_nan or depth_inf:
    nan_count = np.isnan(depth_data).sum()
    inf_count = np.isinf(depth_data).sum()
    print(f"  NaN count: {nan_count}")
    print(f"  Inf count: {inf_count}")
    print(f"  Total bad values: {nan_count + inf_count} / {depth_data.size}")
print()

# Load features and test computation
print("Loading feature preset...")
load_preset('live')
feature_names = get_enabled_features()
print(f"Features: {feature_names}")
print()

print("Initializing FeatureAdapter...")
adapter = FeatureAdapter(feature_names)
print()

# Test feature computation on first few rows
print("Computing features for first 5 rows:")
for i in range(min(5, len(depth_data))):
    if i > 0:
        prev_row = depth_data[i-1]
    else:
        prev_row = None
    
    features = adapter.compute_features(depth_data[i], prev_row)
    
    has_nan = np.isnan(features).any()
    has_inf = np.isinf(features).any()
    
    print(f"Row {i}:")
    print(f"  Features: {features}")
    print(f"  Has NaN: {has_nan}, Has inf: {has_inf}")
    
    if has_nan or has_inf:
        for j, (fname, fval) in enumerate(zip(feature_names, features)):
            if np.isnan(fval) or np.isinf(fval):
                print(f"    BAD: {fname} = {fval}")
print()

# Sample features from random indices
print("Computing features from 100 random samples...")
sample_indices = np.random.choice(len(depth_data), min(100, len(depth_data)), replace=False)
all_features = []

for idx in sample_indices:
    if idx > 0:
        prev_row = depth_data[idx-1]
    else:
        prev_row = None
    
    features = adapter.compute_features(depth_data[idx], prev_row)
    all_features.append(features)

all_features = np.array(all_features)

print(f"Feature matrix shape: {all_features.shape}")
print(f"Has NaN: {np.isnan(all_features).any()}")
print(f"Has inf: {np.isinf(all_features).any()}")

if np.isnan(all_features).any() or np.isinf(all_features).any():
    for i, fname in enumerate(feature_names):
        col = all_features[:, i]
        nan_count = np.isnan(col).sum()
        inf_count = np.isinf(col).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"  {fname}: {nan_count} NaN, {inf_count} inf")

print("\nFeature statistics:")
for i, fname in enumerate(feature_names):
    col = all_features[:, i]
    # Use nanmean/nanstd to compute stats ignoring NaN
    mean = np.nanmean(col)
    std = np.nanstd(col)
    min_val = np.nanmin(col)
    max_val = np.nanmax(col)
    print(f"  {fname:20s}: mean={mean:+.4f}, std={std:.4f}, min={min_val:+.4f}, max={max_val:+.4f}")

print("\nDiagnostic complete!")