#!/usr/bin/env python3
"""
Compute normalization statistics from TRAINING data only
This prevents data leakage during evaluation
"""

import numpy as np
from features import FeatureAdapter, load_preset

def compute_normalization_stats(
    depth_data_path,
    feature_preset='live',
    sample_size=10000,
    save_path='checkpoints/normalization_stats.npz'
):
    """
    Compute and save normalization statistics from training data
    
    Args:
        depth_data_path: Path to TRAINING depth data (not test!)
        feature_preset: Feature preset to use
        sample_size: Number of samples for computing stats
        save_path: Where to save the stats
    """
    print("="*80)
    print("COMPUTING NORMALIZATION STATS FROM TRAINING DATA ONLY")
    print("="*80)
    
    # Load training data
    print(f"\nLoading training data from: {depth_data_path}")
    depth_data = np.load(depth_data_path)
    print(f"Training data shape: {depth_data.shape}")
    
    # Load feature preset
    load_preset(feature_preset)
    from features import get_enabled_features
    feature_names = get_enabled_features()
    print(f"\nUsing {len(feature_names)} features: {feature_names}")
    
    # Create feature adapter
    feature_adapter = FeatureAdapter(feature_names)
    
    # Sample from training data
    sample_size = min(sample_size, len(depth_data))
    sample_indices = np.random.choice(len(depth_data), sample_size, replace=False)
    
    print(f"\nComputing stats from {sample_size:,} random samples...")
    
    # Compute features for samples
    feature_samples = []
    for i, idx in enumerate(sample_indices):
        if i % 1000 == 0:
            print(f"  Processed {i}/{sample_size}...")
        
        # Get previous row for feature computation
        if idx > 0:
            prev_row = depth_data[idx - 1]
        else:
            prev_row = None
        
        features = feature_adapter.compute_features(depth_data[idx], prev_row)
        feature_samples.append(features)
    
    feature_samples = np.array(feature_samples, dtype=np.float32)
    
    # Replace NaN/inf before computing stats
    feature_samples = np.nan_to_num(feature_samples, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Compute statistics
    obs_mean = np.mean(feature_samples, axis=0)
    obs_std = np.std(feature_samples, axis=0)
    
    # Safety checks
    obs_mean = np.nan_to_num(obs_mean, nan=0.0)
    obs_std = np.nan_to_num(obs_std, nan=1.0)
    obs_std = np.maximum(obs_std, 1e-8)  # Prevent division by zero
    
    print(f"\nStatistics computed:")
    print(f"  Mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
    print(f"  Std range: [{obs_std.min():.4f}, {obs_std.max():.4f}]")
    
    # Save statistics
    np.savez(save_path, obs_mean=obs_mean, obs_std=obs_std, feature_names=feature_names)
    print(f"\n✓ Saved normalization stats to: {save_path}")
    
    print("\n" + "="*80)
    print("NORMALIZATION STATS SAVED")
    print("="*80)
    print("\nNext steps:")
    print("  1. During TRAINING: Load these stats instead of computing")
    print("  2. During EVALUATION: Load these SAME stats (from training)")
    print("  3. NEVER compute stats from test data!")
    print("="*80)
    
    return obs_mean, obs_std


if __name__ == "__main__":
    import sys
    
    # Default to ETHBTC training data
    instrument = sys.argv[1] if len(sys.argv) > 1 else 'ETHBTC'
    
    depth_data_path = f'data/{instrument}-depth5-train.npy'
    
    print(f"\nInstrument: {instrument}")
    print(f"Using training data: {depth_data_path}\n")
    
    compute_normalization_stats(
        depth_data_path=depth_data_path,
        feature_preset='live',
        sample_size=10000,
        save_path='checkpoints/normalization_stats.npz'
    )
    
    print("\n✓ Done! Use these stats for both training and evaluation.")