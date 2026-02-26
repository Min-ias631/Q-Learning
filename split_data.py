#!/usr/bin/env python3
"""
Properly split data into train/validation/test sets to prevent overfitting
"""

import numpy as np
from pathlib import Path

def split_data(
    instrument='ETHBTC',
    train_split=0.70,
    val_split=0.15,
    test_split=0.15
):
    """
    Split data chronologically into train/val/test
    
    Args:
        instrument: Instrument name
        train_split: Fraction for training (default 70%)
        val_split: Fraction for validation (default 15%)
        test_split: Fraction for test (default 15%)
    """
    
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"
    
    print("="*80)
    print("DATA SPLITTING FOR PROPER TRAIN/TEST EVALUATION")
    print("="*80)
    
    # Load full data
    depth_file = f'data/{instrument}-depth5-combined.npy'
    trade_file = f'data/{instrument}-trades-combined.npy'
    
    print(f"\nLoading data from:")
    print(f"  {depth_file}")
    print(f"  {trade_file}")
    
    depth_data = np.load(depth_file)
    trade_data = np.load(trade_file)
    
    total_rows = len(depth_data)
    total_trades = len(trade_data)
    
    print(f"\nOriginal data:")
    print(f"  Depth rows: {total_rows:,}")
    print(f"  Trade rows: {total_trades:,}")
    print(f"  Time span: {depth_data[0, 0]:.0f} to {depth_data[-1, 0]:.0f}")
    
    # Calculate split points
    train_end = int(total_rows * train_split)
    val_end = int(total_rows * (train_split + val_split))
    
    print(f"\nSplit configuration:")
    print(f"  Training:   {train_split*100:.0f}% (rows 0 to {train_end:,})")
    print(f"  Validation: {val_split*100:.0f}% (rows {train_end:,} to {val_end:,})")
    print(f"  Test:       {test_split*100:.0f}% (rows {val_end:,} to {total_rows:,})")
    
    # Split depth data
    train_depth = depth_data[:train_end]
    val_depth = depth_data[train_end:val_end]
    test_depth = depth_data[val_end:]
    
    # Split trade data by timestamp
    train_ts_end = depth_data[train_end, 0]
    val_ts_end = depth_data[val_end, 0]
    
    train_trades = trade_data[trade_data[:, 0] < train_ts_end]
    val_trades = trade_data[(trade_data[:, 0] >= train_ts_end) & (trade_data[:, 0] < val_ts_end)]
    test_trades = trade_data[trade_data[:, 0] >= val_ts_end]
    
    print(f"\nResulting splits:")
    print(f"  Training:")
    print(f"    Depth: {len(train_depth):,} rows")
    print(f"    Trades: {len(train_trades):,} rows")
    print(f"  Validation:")
    print(f"    Depth: {len(val_depth):,} rows")
    print(f"    Trades: {len(val_trades):,} rows")
    print(f"  Test:")
    print(f"    Depth: {len(test_depth):,} rows")
    print(f"    Trades: {len(test_trades):,} rows")
    
    # Save split data
    print(f"\nSaving split data...")
    
    np.save(f'data/{instrument}-depth5-train.npy', train_depth)
    np.save(f'data/{instrument}-trades-train.npy', train_trades)
    print(f"  ✓ Saved training data")
    
    np.save(f'data/{instrument}-depth5-val.npy', val_depth)
    np.save(f'data/{instrument}-trades-val.npy', val_trades)
    print(f"  ✓ Saved validation data")
    
    np.save(f'data/{instrument}-depth5-test.npy', test_depth)
    np.save(f'data/{instrument}-trades-test.npy', test_trades)
    print(f"  ✓ Saved test data")
    
    print("\n" + "="*80)
    print("DATA SPLIT COMPLETE!")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Train ONLY on train data:")
    print(f"     depth_data_path='data/{instrument}-depth5-train.npy'")
    print(f"     trade_data_path='data/{instrument}-trades-train.npy'")
    print()
    print("  2. Evaluate ONLY on test data:")
    print(f"     depth_data_path='data/{instrument}-depth5-test.npy'")
    print(f"     trade_data_path='data/{instrument}-trades-test.npy'")
    print()
    print("  3. Use validation data for hyperparameter tuning")
    print()
    print("NEVER train on test data!")
    print("="*80)
    
    return {
        'train': {'depth': len(train_depth), 'trades': len(train_trades)},
        'val': {'depth': len(val_depth), 'trades': len(val_trades)},
        'test': {'depth': len(test_depth), 'trades': len(test_trades)},
    }


if __name__ == "__main__":
    import sys
    
    instrument = sys.argv[1] if len(sys.argv) > 1 else 'ETHBTC'
    
    print(f"Splitting data for instrument: {instrument}\n")
    
    result = split_data(instrument)
    
    print("\nVerifying splits don't overlap...")
    
    # Load and verify
    train_depth = np.load(f'data/{instrument}-depth5-train.npy')
    val_depth = np.load(f'data/{instrument}-depth5-val.npy')
    test_depth = np.load(f'data/{instrument}-depth5-test.npy')
    
    # Check timestamps don't overlap
    train_end_ts = train_depth[-1, 0]
    val_start_ts = val_depth[0, 0]
    val_end_ts = val_depth[-1, 0]
    test_start_ts = test_depth[0, 0]
    
    assert train_end_ts < val_start_ts, "Train and val overlap!"
    assert val_end_ts < test_start_ts, "Val and test overlap!"
    
    print("  ✓ No overlap detected")
    print("  ✓ Chronological order preserved")
    print("\nData splitting successful!")