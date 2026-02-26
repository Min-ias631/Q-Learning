#!/usr/bin/env python3
"""
Split data into alternating 1.5 hour train/test chunks
This prevents overfitting to specific time periods
"""

import numpy as np
from datetime import datetime, timedelta

def split_alternating_chunks(
    instrument='ETHBTC',
    chunk_hours=1.5,
    train_first=True
):
    """
    Split data into alternating train/test chunks
    
    Pattern: [1.5hr train][1.5hr test][1.5hr train][1.5hr test]...
    
    Args:
        instrument: Instrument name
        chunk_hours: Hours per chunk (default 1.5)
        train_first: If True, first chunk is train. If False, first chunk is test.
    """
    
    print("="*80)
    print("ALTERNATING CHUNK SPLIT FOR ROBUST TRAIN/TEST")
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
    
    # Timestamps are in microseconds
    start_ts = depth_data[0, 0]
    end_ts = depth_data[-1, 0]
    
    # Convert to datetime for readability
    start_dt = datetime.fromtimestamp(start_ts / 1_000_000)
    end_dt = datetime.fromtimestamp(end_ts / 1_000_000)
    
    print(f"\nOriginal data:")
    print(f"  Depth rows: {total_rows:,}")
    print(f"  Trade rows: {total_trades:,}")
    print(f"  Start time: {start_dt}")
    print(f"  End time: {end_dt}")
    print(f"  Duration: {(end_ts - start_ts) / 1_000_000 / 3600:.1f} hours")
    
    # Chunk duration in microseconds
    chunk_duration_us = int(chunk_hours * 3600 * 1_000_000)
    
    print(f"\nSplit configuration:")
    print(f"  Chunk duration: {chunk_hours} hours")
    print(f"  Pattern: {'[train][test]' if train_first else '[test][train]'} alternating")
    
    # Create chunks
    train_depth_chunks = []
    test_depth_chunks = []
    train_trade_chunks = []
    test_trade_chunks = []
    
    current_ts = start_ts
    chunk_num = 0
    is_train = train_first
    
    while current_ts < end_ts:
        chunk_start = current_ts
        chunk_end = min(current_ts + chunk_duration_us, end_ts)
        
        # Find depth data in this chunk
        depth_mask = (depth_data[:, 0] >= chunk_start) & (depth_data[:, 0] < chunk_end)
        chunk_depth = depth_data[depth_mask]
        
        # Find trade data in this chunk
        trade_mask = (trade_data[:, 0] >= chunk_start) & (trade_data[:, 0] < chunk_end)
        chunk_trades = trade_data[trade_mask]
        
        # Skip empty chunks (gaps in data)
        if len(chunk_depth) == 0:
            chunk_dt = datetime.fromtimestamp(chunk_start / 1_000_000)
            print(f"  Chunk {chunk_num}: SKIPPED (gap in data at {chunk_dt})")
            current_ts = chunk_end
            chunk_num += 1
            is_train = not is_train  # Toggle even for gaps
            continue
        
        # Add to appropriate list
        if is_train:
            train_depth_chunks.append(chunk_depth)
            train_trade_chunks.append(chunk_trades)
            label = "TRAIN"
        else:
            test_depth_chunks.append(chunk_depth)
            test_trade_chunks.append(chunk_trades)
            label = "TEST"
        
        chunk_start_dt = datetime.fromtimestamp(chunk_start / 1_000_000)
        chunk_end_dt = datetime.fromtimestamp(chunk_end / 1_000_000)
        
        print(f"  Chunk {chunk_num:3d} [{label:5s}]: {chunk_start_dt.strftime('%Y-%m-%d %H:%M')} to "
              f"{chunk_end_dt.strftime('%H:%M')} | {len(chunk_depth):,} rows, {len(chunk_trades):,} trades")
        
        current_ts = chunk_end
        chunk_num += 1
        is_train = not is_train
    
    # Combine chunks
    train_depth = np.vstack(train_depth_chunks) if train_depth_chunks else np.array([])
    test_depth = np.vstack(test_depth_chunks) if test_depth_chunks else np.array([])
    train_trades = np.vstack(train_trade_chunks) if train_trade_chunks else np.array([])
    test_trades = np.vstack(test_trade_chunks) if test_trade_chunks else np.array([])
    
    print(f"\n{'='*80}")
    print("SPLIT SUMMARY")
    print(f"{'='*80}")
    print(f"\nTraining data:")
    print(f"  Chunks: {len(train_depth_chunks)}")
    print(f"  Depth rows: {len(train_depth):,}")
    print(f"  Trade rows: {len(train_trades):,}")
    print(f"  Percentage: {len(train_depth)/total_rows*100:.1f}%")
    
    print(f"\nTest data:")
    print(f"  Chunks: {len(test_depth_chunks)}")
    print(f"  Depth rows: {len(test_depth):,}")
    print(f"  Trade rows: {len(test_trades):,}")
    print(f"  Percentage: {len(test_depth)/total_rows*100:.1f}%")
    
    # Check for gaps
    print(f"\n{'='*80}")
    print("GAP DETECTION")
    print(f"{'='*80}")
    
    def find_gaps(data, threshold_minutes=10):
        """Find gaps larger than threshold in data"""
        gaps = []
        for i in range(1, len(data)):
            time_diff = (data[i, 0] - data[i-1, 0]) / 1_000_000 / 60  # minutes
            if time_diff > threshold_minutes:
                gap_start = datetime.fromtimestamp(data[i-1, 0] / 1_000_000)
                gap_end = datetime.fromtimestamp(data[i, 0] / 1_000_000)
                gaps.append({
                    'start': gap_start,
                    'end': gap_end,
                    'duration_hours': time_diff / 60
                })
        return gaps
    
    train_gaps = find_gaps(train_depth)
    test_gaps = find_gaps(test_depth)
    
    if train_gaps:
        print(f"\nTrain data has {len(train_gaps)} gaps:")
        for i, gap in enumerate(train_gaps[:5]):  # Show first 5
            print(f"  Gap {i+1}: {gap['start'].strftime('%Y-%m-%d %H:%M')} to "
                  f"{gap['end'].strftime('%H:%M')} ({gap['duration_hours']:.1f} hours)")
        if len(train_gaps) > 5:
            print(f"  ... and {len(train_gaps)-5} more gaps")
    else:
        print("\n✓ Train data has no gaps")
    
    if test_gaps:
        print(f"\nTest data has {len(test_gaps)} gaps:")
        for i, gap in enumerate(test_gaps[:5]):
            print(f"  Gap {i+1}: {gap['start'].strftime('%Y-%m-%d %H:%M')} to "
                  f"{gap['end'].strftime('%H:%M')} ({gap['duration_hours']:.1f} hours)")
        if len(test_gaps) > 5:
            print(f"  ... and {len(test_gaps)-5} more gaps")
    else:
        print("\n✓ Test data has no gaps")
    
    # Save split data
    print(f"\n{'='*80}")
    print("SAVING SPLIT DATA")
    print(f"{'='*80}")
    
    if len(train_depth) > 0:
        np.save(f'data/{instrument}-depth5-train.npy', train_depth)
        np.save(f'data/{instrument}-trades-train.npy', train_trades)
        print(f"  ✓ Saved training data")
    else:
        print(f"  ⚠ No training data to save!")
    
    if len(test_depth) > 0:
        np.save(f'data/{instrument}-depth5-test.npy', test_depth)
        np.save(f'data/{instrument}-trades-test.npy', test_trades)
        print(f"  ✓ Saved test data")
    else:
        print(f"  ⚠ No test data to save!")
    
    print(f"\n{'='*80}")
    print("ALTERNATING SPLIT COMPLETE!")
    print(f"{'='*80}")
    
    print("\nNext steps:")
    print("  1. Train on alternating chunks:")
    print(f"     depth_data_path='data/{instrument}-depth5-train.npy'")
    print(f"     trade_data_path='data/{instrument}-trades-train.npy'")
    print()
    print("  2. Evaluate on held-out chunks:")
    print(f"     depth_data_path='data/{instrument}-depth5-test.npy'")
    print(f"     trade_data_path='data/{instrument}-trades-test.npy'")
    print()
    print("Benefits of alternating split:")
    print("  ✓ Tests across different time periods")
    print("  ✓ Prevents overfitting to specific market conditions")
    print("  ✓ More robust evaluation")
    print("  ✓ Better simulates deployment across changing markets")
    print(f"\n{'='*80}")
    
    return {
        'train': {
            'chunks': len(train_depth_chunks),
            'depth_rows': len(train_depth),
            'trade_rows': len(train_trades),
            'gaps': len(train_gaps)
        },
        'test': {
            'chunks': len(test_depth_chunks),
            'depth_rows': len(test_depth),
            'trade_rows': len(test_trades),
            'gaps': len(test_gaps)
        }
    }


if __name__ == "__main__":
    import sys
    
    instrument = sys.argv[1] if len(sys.argv) > 1 else 'ETHBTC'
    chunk_hours = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    
    print(f"Instrument: {instrument}")
    print(f"Chunk duration: {chunk_hours} hours")
    print()
    
    result = split_alternating_chunks(
        instrument=instrument,
        chunk_hours=chunk_hours,
        train_first=True  # Start with train chunk
    )
    
    print("\n✓ Data splitting successful!")