import os
import glob
import pandas as pd
import numpy as np

BASE_DIR = '/mnt/sda/Learning'
OUTPUT_DIR = './data'
INSTRUMENT = 'ETHBTC'

SKIP = False

def process_files(datatype):
    search_path = os.path.join(BASE_DIR, '**', 'binance', 'spot', datatype, INSTRUMENT, '*.csv')
    files = glob.glob(search_path, recursive = True)
    files.sort()

    is_depth = (datatype == 'depth5')

    os.makedirs(OUTPUT_DIR, exist_ok = True)

    combined_data = []

    for file in files:
        fname = os.path.basename(file).replace('.csv','.npy')
        save_path = os.path.join(OUTPUT_DIR, fname)

        if SKIP and os.path.isfile(save_path):
            print(f'Found and Skipped {save_path}')
            data = np.load(save_path)
            combined_data.append(data)
            continue
        df = pd.read_csv(file)

        drop_columns = ['symbol', 'ts', 'lastUpdateId'] if is_depth else ['symbol', 'tradeId', 'isBestMatch', 'ts']
        df = df.drop(columns = drop_columns)
        df['ts_us'] = pd.to_datetime(df['ts_us'], unit = 'us').astype(np.int64) // 1000

        if not is_depth:
            cols = ['ts_us'] + [col for col in df.columns if col != 'ts_us']
            df = df[cols]
        
        df = df[df['ts_us'].diff() != 0]
        df = df.to_numpy(dtype=np.float64)

        combined_data.append(df)

        #np.save(save_path, df)
        #print(f'Saved to {save_path}')
    
    combined = np.vstack(combined_data)
    combined_path = os.path.join(OUTPUT_DIR, f'{INSTRUMENT}-{datatype}-combined.npy')
    np.save(combined_path, combined)
    print(f'Saved combined: {combined_path} with shape {combined.shape}')

if __name__ == '__main__':
    process_files('depth5')
    process_files('trades')
    print('=' * 70)
    print('Done Saving')