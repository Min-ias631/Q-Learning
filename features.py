#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler


def transform_none(x):
    return x


def transform_log(x):
    if x <= 0:
        x = 1e-8
    if x > 1e6:
        x = 1e6
    return np.log(x)


def _get_book_data(state):
    """Extract bid/ask price and qty from state dict (handles eda.py and environment.py formats)"""
    if 'd_best_price' in state:
        d = state['d_best_price']
        bid_px, bid_qty = d.get('BID', (0, 0))
        ask_px, ask_qty = d.get('ASK', (0, 0))
    else:
        bid_px = state.get('bid_price', 0)
        ask_px = state.get('ask_price', 0)
        bid_qty = state.get('qbid', state.get('qBID', 0))
        ask_qty = state.get('qask', state.get('qASK', 0))
    return bid_px, bid_qty, ask_px, ask_qty


def calc_ofi(state):
    return state.get('f_ofi', state.get('ofi', 0))


def calc_book_ratio(state):
    _, qbid, _, qask = _get_book_data(state)
    if qask == 0:
        qask = 1e-8
    return qbid / qask


def calc_qbid(state):
    _, qbid, _, _ = _get_book_data(state)
    return qbid


def calc_qask(state):
    _, _, _, qask = _get_book_data(state)
    return qask


def calc_spread(state):
    bid_px, _, ask_px, _ = _get_book_data(state)
    return ask_px - bid_px


def calc_relative_spread(state):
    bid_px, _, ask_px, _ = _get_book_data(state)
    spread = ask_px - bid_px
    mid = state.get('f_mid', (bid_px + ask_px) / 2)
    if mid == 0:
        return 0
    return spread / mid


def calc_depth_imbalance(state):
    _, qbid, _, qask = _get_book_data(state)
    total = qbid + qask
    if total == 0:
        return 0
    return (qbid - qask) / total


def calc_microprice(state):
    bid_px, qbid, ask_px, qask = _get_book_data(state)
    total = qbid + qask
    if total == 0:
        return (bid_px + ask_px) / 2
    return (bid_px * qask + ask_px * qbid) / total


def calc_weighted_depth(state):
    _, qbid, _, qask = _get_book_data(state)
    return qbid + qask


def calc_bid_ask_vol_ratio(state):
    _, qbid, _, qask = _get_book_data(state)
    qbid = max(qbid, 1e-8)
    qask = max(qask, 1e-8)
    return np.log(qbid / qask)


def calc_queue_imbalance(state):
    _, qbid, _, qask = _get_book_data(state)
    total = qbid + qask
    if total == 0:
        return 0.5
    return qbid / total


def calc_price_impact(state):
    bid_px, _, ask_px, _ = _get_book_data(state)
    spread = ask_px - bid_px
    mid = state.get('f_mid', (bid_px + ask_px) / 2)
    if mid == 0:
        return 0
    return spread / (2 * mid)


def calc_log_return(state):
    return state.get('f_logrtn', state.get('log_ret', 0))


def calc_delta_mid(state):
    return state.get('f_delta_mid', state.get('delta_mid', 0))


def calc_return_sign(state):
    ret = state.get('f_logrtn', state.get('log_ret', 0))
    if ret > 0:
        return 1
    elif ret < 0:
        return -1
    return 0


def live_calc_ofi(sense_dict, order_matching):
    return sense_dict.get('qOfi', sense_dict.get('OFI', 0))


def live_calc_book_ratio(sense_dict, order_matching):
    return sense_dict.get('BOOK_RATIO', 1.0)


def live_calc_qbid(sense_dict, order_matching):
    return sense_dict.get('qBid', sense_dict.get('qBID', 0))


def live_calc_qask(sense_dict, order_matching):
    return sense_dict.get('qAsk', sense_dict.get('qASK', 0))


def live_calc_spread(sense_dict, order_matching):
    return sense_dict.get('SPREAD', 0)


def live_calc_relative_spread(sense_dict, order_matching):
    return sense_dict.get('RELATIVE_SPREAD', 0)


def live_calc_depth_imbalance(sense_dict, order_matching):
    return sense_dict.get('DEPTH_IMBALANCE', 0)


def live_calc_microprice(sense_dict, order_matching):
    return sense_dict.get('MICROPRICE', 0)


def live_calc_weighted_depth(sense_dict, order_matching):
    return sense_dict.get('WEIGHTED_DEPTH', 0)


def live_calc_bid_ask_volume_ratio(sense_dict, order_matching):
    return sense_dict.get('BID_ASK_VOL_RATIO', 0)


def live_calc_queue_imbalance(sense_dict, order_matching):
    return sense_dict.get('QUEUE_IMBALANCE', 0.5)


def live_calc_price_impact(sense_dict, order_matching):
    return sense_dict.get('PRICE_IMPACT', 0)


def live_calc_log_return(sense_dict, order_matching):
    return sense_dict.get('logret', sense_dict.get('LOG_RET', 0))


def live_calc_delta_mid(sense_dict, order_matching):
    return sense_dict.get('deltaMid', sense_dict.get('DELTA_MID', 0))


def live_calc_return_sign(sense_dict, order_matching):
    ret = sense_dict.get('logret', sense_dict.get('LOG_RET', 0))
    if ret > 0:
        return 1
    elif ret < 0:
        return -1
    return 0


FEATURE_REGISTRY = {
    'OFI': {
        'calc_fn': calc_ofi,
        'live_fn': live_calc_ofi,
        'transform': transform_none,
        'scaler_file': 'data/scale_ofi.dat',
        'column_name': 'OFI',
        'enabled': True,
        'description': 'Order Flow Imbalance'
    },
    'BOOK_RATIO': {
        'calc_fn': calc_book_ratio,
        'live_fn': live_calc_book_ratio,
        'transform': transform_log,
        'scaler_file': 'data/scale_bookratio.dat',
        'column_name': 'BOOK_RATIO',
        'enabled': True,
        'description': 'Bid qty / Ask qty'
    },
    'qBID': {
        'calc_fn': calc_qbid,
        'live_fn': live_calc_qbid,
        'transform': transform_log,
        'scaler_file': 'data/scale_qbid.dat',
        'column_name': 'qBID',
        'enabled': True,
        'description': 'Quantity at best bid'
    },
    'qASK': {
        'calc_fn': calc_qask,
        'live_fn': live_calc_qask,
        'transform': transform_log,
        'scaler_file': 'data/scale_qask.dat',
        'column_name': 'qASK',
        'enabled': True,
        'description': 'Quantity at best ask'
    },
    'SPREAD': {
        'calc_fn': calc_spread,
        'live_fn': live_calc_spread,
        'transform': transform_none,
        'scaler_file': 'data/scale_spread.dat',
        'column_name': 'SPREAD',
        'enabled': True,
        'description': 'Ask - Bid price'
    },
    'RELATIVE_SPREAD': {
        'calc_fn': calc_relative_spread,
        'live_fn': live_calc_relative_spread,
        'transform': transform_none,
        'scaler_file': 'data/scale_relative_spread.dat',
        'column_name': 'RELATIVE_SPREAD',
        'enabled': True,
        'description': 'Spread / Mid price'
    },
    'DEPTH_IMBALANCE': {
        'calc_fn': calc_depth_imbalance,
        'live_fn': live_calc_depth_imbalance,
        'transform': transform_none,
        'scaler_file': 'data/scale_depth_imbalance.dat',
        'column_name': 'DEPTH_IMBALANCE',
        'enabled': True,
        'description': '(Bid - Ask qty) / Total'
    },
    'MICROPRICE': {
        'calc_fn': calc_microprice,
        'live_fn': live_calc_microprice,
        'transform': transform_none,
        'scaler_file': 'data/scale_microprice.dat',
        'column_name': 'MICROPRICE',
        'enabled': True,
        'description': 'Volume-weighted mid price'
    },
    'WEIGHTED_DEPTH': {
        'calc_fn': calc_weighted_depth,
        'live_fn': live_calc_weighted_depth,
        'transform': transform_log,
        'scaler_file': 'data/scale_weighted_depth.dat',
        'column_name': 'WEIGHTED_DEPTH',
        'enabled': True,
        'description': 'Total depth (bid + ask qty)'
    },
    'BID_ASK_VOL_RATIO': {
        'calc_fn': calc_bid_ask_vol_ratio,
        'live_fn': live_calc_bid_ask_volume_ratio,
        'transform': transform_none,
        'scaler_file': 'data/scale_bavr.dat',
        'column_name': 'BID_ASK_VOL_RATIO',
        'enabled': True,
        'description': 'log(bid_qty / ask_qty)'
    },
    'QUEUE_IMBALANCE': {
        'calc_fn': calc_queue_imbalance,
        'live_fn': live_calc_queue_imbalance,
        'transform': transform_none,
        'scaler_file': 'data/scale_queue_imbalance.dat',
        'column_name': 'QUEUE_IMBALANCE',
        'enabled': True,
        'description': 'bid_qty / total_qty'
    },
    'PRICE_IMPACT': {
        'calc_fn': calc_price_impact,
        'live_fn': live_calc_price_impact,
        'transform': transform_none,
        'scaler_file': 'data/scale_price_impact.dat',
        'column_name': 'PRICE_IMPACT',
        'enabled': True,
        'description': 'Half-spread / mid'
    },
    'LOG_RET': {
        'calc_fn': calc_log_return,
        'live_fn': live_calc_log_return,
        'transform': transform_none,
        'scaler_file': 'data/scale_logret.dat',
        'column_name': 'LOG_RET',
        'enabled': True,
        'description': 'Log return'
    },
    'DELTA_MID': {
        'calc_fn': calc_delta_mid,
        'live_fn': live_calc_delta_mid,
        'transform': transform_none,
        'scaler_file': 'data/scale_delta_mid.dat',
        'column_name': 'DELTA_MID',
        'enabled': True,
        'description': 'Change in mid price'
    },
    'RETURN_SIGN': {
        'calc_fn': calc_return_sign,
        'live_fn': live_calc_return_sign,
        'transform': transform_none,
        'scaler_file': 'data/scale_return_sign.dat',
        'column_name': 'RETURN_SIGN',
        'enabled': True,
        'description': 'Direction of return (+1, 0, -1)'
    },
}


def get_enabled_features():
    return [name for name, config in FEATURE_REGISTRY.items() if config['enabled']]


def enable_features(feature_names):
    for name in feature_names:
        if name in FEATURE_REGISTRY:
            FEATURE_REGISTRY[name]['enabled'] = True


def disable_features(feature_names):
    for name in feature_names:
        if name in FEATURE_REGISTRY:
            FEATURE_REGISTRY[name]['enabled'] = False


def set_enabled_features(feature_names):
    for name in FEATURE_REGISTRY:
        FEATURE_REGISTRY[name]['enabled'] = name in feature_names


def get_feature_columns():
    return [FEATURE_REGISTRY[name]['column_name'] for name in get_enabled_features()]


def list_available_features():
    print(f"\nAvailable features ({len(FEATURE_REGISTRY)} total):")
    for name, config in FEATURE_REGISTRY.items():
        status = "x" if config['enabled'] else " "
        print(f"  [{status}] {name:20s} - {config['description']}")
    print(f"\nEnabled: {len(get_enabled_features())}")


def fit_scalers_from_dataframe(df, save=True):
    scalers = {}
    for name in get_enabled_features():
        config = FEATURE_REGISTRY[name]
        col = config['column_name']
        if col not in df.columns:
            continue
        transform_fn = config['transform']
        values = df[col].apply(transform_fn)
        values = values.replace([np.inf, -np.inf], np.nan).dropna()
        if len(values) == 0:
            continue
        scaler = MinMaxScaler()
        scaler.fit(values.values.reshape(-1, 1))
        scalers[name] = scaler
        if save:
            with open(config['scaler_file'], 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Saved scaler for {name}")
    return scalers


def load_scalers():
    scalers = {}
    for name in get_enabled_features():
        config = FEATURE_REGISTRY[name]
        try:
            scalers[name] = pickle.load(open(config['scaler_file'], 'rb'))
        except FileNotFoundError:
            print(f"Warning: Scaler not found for {name}")
            scalers[name] = None
    return scalers


def transform_features(raw_features, scalers):
    transformed = {}
    for name in get_enabled_features():
        if name not in raw_features:
            continue
        config = FEATURE_REGISTRY[name]
        value = raw_features[name]
        transform_fn = config['transform']
        try:
            transformed_val = transform_fn(value)
            if scalers.get(name) is not None:
                scaled = scalers[name].transform([[transformed_val]])[0][0]
                transformed[name] = np.clip(scaled, 0, 1)
            else:
                transformed[name] = transformed_val
        except:
            transformed[name] = 0.0
    return transformed


def generate_feature_header():
    cols = ['TIME'] + get_feature_columns()
    return '\t'.join(cols)


def generate_feature_row(time_str, feature_values):
    cols = [time_str]
    for name in get_enabled_features():
        col = FEATURE_REGISTRY[name]['column_name']
        cols.append(str(feature_values.get(col, 0)))
    return '\t'.join(cols)


FEATURE_PRESETS = {
    'minimal': ['OFI', 'BOOK_RATIO'],
    'basic': ['OFI', 'BOOK_RATIO', 'SPREAD', 'DEPTH_IMBALANCE', 'LOG_RET'],
    'live': ['OFI', 'BOOK_RATIO', 'SPREAD', 'DEPTH_IMBALANCE', 'MICROPRICE', 
             'LOG_RET', 'RELATIVE_SPREAD', 'QUEUE_IMBALANCE'],
    'full': list(FEATURE_REGISTRY.keys()),
}


def load_preset(preset_name):
    if preset_name not in FEATURE_PRESETS:
        print(f"Unknown preset: {preset_name}")
        print(f"Available: {list(FEATURE_PRESETS.keys())}")
        return
    features = FEATURE_PRESETS[preset_name]
    set_enabled_features(features)
    print(f"Loaded preset '{preset_name}' with {len(features)} features:")
    print(f"  {features}")


# ============================================================================
# FEATURE ADAPTER - Converts raw LOB data to features
# ============================================================================

class FeatureAdapter:
    """
    Adapts raw LOB data from backtest engine to compute features
    Combines functionality that was previously in feature_adapter.py
    """
    
    def __init__(self, feature_names=None):
        """
        Initialize feature adapter
        
        Args:
            feature_names: List of feature names to compute. If None, uses all enabled features.
        """
        if feature_names is None:
            self.feature_names = get_enabled_features()
        else:
            self.feature_names = feature_names
        
        self.feature_dim = len(self.feature_names)
        print(f"FeatureAdapter initialized with {self.feature_dim} features: {self.feature_names}")
    
    def _row_to_state_dict(self, row, prev_row=None):
        """
        Convert raw LOB data row to state dict expected by feature calculation functions
        
        Row format: [timestamp, bid1_px, bid1_qty, bid2_px, bid2_qty, ..., 
                     ask1_px, ask1_qty, ask2_px, ask2_qty, ...]
        """
        state = {}
        
        # Extract best bid/ask
        state['bid_price'] = float(row[1])
        state['qbid'] = float(row[2])
        state['ask_price'] = float(row[11])
        state['qask'] = float(row[12])
        
        # Mid price
        state['f_mid'] = (state['bid_price'] + state['ask_price']) / 2.0
        
        # Compute OFI if we have previous row
        if prev_row is not None:
            prev_bid_px = float(prev_row[1])
            prev_bid_qty = float(prev_row[2])
            prev_ask_px = float(prev_row[11])
            prev_ask_qty = float(prev_row[12])
            
            # Order flow imbalance calculation
            ofi_bid = 0.0
            ofi_ask = 0.0
            
            if state['bid_price'] >= prev_bid_px:
                ofi_bid = state['qbid']
                if state['bid_price'] == prev_bid_px:
                    ofi_bid -= prev_bid_qty
            else:
                ofi_bid = -prev_bid_qty
            
            if state['ask_price'] <= prev_ask_px:
                ofi_ask = state['qask']
                if state['ask_price'] == prev_ask_px:
                    ofi_ask -= prev_ask_qty
            else:
                ofi_ask = -prev_ask_qty
            
            state['f_ofi'] = ofi_bid - ofi_ask
            
            # Log return
            prev_mid = (prev_bid_px + prev_ask_px) / 2.0
            if prev_mid > 0 and state['f_mid'] > 0:
                state['f_logrtn'] = np.log(state['f_mid'] / prev_mid)
            else:
                state['f_logrtn'] = 0.0
            
            # Delta mid
            state['f_delta_mid'] = state['f_mid'] - prev_mid
        else:
            state['f_ofi'] = 0.0
            state['f_logrtn'] = 0.0
            state['f_delta_mid'] = 0.0
        
        return state
    
    def compute_features(self, row, prev_row=None):
        """
        Compute features from raw LOB row
        
        Returns:
            np.ndarray: Array of computed features
        """
        state = self._row_to_state_dict(row, prev_row)
        
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        for i, feature_name in enumerate(self.feature_names):
            if feature_name not in FEATURE_REGISTRY:
                continue
            
            config = FEATURE_REGISTRY[feature_name]
            calc_fn = config['calc_fn']
            
            try:
                value = calc_fn(state)
                # Convert to float and handle NaN/inf
                value = float(value)
                if np.isnan(value) or np.isinf(value):
                    features[i] = 0.0
                else:
                    features[i] = value
            except Exception as e:
                # Default to 0 if calculation fails
                features[i] = 0.0
        
        return features
    
    def get_feature_dim(self):
        """Return the dimensionality of computed features"""
        return self.feature_dim
    
    def get_feature_names(self):
        """Return list of feature names"""
        return self.feature_names.copy()