# debug_script_updated.py
# Test script for updated HFT_env with features and time intervals
# Run: python debug_script_updated.py

import time
import random
import numpy as np

from HFT_env import HFT_env
from features import load_preset, get_enabled_features


# ---- Hardcoded paths (edit if needed) ----
DEPTH_PATH = "data/ETHBTC-depth5-combined.npy"
TRADES_PATH = "data/ETHBTC-trades-combined.npy"

# ---- Hardcoded run settings ----
SEED = 123
N_STEPS = 100
PRINT_EVERY = 10

# ---- Feature and time settings ----
USE_FEATURES = True
FEATURE_PRESET = 'live'  # Options: 'minimal', 'basic', 'live', 'full'
DECISION_INTERVAL_MS = 2000  # 2 seconds


def safe_pending_len(env) -> int:
    try:
        return int(len(env.engine.pending_orders))
    except Exception:
        return -1


def choose_action() -> int:
    """
    Your action map:
      0 = No action
      1 = Market buy
      2 = Market sell
      3..7  = Limit buy levels
      8..12 = Limit sell levels
      13 = Post both
      14 = Cancel all
    """
    r = random.random()
    if r < 0.15:
        return 0
    if r < 0.25:
        return random.choice([1, 2])
    if r < 0.85:
        return random.choice([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    return 14


def print_line(tag: str, t: int, a: int, r: float, cum_r: float, info: dict, pending: int):
    pos = float(info.get("position", 0.0))
    pnl = float(info.get("realized_pnl", 0.0))
    cash = float(info.get("cash", 0.0))
    trades = int(info.get("total_trades", 0))
    timestamp = info.get("timestamp", 0.0)
    steps_taken = info.get("steps_taken", 1)

    print(
        f"{tag} t={t:4d} a={a:2d} r={r:+.6f} cum_r={cum_r:+.6f} "
        f"pos={pos:+.3f} pnl={pnl:+.6f} trades={trades} cash={cash:+.6f} "
        f"pending={pending} ts={timestamp:.0f} steps={steps_taken}"
    )


def main():
    random.seed(SEED)
    np.random.seed(SEED)

    # Load feature preset
    if USE_FEATURES:
        print(f"Loading feature preset: {FEATURE_PRESET}")
        load_preset(FEATURE_PRESET)
        feature_names = get_enabled_features()
        print(f"Using {len(feature_names)} features: {feature_names}")
    else:
        print("Using raw LOB features")
        feature_names = None

    print("\nCreating env...")
    print(f"Decision interval: {DECISION_INTERVAL_MS} ms")
    print(f"Transaction cost: 5.0 bps")
    
    env = HFT_env(
        depth_data_path=DEPTH_PATH,
        trade_data_path=TRADES_PATH,
        transaction_cost_bps=5.0,
        decision_interval_ms=DECISION_INTERVAL_MS,
    )

    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space.n}")

    print("\nResetting...")
    obs, info = env.reset(seed=SEED)

    # Basic checks
    assert isinstance(obs, np.ndarray), "reset() obs must be a numpy array"
    expected_shape = (env.feature_dim + 3,)
    assert obs.shape == expected_shape, f"Expected obs shape {expected_shape}, got {obs.shape}"
    assert np.isfinite(obs).all(), "reset() obs contains NaN/inf"

    print(f"RESET obs.shape={obs.shape} dtype={obs.dtype} min={float(obs.min()):+.4f} max={float(obs.max()):+.4f}")
    print("RESET info:", {k: info.get(k) for k in ("position", "realized_pnl", "total_trades", "cash", "timestamp")})

    cum_r = 0.0

    print("\nStepping... (first step may be slow due to Numba compilation)")
    t0 = time.time()
    a0 = choose_action()
    obs, r, done, info = env.step(a0)
    dt = time.time() - t0

    assert obs.shape == expected_shape, f"Step0: Expected obs shape {expected_shape}, got {obs.shape}"
    assert np.isfinite(obs).all(), "Step0: obs contains NaN/inf"
    assert np.isfinite(r), "Step0: reward is NaN/inf"

    cum_r += float(r)
    print(f"First step took {dt:.3f}s")
    print_line("STEP", 0, a0, float(r), cum_r, info, safe_pending_len(env))

    for t in range(1, N_STEPS):
        a = choose_action()
        obs, r, done, info = env.step(a)

        # Fast-fail checks
        if obs.shape != expected_shape:
            raise RuntimeError(f"t={t}: obs shape wrong: {obs.shape}")
        if not np.isfinite(obs).all():
            raise RuntimeError(f"t={t}: obs has NaN/inf")
        if not np.isfinite(r):
            raise RuntimeError(f"t={t}: reward has NaN/inf")

        cum_r += float(r)

        if (t % PRINT_EVERY) == 0 or done:
            print_line("STEP", t, a, float(r), cum_r, info, safe_pending_len(env))

        if done:
            print("\nDONE reached.")
            break

    print("\nFinal summary:")
    print("cum_reward:", cum_r)
    print("final info:", {k: info.get(k) for k in ("position", "realized_pnl", "total_trades", "cash", "timestamp")})

    env.close()
    print("\nOK - All tests passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\nERROR:", repr(e))
        import traceback
        traceback.print_exc()
        raise