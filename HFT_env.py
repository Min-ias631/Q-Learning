import gymnasium as gym
import numpy as np
import os
from typing import Tuple, Dict, Any, Optional
from backtest_engine import BacktestEngine
from features import FeatureAdapter, get_enabled_features

class HFT_env(gym.Env):
    
    def __init__(
        self,
        depth_data_path: str,
        trade_data_path: str,
        normalization_stats_path: Optional[str] = None,
        transaction_cost_bps: float = 5.0,
        max_position: float = 5.0,
        initial_cash: float = 1.0,
        reward_scaling: float = 1000.0,
        order_expire_steps: int = 100,
        decision_interval_ms: int = 2000,
        reward_clip: float = 10.0,
        inventory_penalty_factor: float = 0.00,
        blocked_action_penalty: float = 0.0,
    ):
        super().__init__()

        self.depth_data = np.load(depth_data_path)
        self.trade_data = np.load(trade_data_path)

        self.transaction_cost_bps = transaction_cost_bps
        self.max_position = max_position
        self.initial_cash = initial_cash
        self.reward_scaling = reward_scaling
        self.order_expire_steps = order_expire_steps
        self.reward_clip = reward_clip
        self.inventory_penalty_factor = inventory_penalty_factor
        self.blocked_action_penalty = blocked_action_penalty

        self.decision_interval_ms = decision_interval_ms
        
        feature_names = get_enabled_features()
        self.feature_adapter = FeatureAdapter(feature_names)
        self.feature_dim = self.feature_adapter.get_feature_dim()
        print(f'Using {self.feature_dim} features: {self.feature_adapter.get_feature_names()}')

        self.action_space = gym.spaces.Discrete(15)
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = (self.feature_dim + 3, ),
            dtype = np.float32
        )

        self.engine = BacktestEngine(
            depth_data=self.depth_data,
            trade_data=self.trade_data,
            transaction_cost_bps=transaction_cost_bps,
            max_position=max_position,
            initial_cash=initial_cash,
            reward_scaling=reward_scaling,
            order_expire_steps=order_expire_steps
        )

        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0
        self.last_decision_timestamp = 0.0

        # Load or compute normalization stats
        if normalization_stats_path and os.path.exists(normalization_stats_path):
            print(f"✓ Loading pre-computed normalization stats from: {normalization_stats_path}")
            stats = np.load(normalization_stats_path)
            self.obs_mean = stats['obs_mean']
            self.obs_std = stats['obs_std']
            print(f"  Mean range: [{self.obs_mean.min():.4f}, {self.obs_mean.max():.4f}]")
            print(f"  Std range: [{self.obs_std.min():.4f}, {self.obs_std.max():.4f}]")
            print("  Using training data stats (prevents distribution shift)")
        else:
            print("⚠ Computing normalization stats from current data...")
            self._compute_normalization_stats()
            if 'test' in depth_data_path.lower() or 'val' in depth_data_path.lower():
                print("⚠⚠⚠ WARNING: Computing stats from test/val data causes distribution shift!")
                print("⚠⚠⚠ Use: normalization_stats_path='checkpoints/normalization_stats.npz'")
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics for features"""
        sample_size = min(10_000, len(self.depth_data))
        sample_indices = np.random.choice(len(self.depth_data), sample_size, replace = False)

        feature_samples = []
        for i in range(len(sample_indices)):
            idx = sample_indices[i]
            
            # Get previous row for feature computation
            if idx > 0:
                prev_row = self.depth_data[idx - 1]
            else:
                prev_row = None
                
            features = self.feature_adapter.compute_features(
                self.depth_data[idx], prev_row
            )
            feature_samples.append(features)

        feature_samples = np.array(feature_samples, dtype = np.float32)
        
        # Replace NaN/inf before computing stats
        feature_samples = np.nan_to_num(feature_samples, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.obs_mean = np.mean(feature_samples, axis = 0)
        self.obs_std = np.std(feature_samples, axis = 0)
        
        # Safety checks
        self.obs_mean = np.nan_to_num(self.obs_mean, nan=0.0)
        self.obs_std = np.nan_to_num(self.obs_std, nan=1.0)

        # Prevent division by zero
        self.obs_std = np.maximum(self.obs_std, 1e-8)
        
        print(f"Feature normalization stats computed:")
        print(f"  Mean range: [{self.obs_mean.min():.4f}, {self.obs_mean.max():.4f}]")
        print(f"  Std range: [{self.obs_std.min():.4f}, {self.obs_std.max():.4f}]")
    
    def _normalize_observation(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observation:
        - First feature_dim dimensions: z-score normalization
        - Last 3 dimensions: already scaled (position, pending_orders, pnl)
        """
        obs_norm = obs.astype(np.float32, copy = True)
        
        # Normalize features only
        obs_norm[:self.feature_dim] = (obs_norm[:self.feature_dim] - self.obs_mean) / self.obs_std
        
        # Replace any remaining NaN/inf
        obs_norm = np.nan_to_num(obs_norm, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return obs_norm

    def _build_observation(self, idx: int) -> np.ndarray:
        """
        Build complete observation from current state
        
        Observation structure:
        [0:feature_dim] = LOB features
        [feature_dim] = normalized position
        [feature_dim+1] = normalized pending order count
        [feature_dim+2] = normalized total PnL
        """
        obs = np.empty(self.feature_dim + 3, dtype = np.float64)
        
        # Get LOB features
        if idx > 0:
            prev_row = self.depth_data[idx - 1]
        else:
            prev_row = None
            
        obs[:self.feature_dim] = self.feature_adapter.compute_features(
            self.depth_data[idx], prev_row
        )
        
        # Get current mid price for PnL calculation
        current_row = self.depth_data[idx]
        current_mid = (current_row[1] + current_row[11]) * 0.5
        
        # Calculate total PnL (realized + unrealized)
        total_pnl = self.engine.get_total_pnl(current_mid)
        
        # Agent state features (already normalized)
        obs[self.feature_dim] = self.engine.position / self.max_position
        obs[self.feature_dim + 1] = float(len(self.engine.pending_orders)) / 10.0
        obs[self.feature_dim + 2] = total_pnl / self.initial_cash
        
        return obs
    
    def _find_next_decision_time(self, current_timestamp: float) -> int:
        """
        Find the index of the next decision point
        
        CRITICAL: Timestamps in data are in MICROSECONDS
        decision_interval_ms is in MILLISECONDS
        Must convert: ms * 1000 = microseconds
        """
        # Convert milliseconds to microseconds
        target_timestamp = current_timestamp + (self.decision_interval_ms * 1000)

        current_idx = self.engine.idx

        while current_idx < len(self.depth_data):
            if self.depth_data[current_idx, 0] >= target_timestamp:
                return current_idx
            current_idx += 1

        return len(self.depth_data) - 1

    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to initial state with random starting position"""
        super().reset(seed = seed)
        
        max_start = max(0, len(self.depth_data) - 500_000)
        if max_start > 0:
            start_idx = self.np_random.integers(0, max_start)  # Uses gym's RNG
        else:
            start_idx = 0
        # Reset with random starting position (-1 = random)
        # IMPORTANT: Use positional arg, not keyword (Numba jitclass limitation)
        self.engine.reset_state(start_idx)
        
        # Take first step to initialize
        _, _, _, _ = self.engine.step(0)  # no-op action

        self.last_decision_timestamp = self.depth_data[self.engine.idx, 0]
        
        # Build and normalize observation
        obs = self._build_observation(self.engine.idx)
        obs = self._normalize_observation(obs)

        self.episode_step = 0
        self.episode_reward = 0.0
        self.episode_pnl = 0.0

        # Get current state for info
        current_row = self.depth_data[self.engine.idx]
        current_mid = (current_row[1] + current_row[11]) * 0.5
        total_pnl = self.engine.get_total_pnl(current_mid)
        
        info = {
            'realized_pnl': self.engine.realized_pnl,
            'total_pnl': total_pnl,
            'total_trades': self.engine.total_trades,
            'position': self.engine.position,
            'cash': self.engine.cash,
            'timestamp': self.last_decision_timestamp,
            'start_idx': self.engine.idx  # NEW: track where episode started
        }
        
        return obs.astype(np.float32), info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Steps until next decision point, accumulating rewards
        """
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid Action {action} in HFT_env')

        current_timestamp = self.depth_data[self.engine.idx, 0]

        # Snapshot engine state before action to detect if it was blocked
        prev_position = self.engine.position
        prev_pending_count = len(self.engine.pending_orders)
        prev_trades = self.engine.total_trades

        # Execute the action
        reward, done, current_mid, next_mid = self.engine.step(action)

        # An action is "blocked" if it was non-zero but changed nothing
        # (e.g. sell at max short, buy with no cash, cancel with no orders)
        action_executed = (
            action == 0
            or self.engine.total_trades != prev_trades
            or len(self.engine.pending_orders) != prev_pending_count
            or abs(self.engine.position - prev_position) > 1e-12
        )

        cumulative_reward = reward
        steps_taken = 1

        # Step forward until next decision point (with no-op actions)
        if not done:
            next_decision_idx = self._find_next_decision_time(current_timestamp)

            while self.engine.idx < next_decision_idx and not done:
                step_reward, done, _, next_mid = self.engine.step(0)  # no-op
                cumulative_reward += step_reward
                steps_taken += 1

            if not done:
                self.last_decision_timestamp = self.depth_data[self.engine.idx, 0]
        
        # Build observation
        if not done:
            obs = self._build_observation(self.engine.idx)
        else:
            # Terminal state - zero features, current agent state
            obs = np.zeros(self.feature_dim + 3, dtype=np.float64)
            obs[self.feature_dim] = self.engine.position / self.max_position
            obs[self.feature_dim + 1] = float(len(self.engine.pending_orders)) / 10.0
            obs[self.feature_dim + 2] = self.engine.realized_pnl / self.initial_cash  # Terminal PnL
        
        obs = self._normalize_observation(obs)

        # Clip reward to prevent extreme values
        cumulative_reward = np.clip(cumulative_reward, -self.reward_clip, self.reward_clip)

        # Penalty 1: discourage actions that did nothing (blocked by position/cash limits)
        # This separates Q(sell_when_blocked) from Q(sell_when_executes)
        if not action_executed:
            cumulative_reward -= self.blocked_action_penalty

        # Penalty 2: inventory penalty — penalise holding large one-sided positions
        # Prevents the agent earning "free" rewards by just sitting on a max short/long
        norm_position = self.engine.position / self.max_position  # in [-1, 1]
        cumulative_reward -= self.inventory_penalty_factor * (norm_position ** 2)

        # Update episode tracking
        self.episode_step += 1
        self.episode_reward += cumulative_reward
        
        # Get final PnL (realized at episode end, total during episode)
        current_row = self.depth_data[min(self.engine.idx, len(self.depth_data) - 1)]
        current_mid_for_pnl = (current_row[1] + current_row[11]) * 0.5
        
        if done:
            self.episode_pnl = self.engine.realized_pnl
            total_pnl = self.engine.realized_pnl
        else:
            total_pnl = self.engine.get_total_pnl(current_mid_for_pnl)
            self.episode_pnl = total_pnl

        # Build info dict
        info = {
            'realized_pnl': self.engine.realized_pnl,
            'total_pnl': total_pnl,
            'total_trades': self.engine.total_trades,
            'position': self.engine.position,
            'cash': self.engine.cash,
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'timestamp': self.depth_data[self.engine.idx, 0] if not done else current_timestamp,
            'steps_taken': steps_taken
        }

        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_step,
                'pnl': self.episode_pnl,
                'trades': self.engine.total_trades
            }
        
        return obs.astype(np.float32), cumulative_reward, done, info

    def render(self):
        pass
    
    def close(self):
        pass

    def get_episode_stats(self) -> Dict[str, float]:
        return {
            'episode_step': self.episode_step,
            'episode_reward': self.episode_reward,
            'episode_pnl': self.episode_pnl,
            'position': self.engine.position,
            'cash': self.engine.cash,
            'total_trades': self.engine.total_trades,
            'sharpe_ratio': 0.0
        }