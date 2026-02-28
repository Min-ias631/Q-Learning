import numpy as np
import torch
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from features import load_preset, get_enabled_features

from HFT_env import HFT_env
from dqn_agent import DQNAgent

logging.basicConfig(
    level = logging.INFO,
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingLogger:
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents = True, exist_ok = True)

        self.episode_rewards = []
        self.episode_pnls = []
        self.episode_lengths = []
        self.episode_trades = []
        self.losses = []
        self.epsilons = []
        self.eval_rewards = []
        self.eval_pnls = []

        logger.info(f'Training logs will be saved to {self.log_dir}')
    
    def log_episode(self, episode: int, reward: float, pnl: float, length: int, trades: int):
        self.episode_rewards.append(reward)
        self.episode_pnls.append(pnl)
        self.episode_lengths.append(length)
        self.episode_trades.append(trades)

    def log_training(self, loss: float, epsilon: float):
        if loss is not None:
            self.losses.append(loss)
        self.epsilons.append(epsilon)
    
    def log_evaluation(self, reward: float, pnl: float):
        self.eval_rewards.append(reward)
        self.eval_pnls.append(pnl)

    def save_metrics(self):
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_pnls': self.episode_pnls,
            'episode_lengths': self.episode_lengths,
            'episode_trades': self.episode_trades,
            'losses': self.losses,
            'epsilons': self.epsilons,
            'eval_rewards': self.eval_rewards,
            'eval_pnls': self.eval_pnls,
        }
        
        filepath = self.log_dir / 'training_metrics.json'
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to {filepath}")
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize = (18, 10))

        if len(self.episode_rewards) > 0:
            axes[0, 0].plot(self.episode_rewards, alpha = 0.3, label = 'Raw')
            if len(self.episode_rewards) >= 10:
                smoothed = np.convolve(self.episode_rewards, np.ones(10) / 10, mode = 'valid')
                axes[0, 0].plot(smoothed, label = 'Smoothed (10 ep)')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        if len(self.episode_pnls) > 0:
            axes[0, 1].plot(self.episode_pnls, alpha=0.3, label='Raw')
            if len(self.episode_pnls) >= 10:
                smoothed = np.convolve(self.episode_pnls, np.ones(10)/10, mode='valid')
                axes[0, 1].plot(smoothed, label='Smoothed (10 ep)')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('PnL')
            axes[0, 1].set_title('Episode PnL')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        if len(self.losses) > 0:
            axes[0, 2].plot(self.losses, alpha=0.5)
            if len(self.losses) >= 100:
                smoothed = np.convolve(self.losses, np.ones(100)/100, mode='valid')
                axes[0, 2].plot(smoothed, label='Smoothed (100 steps)')
            axes[0, 2].set_xlabel('Training Step')
            axes[0, 2].set_ylabel('Loss')
            axes[0, 2].set_title('Training Loss')
            axes[0, 2].set_yscale('log')
            axes[0, 2].grid(True)
        
        if len(self.epsilons) > 0:
            axes[1, 0].plot(self.epsilons)
            axes[1, 0].set_xlabel('Training Step')
            axes[1, 0].set_ylabel('Epsilon')
            axes[1, 0].set_title('Exploration Rate')
            axes[1, 0].grid(True)

        if len(self.episode_lengths) > 0:
            axes[1, 1].plot(self.episode_lengths, alpha=0.3)
            if len(self.episode_lengths) >= 10:
                smoothed = np.convolve(self.episode_lengths, np.ones(10)/10, mode='valid')
                axes[1, 1].plot(smoothed, label='Smoothed (10 ep)')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Steps')
            axes[1, 1].set_title('Episode Length')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        if len(self.eval_rewards) > 0 and len(self.eval_pnls) > 0:
            ax1 = axes[1, 2]
            ax2 = ax1.twinx()
            
            ax1.plot(self.eval_rewards, 'b-', label='Reward')
            ax2.plot(self.eval_pnls, 'r-', label='PnL')
            
            ax1.set_xlabel('Evaluation')
            ax1.set_ylabel('Reward', color='b')
            ax2.set_ylabel('PnL', color='r')
            ax1.set_title('Evaluation Metrics')
            ax1.grid(True)
            ax1.legend(loc='upper left')
            ax2.legend(loc='upper right')
        
        plt.tight_layout()
        filepath = self.log_dir / 'training_curves.png'
        plt.savefig(filepath, dpi=150)
        plt.close()
        
        logger.info(f"Training curves saved to {filepath}")

def evaluate_agent(agent, env, num_episodes=10, max_steps=5000, render=False):
    episode_rewards, episode_pnls, episode_lengths, episode_trades = [], [], [], []

    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0.0
        done = False
        steps = 0

        while not done and steps < max_steps:
            action = agent.select_action(state, explore=False)
            next_state, reward, terminated, info = env.step(action)
            done = terminated

            episode_reward += reward
            state = next_state
            steps += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_pnls.append(info["total_pnl"])  # Use total_pnl
        episode_lengths.append(steps)
        episode_trades.append(info["total_trades"])

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_pnl": float(np.mean(episode_pnls)),
        "std_pnl": float(np.std(episode_pnls)),
        "mean_length": float(np.mean(episode_lengths)),
        "mean_trades": float(np.mean(episode_trades)),
        "sharpe_ratio": float(np.mean(episode_pnls) / (np.std(episode_pnls) + 1e-8)),
    }


def calculate_optimal_episodes(depth_data_path: str, target_episodes: int = None) -> dict:
    depth_data = np.load(depth_data_path)
    total_rows = len(depth_data)
    
    if target_episodes is None:
        if total_rows < 100_000:
            target_episodes = 50
        elif total_rows < 500_000:
            target_episodes = 100
        elif total_rows < 1_000_000:
            target_episodes = 150
        else:
            target_episodes = 200
    
    steps_per_episode = int(np.ceil(total_rows / target_episodes))
    max_steps_per_episode = int(steps_per_episode * 1.2)
    
    eval_freq = max(1, target_episodes // 10)
    save_freq = max(1, target_episodes // 5)
    log_freq = max(1, target_episodes // 20)
    
    logger.info(f"=" * 80)
    logger.info(f"DATASET ANALYSIS")
    logger.info(f"=" * 80)
    logger.info(f"Total data rows: {total_rows:,}")
    logger.info(f"Target episodes: {target_episodes}")
    logger.info(f"Steps per episode: {steps_per_episode:,}")
    logger.info(f"Max steps per episode (with buffer): {max_steps_per_episode:,}")
    logger.info(f"Expected data coverage: ~100%")
    logger.info(f"=" * 80)
    
    return {
        'num_episodes': target_episodes,
        'max_steps_per_episode': max_steps_per_episode,
        'eval_freq': eval_freq,
        'save_freq': save_freq,
        'log_freq': log_freq,
    }


def train_dqn(
    depth_data_path: str,
    trade_data_path: str,
    num_episodes: Optional[int] = None,
    max_steps_per_episode: Optional[int] = None,
    eval_freq: Optional[int] = None,
    save_freq: Optional[int] = None,
    log_freq: Optional[int] = None,
    eval_episodes: int = 5,
    auto_calculate_episodes: bool = True,
    
    # Environment parameters
    transaction_cost_bps: float = 5.0,
    decision_interval_ms: int = 10000,
    feature_preset: str = 'live',
    initial_cash: float = 5.0,
    max_position: float = 3.0,
    reward_scaling: float = 1000.0,
    reward_clip: float = 10.0,
    inventory_penalty_factor: float = 0.005,
    blocked_action_penalty: float = 0.05,

    # Agent parameters - FIXED HYPERPARAMETERS
    hidden_dim: int = 128,
    learning_rate: float = 5e-5,  # REDUCED from 1e-3
    gamma: float = 0.90,  # REDUCED from 0.95
    batch_size: int = 32,
    buffer_size: int = 100_000,
    epsilon_decay: int = 50_000,  # REDUCED from 1_000_000
    target_update_freq: int = 500,  # REDUCED from 1_000
    train_freq: int = 2,
    warmup_steps: int = 1000,  # REDUCED from 5000
    
    # Logging
    checkpoint_dir: str = './checkpoints',
    log_dir: str = './logs',
    device: Optional[str] = None,
):
    if auto_calculate_episodes or num_episodes is None or max_steps_per_episode is None:
        optimal_config = calculate_optimal_episodes(depth_data_path, num_episodes)
        
        if num_episodes is None:
            num_episodes = optimal_config['num_episodes']
        if max_steps_per_episode is None:
            max_steps_per_episode = optimal_config['max_steps_per_episode']
        if eval_freq is None:
            eval_freq = optimal_config['eval_freq']
        if save_freq is None:
            save_freq = optimal_config['save_freq']
        if log_freq is None:
            log_freq = optimal_config['log_freq']
    
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents = True, exist_ok = True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_log_dir = Path(log_dir) / f'run_{timestamp}'

    training_logger = TrainingLogger(run_log_dir)

    logger.info(f'Loading feature preset: {feature_preset}')
    load_preset(feature_preset)
    feature_names = get_enabled_features()
    logger.info(f'Using features: {feature_names}')

    logger.info('Creating environment...')
    logger.info(f"Transaction cost: {transaction_cost_bps} bps")
    logger.info(f"Decision interval: {decision_interval_ms} ms")
    logger.info(f"Reward scaling: {reward_scaling}")
    logger.info(f"Reward clip: ±{reward_clip}")

    env = HFT_env(
        depth_data_path=depth_data_path,
        trade_data_path=trade_data_path,
        transaction_cost_bps=transaction_cost_bps,
        max_position=max_position,
        initial_cash=initial_cash,
        reward_scaling=reward_scaling,
        order_expire_steps=100,
        decision_interval_ms=decision_interval_ms,
        reward_clip=reward_clip,
        inventory_penalty_factor=inventory_penalty_factor,
        blocked_action_penalty=blocked_action_penalty,
    )

    logger.info(f"Observation space shape: {env.observation_space.shape}")
    logger.info(f"Action space size: {env.action_space.n}")
    
    state_dim = env.observation_space.shape[0]
    
    logger.info("Creating agent...")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Gamma: {gamma}")
    logger.info(f"  Epsilon decay: {epsilon_decay}")
    logger.info(f"  Target update freq: {target_update_freq}")
    
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=env.action_space.n,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        epsilon_decay=epsilon_decay,
        target_update_freq=target_update_freq,
        device=device
    )

    logger.info(f'Starting training for {num_episodes} episodes...')

    best_eval_reward = -np.inf
    global_step = 0

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        with tqdm(total = max_steps_per_episode, desc = f'Episode {episode}/{num_episodes}') as pbar:
            while not done and episode_steps < max_steps_per_episode:
                if global_step < warmup_steps:
                    action = env.action_space.sample()
                else:
                    action = agent.select_action(state, explore = True)
                
                next_state, reward, terminated, info = env.step(action)
                done = terminated

                agent.store_transition(state, action, reward, next_state, done)

                if global_step >= warmup_steps and global_step % train_freq == 0:
                    loss = agent.train_step()
                    training_logger.log_training(loss, agent.get_epsilon())
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                global_step += 1

                pbar.update(1)
                pbar.set_postfix({
                    'reward': f'{episode_reward:.6f}',
                    'pnl': f'{info["total_pnl"]:.8f}',
                    'eps': f'{agent.get_epsilon():.3f}'
                })
            
        training_logger.log_episode(
            episode,
            episode_reward,
            info['total_pnl'],
            episode_steps,
            info['total_trades']
        )
        agent.episodes_done += 1

        if episode % log_freq == 0:
            env_stats = env.get_episode_stats()
            agent_stats = agent.get_statistics()
            
            logger.info(
                f"\nEpisode {episode}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"PnL: {info['total_pnl']:.4f} | "
                f"Trades: {info['total_trades']} | "
                f"Epsilon: {agent_stats['epsilon']:.3f} | "
                f"Loss: {agent_stats.get('mean_loss', 0):.4f}"
            )

        if episode % eval_freq == 0:
            logger.info(f"\n{'='*80}\nEvaluating agent at episode {episode}...\n{'='*80}")
            eval_metrics = evaluate_agent(agent, env, num_episodes=eval_episodes, max_steps=max_steps_per_episode)
            
            logger.info(
                f"Evaluation Results:\n"
                f"  Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}\n"
                f"  Mean PnL: {eval_metrics['mean_pnl']:.4f} ± {eval_metrics['std_pnl']:.4f}\n"
                f"  Mean Length: {eval_metrics['mean_length']:.0f}\n"
                f"  Mean Trades: {eval_metrics['mean_trades']:.0f}\n"
                f"  Sharpe Ratio: {eval_metrics['sharpe_ratio']:.3f}"
            )

            training_logger.log_evaluation(eval_metrics['mean_reward'], eval_metrics['mean_pnl'])

            agent.scheduler.step(eval_metrics['mean_reward'])

            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                best_model_path = checkpoint_dir / 'best_model.pth'
                agent.save(str(best_model_path))
                logger.info(f"New best model saved! Reward: {best_eval_reward:.2f}")
            
        if episode % save_freq == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_ep{episode}.pth'
            agent.save(str(checkpoint_path))
            logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            training_logger.save_metrics()
            training_logger.plot_training_curves()

    logger.info("\n" + "="*80)
    logger.info("Training completed!")
    logger.info("="*80)
    
    final_model_path = checkpoint_dir / 'final_model.pth'
    agent.save(str(final_model_path))
    training_logger.save_metrics()
    training_logger.plot_training_curves()
    
    logger.info("\nRunning final evaluation...")
    final_eval_metrics = evaluate_agent(agent, env, num_episodes=eval_episodes, max_steps=max_steps_per_episode)

    logger.info(
        f"\nFinal Evaluation Results:\n"
        f"  Mean Reward: {final_eval_metrics['mean_reward']:.2f} ± {final_eval_metrics['std_reward']:.2f}\n"
        f"  Mean PnL: {final_eval_metrics['mean_pnl']:.4f} ± {final_eval_metrics['std_pnl']:.4f}\n"
        f"  Sharpe Ratio: {final_eval_metrics['sharpe_ratio']:.3f}"
    )
    
    env.close()
    
    return agent, training_logger

if __name__ == '__main__':
    INSTRUMENT = 'ETHBTC'
    
    config = {  
        'depth_data_path': f'data/{INSTRUMENT}-depth5-train.npy',
        'trade_data_path': f'data/{INSTRUMENT}-trades-train.npy',
        'auto_calculate_episodes': True,
        'num_episodes': None,
        'max_steps_per_episode': None,
        'eval_freq': None,
        'save_freq': None,
        'log_freq': None,
        
        'transaction_cost_bps': 0.0,
        'decision_interval_ms': 10000,
        'feature_preset': 'live',
        'initial_cash': 1000.0,
        'max_position': 500.0,
        'reward_scaling': 5000.0,
        'reward_clip': 10.0,
        'inventory_penalty_factor': 0.005,
        'blocked_action_penalty': 0.05,
        
        'hidden_dim': 128,
        'learning_rate': 5e-5,
        'gamma': 0.90,
        'batch_size': 32,
        'buffer_size': 100_000,
        'epsilon_decay': 500_000,
        'target_update_freq': 10_000,
        'train_freq': 1,
        'warmup_steps': 5_000,
    }

    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

    agent, training_log = train_dqn(**config)
    
    print("\nTraining complete!")
    print(f"Logs saved to: {training_log.log_dir}")
    print("Best model: ./checkpoints/best_model.pth")
    print("Final model: ./checkpoints/final_model.pth")