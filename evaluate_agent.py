#!/usr/bin/env python3
"""
Comprehensive Model Evaluation Script
Tests trained DQN agent and generates performance metrics
"""

import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from HFT_env import HFT_env
from dqn_agent import DQNAgent
from features import load_preset

class ModelEvaluator:
    def __init__(self, checkpoint_path, config):
        """
        Args:
            checkpoint_path: Path to saved model (.pth file)
            config: Dictionary with environment configuration
        """
        self.checkpoint_path = checkpoint_path
        self.config = config
        
        # Create unique results directory for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path('./evaluation_results') / f'run_{timestamp}'
        self.results_dir.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {self.results_dir}")
        
        # Create environment
        load_preset(config['feature_preset'])
        self.env = HFT_env(
            depth_data_path=config['depth_data_path'],
            trade_data_path=config['trade_data_path'],
            normalization_stats_path=config.get('normalization_stats_path'),
            transaction_cost_bps=config['transaction_cost_bps'],
            max_position=config['max_position'],
            initial_cash=config['initial_cash'],
            reward_scaling=config['reward_scaling'],
            decision_interval_ms=config['decision_interval_ms'],
        )
        
        # Create and load agent
        self.agent = DQNAgent(
            state_dim=self.env.observation_space.shape[0],
            action_dim=self.env.action_space.n,
            hidden_dim=128
        )
        self.agent.load(checkpoint_path)
        print(f"Loaded model from: {checkpoint_path}")
    
    def run_episode(self, max_steps=None, deterministic=True, episode_num=0):
        """Run single evaluation episode"""
        obs, info = self.env.reset()
        
        episode_data = {
            'rewards': [],
            'pnls': [],
            'positions': [],
            'trades': [],
            'actions': [],
            'timestamps': [],
            'cash': [],
        }
        
        # Track non-zero actions for detailed logging
        non_zero_actions = []
        
        done = False
        steps = 0
        
        while not done:
            if max_steps and steps >= max_steps:
                break
            
            # Select action (no exploration)
            action = self.agent.select_action(obs, explore=not deterministic)
            
            # Step
            next_obs, reward, done, info = self.env.step(action)
            
            # Record
            episode_data['rewards'].append(reward)
            episode_data['pnls'].append(info['total_pnl'])
            episode_data['positions'].append(info['position'])
            episode_data['trades'].append(info['total_trades'])
            episode_data['actions'].append(action)
            episode_data['timestamps'].append(info['timestamp'])
            episode_data['cash'].append(info['cash'])
            
            # Track non-zero actions (not NoOp)
            if action != 0:
                non_zero_actions.append({
                    'episode': episode_num,
                    'step': steps,
                    'timestamp': info['timestamp'],
                    'action': action,
                    'position_before': info['position'],
                    'pnl': info['total_pnl'],
                    'total_trades': info['total_trades'],
                })
            
            obs = next_obs
            steps += 1
        
        return episode_data, info, non_zero_actions
    
    def evaluate(self, num_episodes=20, max_steps_per_episode=None):
        """
        Evaluate agent over multiple episodes
        
        Returns:
            Dictionary with comprehensive metrics
        """
        print(f"\nEvaluating model over {num_episodes} episodes...")
        
        all_episodes = []
        all_non_zero_actions = []
        
        for ep in tqdm(range(num_episodes), desc="Evaluating"):
            episode_data, final_info, non_zero_actions = self.run_episode(
                max_steps_per_episode, 
                episode_num=ep
            )
            
            # Collect non-zero actions
            all_non_zero_actions.extend(non_zero_actions)
            
            # Compute episode metrics
            episode_metrics = {
                'episode': ep,
                'final_pnl': final_info['total_pnl'],
                'total_reward': sum(episode_data['rewards']),
                'total_trades': final_info['total_trades'],
                'steps': len(episode_data['rewards']),
                'final_position': final_info['position'],
                'final_cash': final_info['cash'],
                'max_position': max(np.abs(episode_data['positions'])),
                'pnl_data': episode_data['pnls'],
                'reward_data': episode_data['rewards'],
                'action_data': episode_data['actions'],
            }
            
            all_episodes.append(episode_metrics)
        
        # Aggregate metrics
        results = self._compute_aggregate_metrics(all_episodes)
        
        return results, all_episodes, all_non_zero_actions
    
    def _compute_aggregate_metrics(self, episodes):
        """Compute aggregate statistics across episodes"""
        pnls = [ep['final_pnl'] for ep in episodes]
        rewards = [ep['total_reward'] for ep in episodes]
        trades = [ep['total_trades'] for ep in episodes]
        steps = [ep['steps'] for ep in episodes]
        
        # Basic stats
        metrics = {
            'num_episodes': len(episodes),
            'mean_pnl': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'median_pnl': np.median(pnls),
            'min_pnl': np.min(pnls),
            'max_pnl': np.max(pnls),
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_trades': np.mean(trades),
            'mean_steps': np.mean(steps),
        }
        
        # Win rate
        winning_episodes = sum(1 for pnl in pnls if pnl > 0)
        metrics['win_rate'] = winning_episodes / len(pnls)
        
        # Sharpe ratio (annualized)
        if np.std(pnls) > 0:
            # Assuming ~250 trading days, episodes are ~1 day each
            metrics['sharpe_ratio'] = (np.mean(pnls) / np.std(pnls)) * np.sqrt(250)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Max drawdown
        cumulative_pnl = np.cumsum(pnls)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        metrics['max_drawdown'] = np.max(drawdown)
        
        # Profit factor (gross profit / gross loss)
        gross_profit = sum(pnl for pnl in pnls if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnls if pnl < 0))
        if gross_loss > 0:
            metrics['profit_factor'] = gross_profit / gross_loss
        else:
            metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
        
        # PnL per trade
        total_trades = sum(trades)
        if total_trades > 0:
            metrics['pnl_per_trade'] = sum(pnls) / total_trades
        else:
            metrics['pnl_per_trade'] = 0.0
        
        # Action distribution
        all_actions = []
        for ep in episodes:
            all_actions.extend(ep['action_data'])
        
        action_counts = np.bincount(all_actions, minlength=15)
        metrics['action_distribution'] = action_counts.tolist()
        metrics['action_frequencies'] = (action_counts / len(all_actions)).tolist()
        
        return metrics
    
    def plot_results(self, results, episodes):
        """Generate comprehensive plots"""
        
        fig = plt.figure(figsize=(20, 12))
        
        # 1. PnL distribution
        ax1 = plt.subplot(3, 3, 1)
        pnls = [ep['final_pnl'] for ep in episodes]
        ax1.hist(pnls, bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(pnls), color='r', linestyle='--', label=f'Mean: {np.mean(pnls):.6f}')
        ax1.set_xlabel('Episode PnL')
        ax1.set_ylabel('Frequency')
        ax1.set_title('PnL Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative PnL
        ax2 = plt.subplot(3, 3, 2)
        cumulative_pnl = np.cumsum(pnls)
        ax2.plot(cumulative_pnl, linewidth=2)
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative PnL')
        ax2.set_title('Cumulative PnL Over Episodes')
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax3 = plt.subplot(3, 3, 3)
        running_max = np.maximum.accumulate(cumulative_pnl)
        drawdown = running_max - cumulative_pnl
        ax3.fill_between(range(len(drawdown)), 0, drawdown, alpha=0.3, color='red')
        ax3.plot(drawdown, color='red', linewidth=1)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Drawdown')
        ax3.set_title(f'Drawdown (Max: {results["max_drawdown"]:.6f})')
        ax3.grid(True, alpha=0.3)
        
        # 4. Trades per episode
        ax4 = plt.subplot(3, 3, 4)
        trades = [ep['total_trades'] for ep in episodes]
        ax4.plot(trades, marker='o', linewidth=1, markersize=3)
        ax4.axhline(np.mean(trades), color='r', linestyle='--', label=f'Mean: {np.mean(trades):.1f}')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Number of Trades')
        ax4.set_title('Trades per Episode')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Action distribution
        ax5 = plt.subplot(3, 3, 5)
        action_labels = [
            'NoOp', 'MktBuy', 'MktSell',
            'LimBuy1', 'LimBuy2', 'LimBuy3', 'LimBuy4', 'LimBuy5',
            'LimSell1', 'LimSell2', 'LimSell3', 'LimSell4', 'LimSell5',
            'PostBoth', 'CancelAll'
        ]
        action_freq = results['action_frequencies']
        bars = ax5.bar(range(len(action_freq)), action_freq, alpha=0.7, edgecolor='black')
        ax5.set_xlabel('Action')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Action Distribution')
        ax5.set_xticks(range(len(action_labels)))
        ax5.set_xticklabels(action_labels, rotation=45, ha='right', fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Sample episode PnL trajectory
        ax6 = plt.subplot(3, 3, 6)
        sample_ep = episodes[0]  # First episode
        ax6.plot(sample_ep['pnl_data'], linewidth=2)
        ax6.set_xlabel('Step')
        ax6.set_ylabel('PnL')
        ax6.set_title(f'Sample Episode PnL Trajectory (Ep 0)')
        ax6.grid(True, alpha=0.3)
        
        # 7. Win rate by episode
        ax7 = plt.subplot(3, 3, 7)
        wins = [1 if ep['final_pnl'] > 0 else 0 for ep in episodes]
        cumulative_win_rate = np.cumsum(wins) / np.arange(1, len(wins) + 1)
        ax7.plot(cumulative_win_rate, linewidth=2)
        ax7.axhline(0.5, color='gray', linestyle='--', label='50%')
        ax7.axhline(results['win_rate'], color='r', linestyle='--', label=f'Final: {results["win_rate"]:.2%}')
        ax7.set_xlabel('Episode')
        ax7.set_ylabel('Cumulative Win Rate')
        ax7.set_title('Win Rate Over Time')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. PnL vs Trades scatter
        ax8 = plt.subplot(3, 3, 8)
        ax8.scatter(trades, pnls, alpha=0.6, s=50)
        ax8.set_xlabel('Number of Trades')
        ax8.set_ylabel('Episode PnL')
        ax8.set_title('PnL vs Number of Trades')
        ax8.grid(True, alpha=0.3)
        
        # Add trend line
        if len(trades) > 1:
            z = np.polyfit(trades, pnls, 1)
            p = np.poly1d(z)
            ax8.plot(sorted(trades), p(sorted(trades)), "r--", alpha=0.8, linewidth=2)
        
        # 9. Summary statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        summary_text = f"""
PERFORMANCE SUMMARY
{'='*40}

Mean PnL:        {results['mean_pnl']:+.6f}
Median PnL:      {results['median_pnl']:+.6f}
Std PnL:         {results['std_pnl']:.6f}
Min/Max:         {results['min_pnl']:+.6f} / {results['max_pnl']:+.6f}

Win Rate:        {results['win_rate']:.2%}
Sharpe Ratio:    {results['sharpe_ratio']:.3f}
Profit Factor:   {results['profit_factor']:.3f}
Max Drawdown:    {results['max_drawdown']:.6f}

Mean Trades:     {results['mean_trades']:.1f}
PnL per Trade:   {results['pnl_per_trade']:.6f}

Episodes:        {results['num_episodes']}
        """
        
        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        
        # Save to results directory
        plot_path = self.results_dir / 'evaluation_plots.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlots saved to: {plot_path}")
        
        plt.close()
    
    def save_results(self, results, episodes, non_zero_actions):
        """Save results to JSON and CSV"""
        
        # Save summary metrics
        results_path = self.results_dir / 'eval_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {results_path}")
        
        # Save episode details to CSV
        episode_df = pd.DataFrame([{
            'episode': ep['episode'],
            'final_pnl': ep['final_pnl'],
            'total_reward': ep['total_reward'],
            'total_trades': ep['total_trades'],
            'steps': ep['steps'],
            'final_position': ep['final_position'],
            'max_position': ep['max_position'],
        } for ep in episodes])
        
        csv_path = self.results_dir / 'eval_episodes.csv'
        episode_df.to_csv(csv_path, index=False)
        print(f"Episode details saved to: {csv_path}")
        
        # Save non-zero actions to CSV
        if non_zero_actions:
            action_labels = [
                'NoOp', 'MktBuy', 'MktSell',
                'LimBuy1', 'LimBuy2', 'LimBuy3', 'LimBuy4', 'LimBuy5',
                'LimSell1', 'LimSell2', 'LimSell3', 'LimSell4', 'LimSell5',
                'PostBoth', 'CancelAll'
            ]
            
            actions_df = pd.DataFrame([{
                'episode': a['episode'],
                'step': a['step'],
                'timestamp': a['timestamp'],
                'action_id': a['action'],
                'action_name': action_labels[a['action']],
                'position_before': a['position_before'],
                'pnl': a['pnl'],
                'total_trades': a['total_trades'],
            } for a in non_zero_actions])
            
            actions_path = self.results_dir / 'non_zero_actions.csv'
            actions_df.to_csv(actions_path, index=False)
            print(f"Non-zero actions saved to: {actions_path} ({len(non_zero_actions)} actions)")
        else:
            print("No non-zero actions to save (agent only used NoOp)")
        
        # Save config used
        config_path = self.results_dir / 'eval_config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Config saved to: {config_path}")
    
    def print_summary(self, results):
        """Print formatted summary to console"""
        print("\n" + "="*80)
        print("  EVALUATION SUMMARY")
        print("="*80)
        
        print(f"\nPERFORMANCE METRICS:")
        print(f"  Mean PnL:         {results['mean_pnl']:+.6f} ± {results['std_pnl']:.6f}")
        print(f"  Median PnL:       {results['median_pnl']:+.6f}")
        print(f"  Min/Max PnL:      {results['min_pnl']:+.6f} / {results['max_pnl']:+.6f}")
        print(f"  Win Rate:         {results['win_rate']:.2%} ({int(results['win_rate'] * results['num_episodes'])}/{results['num_episodes']} episodes)")
        
        print(f"\nRISK METRICS:")
        print(f"  Sharpe Ratio:     {results['sharpe_ratio']:.3f}")
        print(f"  Max Drawdown:     {results['max_drawdown']:.6f}")
        print(f"  Profit Factor:    {results['profit_factor']:.3f}")
        
        print(f"\nTRADING METRICS:")
        print(f"  Mean Trades:      {results['mean_trades']:.1f}")
        print(f"  PnL per Trade:    {results['pnl_per_trade']:.6f}")
        
        print(f"\nTOP 3 ACTIONS:")
        action_labels = [
            'NoOp', 'MktBuy', 'MktSell',
            'LimBuy1', 'LimBuy2', 'LimBuy3', 'LimBuy4', 'LimBuy5',
            'LimSell1', 'LimSell2', 'LimSell3', 'LimSell4', 'LimSell5',
            'PostBoth', 'CancelAll'
        ]
        action_freq = results['action_frequencies']
        top_actions = sorted(enumerate(action_freq), key=lambda x: x[1], reverse=True)[:3]
        for idx, freq in top_actions:
            print(f"  {action_labels[idx]:12s}: {freq:.2%}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation function"""
    
    # Configuration (match your training config)
    INSTRUMENT = 'ETHBTC'
    
    config = {
        'depth_data_path': f'data/{INSTRUMENT}-depth5-val.npy',
        'trade_data_path': f'data/{INSTRUMENT}-trades-val.npy',
        'normalization_stats_path': 'checkpoints/normalization_stats.npz',
        'transaction_cost_bps': 5.0,
        'max_position': 3.0,
        'initial_cash': 5.0,
        'reward_scaling': 500.0,
        'decision_interval_ms': 10000,
        'feature_preset': 'live',
    }
    
    # Which model to evaluate
    checkpoint_path = './checkpoints/best_model.pth'  # or 'final_model.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        for p in Path('./checkpoints').glob('*.pth'):
            print(f"  {p}")
        return
    
    # Create evaluator
    evaluator = ModelEvaluator(checkpoint_path, config)
    
    # Run evaluation
    results, episodes, non_zero_actions = evaluator.evaluate(
        num_episodes=50,  # Test on 50 episodes
        max_steps_per_episode=None  # Full episodes
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Generate plots
    evaluator.plot_results(results, episodes)
    
    # Save results
    evaluator.save_results(results, episodes, non_zero_actions)
    
    print(f"\n✓ Evaluation complete!")
    print(f"✓ All results saved to: {evaluator.results_dir}")


if __name__ == "__main__":
    main()