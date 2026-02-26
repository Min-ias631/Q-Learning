import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque, namedtuple
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        super(DQNetwork, self).__init__()

        self.feature_dim = state_dim - 3
        
        self.lob_encoder = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        combined_dim = (hidden_dim // 2) + 32
        self.value_stream = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim //2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
    
    def forward(self, x):
        lob_features = x[:, :self.feature_dim]
        agent_features = x[:, self.feature_dim:]

        lob_encoded = self.lob_encoder(lob_features)
        state_encoded = self.state_encoder(agent_features)

        combined = torch.cat([lob_encoded, state_encoded], dim = 1)

        value = self.value_stream(combined)
        advantages = self.advantage_stream(combined)

        q_values = value + (advantages - advantages.mean(dim = 1, keepdim = True))

        return q_values

class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-6

        self.buffer = []
        self.priorities = np.ones(capacity, dtype = np.float32)  # Initialize to 1.0, not 0
        self.position = 0

    def push(self, *args):
        """Add transition to buffer"""
        # Validate inputs
        state, action, reward, next_state, done = args
        
        # Check for NaN/inf
        if not np.isfinite(state).all():
            logger.warning(f"State contains NaN/inf, skipping: {state}")
            return
        if not np.isfinite(next_state).all():
            logger.warning(f"Next_state contains NaN/inf, skipping: {next_state}")
            return
        if not np.isfinite(reward):
            logger.warning(f"Reward is NaN/inf, skipping: {reward}")
            return
        
        max_priority = self.priorities.max() if self.buffer else 1.0
        
        # Ensure max_priority is valid
        if not np.isfinite(max_priority) or max_priority <= 0:
            max_priority = 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.position] = Transition(*args)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample batch with prioritized experience replay"""
        if len(self.buffer) < batch_size:
            raise ValueError(f"Not enough samples in buffer: {len(self.buffer)} < {batch_size}")
        
        # Get valid priorities
        if len(self.buffer) == self.capacity:
            priorities = self.priorities.copy()
        else:
            priorities = self.priorities[:len(self.buffer)].copy()
        
        # Safety check: ensure no NaN/inf and all positive
        if not np.isfinite(priorities).all():
            logger.warning("Priorities contain NaN/inf, resetting to 1.0")
            priorities = np.ones_like(priorities)
            if len(self.buffer) == self.capacity:
                self.priorities[:] = 1.0
            else:
                self.priorities[:len(self.buffer)] = 1.0
        
        # Ensure all priorities are positive
        priorities = np.maximum(priorities, self.epsilon)
        
        # Compute probabilities
        probabilities = priorities ** self.alpha
        prob_sum = probabilities.sum()
        
        # Safety check: ensure sum is valid
        if not np.isfinite(prob_sum) or prob_sum <= 0:
            logger.warning(f"Invalid probability sum: {prob_sum}, using uniform sampling")
            probabilities = np.ones_like(probabilities)
            prob_sum = probabilities.sum()
        
        probabilities /= prob_sum

        # Sample indices
        try:
            indices = np.random.choice(len(self.buffer), batch_size, p=probabilities, replace=False)
        except ValueError as e:
            logger.error(f"Error sampling: {e}")
            logger.error(f"Probabilities stats: min={probabilities.min()}, max={probabilities.max()}, sum={probabilities.sum()}")
            # Fallback to uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            probabilities = np.ones(len(self.buffer)) / len(self.buffer)

        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        self.beta = min(1.0, self.beta + self.beta_increment)
        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities with safety checks"""
        for idx, priority in zip(indices, priorities):
            # Ensure priority is valid
            if not np.isfinite(priority):
                priority = 1.0
            priority = max(priority, 0.0)  # Ensure non-negative
            
            self.priorities[idx] = priority + self.epsilon
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        learning_rate: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: int = 50_000,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        target_update_freq: int = 1000,
        gradient_clip: float = 10.0,
        device: Optional[str] = None
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.gradient_clip = gradient_clip

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f'Using device: {self.device}')

        self.policy_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr = learning_rate)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode = 'max', factor = 0.5, patience = 10
        )

        self.memory = ReplayBuffer(buffer_size)
        
        self.steps_done = 0
        self.episodes_done = 0
        self.losses = []
        self.q_values = []

        logger.info(f'DQN Agent initialized with {sum(p.numel() for p in self.policy_net.parameters())} parameters')

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        """Select action using epsilon-greedy policy"""
        # Validate state
        if not np.isfinite(state).all():
            logger.warning(f"State contains NaN/inf in select_action, taking random action")
            return random.randrange(self.action_dim)
        
        if explore and random.random() < self.get_epsilon():
            return random.randrange(self.action_dim)
        
        was_training = self.policy_net.training
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            
            # Check for NaN in Q-values
            if torch.isnan(q_values).any():
                logger.warning("Q-values contain NaN, taking random action")
                action = random.randrange(self.action_dim)
            else:
                action = q_values.argmax(1).item()
        
        if was_training:
            self.policy_net.train()
        return action

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return None
        
        try:
            transitions, indices, weights = self.memory.sample(self.batch_size)
        except Exception as e:
            logger.error(f"Error sampling from replay buffer: {e}")
            return None
        
        weights = torch.FloatTensor(weights).to(self.device)

        batch = Transition(*zip(*transitions))

        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(self.device)

        # Check for NaN in batches
        if torch.isnan(state_batch).any() or torch.isnan(next_state_batch).any():
            logger.warning("NaN detected in state batches, skipping training step")
            return None
        if torch.isnan(reward_batch).any():
            logger.warning("NaN detected in reward batch, skipping training step")
            return None

        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            next_actions = self.policy_net(next_state_batch).argmax(1, keepdim = True)
            next_q_values = self.target_net(next_state_batch).gather(1, next_actions)
            target_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Check for NaN in Q-values
        if torch.isnan(current_q_values).any() or torch.isnan(target_q_values).any():
            logger.warning("NaN detected in Q-values, skipping training step")
            return None
        
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        self.memory.update_priorities(indices, td_errors.flatten())

        loss = (weights * F.smooth_l1_loss(current_q_values, target_q_values, reduction = 'none')).mean()

        # Check for NaN loss
        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping backward pass")
            return None

        self.optimizer.zero_grad()
        loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for param in self.policy_net.parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan_grad = True
                break
        
        if has_nan_grad:
            logger.warning("NaN gradients detected, skipping optimizer step")
            return None

        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

        self.optimizer.step()

        loss_value = loss.item()
        self.losses.append(loss_value)
        
        self.steps_done += 1

        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            logger.info(f'Target network updated at step {self.steps_done}')
        
        return loss_value

    def save(self, filepath: str):
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done
        }
        torch.save(checkpoint, filepath)
        logger.info(f'Model saved to {filepath}')
    
    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.episodes_done = checkpoint.get('episodes_done', 0)
        logger.info(f"Model loaded from {filepath}")
    
    def get_statistics(self) -> dict:
        stats = {
            'epsilon': self.get_epsilon(),
            'steps_done': self.steps_done,
            'episodes_done': self.episodes_done,
            'memory_size': len(self.memory),
        }

        if len(self.losses) > 0:
            recent_losses = self.losses[-100:]
            stats['mean_loss'] = np.mean(recent_losses)
            stats['std_loss'] = np.std(recent_losses)
        
        if len(self.q_values) > 0:
            recent_q = self.q_values[-100:]
            stats['mean_q_value'] = np.mean(recent_q)
            stats['std_q_value'] = np.std(recent_q)
        
        return stats

    def get_epsilon(self) -> float:
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-1.0 * self.steps_done / self.epsilon_decay)
        return epsilon