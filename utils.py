import math
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gymnasium as gym

#====================================================================================
# Lr Schedulers
#====================================================================================

class ConstantLR:
    """Constant learning rate scheduler."""
    
    def __init__(self, lr):
        self.lr = lr
    
    def __call__(self, progress):
        """Return constant learning rate."""
        return self.lr


class ExponentialLR:
    """Exponential decay learning rate scheduler."""
    
    def __init__(self, initial_lr, final_lr=1e-6):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
    
    def __call__(self, progress):
        """Exponentially decay lr from initial_lr to final_lr as progress goes from 1 to 0."""
        return self.final_lr + (self.initial_lr - self.final_lr) * math.exp(-5 * (1 - progress))


class CosineAnnealingLR:
    """Cosine annealing learning rate scheduler."""
    
    def __init__(self, initial_lr, final_lr=1e-6):
        self.initial_lr = initial_lr
        self.final_lr = final_lr
    
    def __call__(self, progress):
        """Cosine annealing: smoothly decay lr as progress goes from 1 to 0."""
        return self.final_lr + (self.initial_lr - self.final_lr) * 0.5 * (1 + math.cos(math.pi * (1 - progress)))
    

#====================================================================================
# Custom callback (plots rewards history, saves best model)
#====================================================================================

#custom callback to plot reward histroty
class RewardHistoryCallback(BaseCallback):
    """Callback to store episode rewards during training."""
    def __init__(self, model=None, save_path="best_model", track_lr=False):
        super().__init__()
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.model = model
        self.save_path = save_path
        self.best_avg_reward = -float('inf')
        self.window_size = 20
        self.track_lr = track_lr
        self.lrs = [] if track_lr else None
    
    def _on_step(self) -> bool:
        # Track learning rate if enabled
        if self.track_lr:
            self.lrs.append(self.model.policy.optimizer.param_groups[0]['lr'])
        
        # Access reward from the environment
        if self.locals.get("dones")[0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            
            # Check if we should save based on 20-episode average
            if len(self.episode_rewards) >= self.window_size:
                avg_reward = np.mean(self.episode_rewards[-self.window_size:])
                if avg_reward > self.best_avg_reward:
                    self.best_avg_reward = avg_reward
                    if self.model is not None:
                        self.model.save(self.save_path)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        else:
            self.current_episode_reward += self.locals.get("rewards")[0]
            self.current_episode_length += 1
        return True
    
    def plot(self):
        """Plot training reward history with optional learning rate tracking."""
        n_plots = 3 if self.track_lr else 2
        fig, axes = plt.subplots(1, n_plots, figsize=(14 if not self.track_lr else 18, 5))
        
        if not self.track_lr:
            axes = [axes[0], axes[1]]  # Ensure axes is always indexable consistently
        
        # Plot raw episode rewards
        axes[0].plot(self.episode_rewards, linewidth=0.5, alpha=0.7)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Episode Reward')
        axes[0].set_title('Training Episode Rewards')
        axes[0].grid(True, alpha=0.3)
        
        # Plot smoothed rewards (moving average)
        window = max(1, len(self.episode_rewards) // 50)
        smoothed = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        axes[1].plot(smoothed, linewidth=1.5)
        axes[1].set_xlabel('Episode (smoothed)')
        axes[1].set_ylabel('Moving Average Reward')
        axes[1].set_title(f'Training Rewards (moving avg, window={window})')
        axes[1].grid(True, alpha=0.3)
        
        # Plot learning rate if tracked
        if self.track_lr:
            axes[2].plot(self.lrs, linewidth=1)
            axes[2].set_xlabel('Step')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_title('Learning Rate Schedule')
            #axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print(f"Total episodes trained: {len(self.episode_rewards)}")
        print(f"Mean reward: {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}")
        print(f"Best reward: {np.max(self.episode_rewards):.2f}")
        print(f"Best {self.window_size}-episode avg: {self.best_avg_reward:.2f}")
        if self.track_lr:
            print(f"Initial LR: {self.lrs[0]:.6f}")
            print(f"Final LR: {self.lrs[-1]:.6f}")


class FloatActionWrapper(gym.ActionWrapper):
    """
    Workaround for SB3/PyTorch 'Float can't be cast to Char' error.
    Overrides the action space dtype to float32 for internal RL buffers.
    """
    def __init__(self, env):
        super().__init__(env)
        # Create a copy of the original space
        new_space = gym.spaces.MultiBinary(self.env.action_space.n)
        # Manually force the dtype attribute to float32
        new_space.dtype = np.dtype('float32')
        self.action_space = new_space

    def action(self, action):
        # Cast the model's float output back to the environment's original type (int8)
        return action.astype(self.env.action_space.dtype)