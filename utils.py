import math
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import gymnasium as gym
from tqdm import tqdm



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


class ZeroRewardWrapper(gym.Wrapper):
    """Replace zero rewards with a specified value."""
    def __init__(self, env, zero_reward_replacement=-0.2):
        super().__init__(env)
        self.zero_reward_replacement = zero_reward_replacement
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward == 0:
            reward = self.zero_reward_replacement
        return obs, reward, terminated, truncated, info


#====================================================================================
# Evaluation
#====================================================================================

def evaluate_model(model, env, num_episodes, verbose=False):
    """Evaluate a trained model on an environment.
    
    Args:
        model_path: Path to saved model (without .zip extension)
        env: Gym environment
        num_episodes: Number of episodes to evaluate
        verbose: If True, print step-by-step info about chest interactions
    
    Returns:
        dict with keys: mean_reward, std_reward, min_reward, max_reward, 
                       mean_length, std_length, success_rate
    """



    
    episode_rewards = []
    episode_lengths = []
    successes = 0
    
    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        step = 0
        
        # For recurrent policies
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        
        while not done:
            past_obs = obs
            action, _ = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
            
            if verbose:
                # Extract target chest from info if available
                target_chest = action
                success = "success" if reward > 0 else "failure"
                print(f"  Episode {episode + 1}, Step {step}:\n  Observation : {past_obs}\n  tried chest {target_chest}: {success}")
            
            step += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if episode_reward > 0:
            successes += 1
    
    env.close()
    
    stats = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'success_rate': successes / num_episodes
    }
    
    return stats