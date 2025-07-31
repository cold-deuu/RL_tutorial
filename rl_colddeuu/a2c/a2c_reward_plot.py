import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(reward_list, window=10, save_path=None):
    """
    Plot training rewards with moving average.

    Args:
        reward_list (list or np.ndarray): List of episode rewards.
        window (int): Moving average window size.
        save_path (str): Path to save the figure (optional).
    """
    reward_array = np.array(reward_list)
    moving_avg = np.convolve(reward_array, np.ones(window)/window, mode='valid')

    plt.figure(figsize=(10,6))
    plt.plot(reward_array, label='Episode Reward', color='skyblue', linewidth=1.5)
    plt.plot(np.arange(window - 1, len(reward_array)), moving_avg, label=f'{window}-Episode Moving Average', color='orange', linewidth=2)

    plt.title('Training Rewards over Episodes', fontsize=16)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Reward', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"Saved reward plot to {save_path}")
    
    plt.show()

rewards = np.loadtxt('./weights/pendulum_epi_reward.txt')
plot_rewards(rewards, window=10)
