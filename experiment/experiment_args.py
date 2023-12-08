from dataclasses import dataclass


@dataclass
class ExperimentArgs:
    # ===== DQN Agent Configuration =====
    batch_size = 128
    """batch_size is the number of transitions sampled from the replay buffer"""
    gamma = 0.99
    """gamma is the discount factor as mentioned in the previous section"""
    epsilon_start = 1  # ! Original was 0.9
    """epsilon_start is the starting value of epsilon"""
    epsilon_end = 0.05
    """epsilon_end is the final value of epsilon"""
    epsilon_decay = 1000
    """epsilon_decay controls the rate of exponential decay of epsilon, higher means a slower decay"""
    tau = 0.005  # ! Original was 0.005
    """tau is the update rate of the target network"""
    learning_rate = 2.5e-4
    """learning_rate is the learning rate of the ``AdamW`` optimizer"""
    buffer_size: int = 10000
    """buffer_size is the replay memory buffer size"""

    # ===== Experiment Configuration =====
    num_episodes_without_cuda: int = 500
    """num_episodes is the number of episodes for which to run the environment when cuda is not available"""
    num_episodes_with_cuda: int = 500
    """num_episodes is the number of episodes for which to run the environment when cuda is available"""
