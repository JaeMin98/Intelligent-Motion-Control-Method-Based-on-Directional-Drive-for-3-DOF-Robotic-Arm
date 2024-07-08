import gym
import numpy as np
import torch
from ddpg_agent import DDPGAgent
from collections import deque
import random
import wandb
import config

class ReplayBuffer:
    def __init__(self, max_size=config.replay_size):
        self.buffer = deque(maxlen=int(max_size))

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(next_state),
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(done).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer)

def init_wandb():
    wandb.init(project='DDPG_TEST')
    wandb.run.name = 'Test'
    wandb.run.save()

def train():
    init_wandb()
    env = gym.make("Pendulum-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device(config.cuda if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = DDPGAgent(state_dim, action_dim, max_action, device)
    replay_buffer = ReplayBuffer()

    total_timesteps = 0
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    state, _ = env.reset()

    for t in range(config.num_steps):
        total_timesteps += 1
        episode_timesteps += 1

        # Select action
        if t < 10000:
            action = env.action_space.sample()
        else:
            action = agent.select_action(np.array(state))
            noise = np.random.normal(0, max_action * 0.1, size=action_dim)
            action = (action + noise).clip(-max_action, max_action)

        # Perform action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Store data in replay buffer
        replay_buffer.add(state, action, reward, next_state, done)

        state = next_state 
        episode_reward += reward

        # Train agent
        if len(replay_buffer) > 10000:
            agent.train(replay_buffer)

        if done:
            print(f"Total T: {total_timesteps}, Episode Num: {episode_num}, Episode T: {episode_timesteps}, Reward: {episode_reward:.3f}")
            wandb.log({"episode_reward": episode_reward})
            # Reset environment
            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

    # Save the trained model
    agent.save("ddpg_pendulum")

if __name__ == "__main__":
    train()