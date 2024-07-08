import numpy as np
import torch
import gym
from ddpg_agent import DDPGAgent

def evaluate():
    env = gym.make("Pendulum-v1", render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = DDPGAgent(state_dim, action_dim, max_action, device)
    
    # Load the trained model
    agent.load("ddpg_pendulum")

    for episode in range(10):  # Evaluate for 10 episodes
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(np.array(state))
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state

        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

    env.close()

if __name__ == "__main__":
    evaluate()