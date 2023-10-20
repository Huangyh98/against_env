import gymnasium as gym
import highway_env

highway_env.register_highway_envs()

env = gym.make('highway-against-v0', render_mode="rgb_array")
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
env.render()