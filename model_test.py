import gymnasium as gym
from guessing_number_env import GuessNumEnv

env = GuessNumEnv(display=True)

episodes = 1
max_steps = 100

for episode in range(episodes):
    env.reset()
    done = False
    
    for _ in range(max_steps):
        action = int(input("Guess The Number: "))
        obs, reward, done, _, info = env.step(action)
        print("reward:", reward)
        print("info:", info)
        print("------------------------")
        if done:
            break