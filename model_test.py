import gymnasium as gym
from guessing_number_env import GuessNumEnv

env = GuessNumEnv(display=True)

episodes = 1

for episode in range(episodes):
    env.reset()
    done = False
    
    while True:
        action = int(input("Guess The Number: "))
        obs, reward, done, _, info = env.step(action)
        print(obs)
        print("------------------------")
        if done:
            break