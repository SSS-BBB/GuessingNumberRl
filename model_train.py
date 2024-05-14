from stable_baselines3 import PPO
import os
import time
from guessing_number_env import GuessNumEnv
from model_debug_callback import DebugCallback

model_dir = f"models/PPO/{int(time.time())}"
logdir = f"logs/PPO/{int(time.time())}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GuessNumEnv()
env.reset()

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
debug_callback = DebugCallback()

TIMESTEPS = 10_000
for i in range(1, 10000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", progress_bar=True, 
                callback=None)
    model.save(f"{model_dir}/{TIMESTEPS*i}")


env.close()