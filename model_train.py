from stable_baselines3 import A2C
import os
import time
from guessing_number_env import GuessNumEnv

model_dir = f"models/A2C/{int(time.time())}"
logdir = f"logs/A2C/{int(time.time())}"

if not os.path.exists(model_dir):
    os.makedirs(model_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = GuessNumEnv()
env.reset()

model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100_000
for i in range(1, 10000000):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="A2C", progress_bar=True)
    model.save(f"{model_dir}/{TIMESTEPS*i}")


env.close()