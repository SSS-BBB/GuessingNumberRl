from stable_baselines3.common.env_checker import check_env
from guessing_number_env import GuessNumEnv

env = GuessNumEnv()

check_env(env)