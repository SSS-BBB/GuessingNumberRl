import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
from collections import deque

LAST_NUM = 10_000
LAST_GUESSES_LEN = 100
MAX_GUESSES = 15

class GuessNumEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 0}

    def __init__(self, display=False):
        super().__init__()

        self.display = display

        self.action_space = spaces.Discrete(LAST_NUM + 1)

        self.observation_space = spaces.Box(low=-2, high=LAST_NUM + 1,
                                            shape=(1 + 2*LAST_GUESSES_LEN, ), dtype=np.int64)

    def step(self, action):
        # action -> the number you guess
        self.last_guesses.append(action)
        self.total_guesses += 1

        if self.the_num < action:
            self.num_state = -1
        elif self.the_num > action:
            self.num_state = 1
        elif self.the_num == action:
            self.num_state = 0

        self.last_states.append(self.num_state)

        if (self.display):
            self.display_game()
        
        self.set_obs()
        self.set_reward(action)

        info = { 
                    "num_state": self.num_state,
                    "last_guesses": list(self.last_guesses),
                    "last_states": list(self.last_states)
               }

        return self.observation, self.reward, self.done, self.truncated, info

    def reset(self, seed=None, options=None):
        # -1 the number is lower than the number you guess
        # 0 the number is equal to the number you guess
        # 1 the number is greater than the number you guess
        self.num_state = -2
        self.the_num = random.randint(0, LAST_NUM)
        self.total_guesses = 0

        self.last_guesses = deque(maxlen=LAST_GUESSES_LEN)
        self.last_states = deque(maxlen=LAST_GUESSES_LEN)
        for _ in range(LAST_GUESSES_LEN):
            self.last_guesses.append(-1)
            self.last_states.append(-2)

        self.set_obs()
        info = {}

        return self.observation, info
    
    def set_obs(self):
        self.observation = np.array([self.num_state] + list(self.last_guesses) + list(self.last_states))

    def set_reward(self, guess_num):
        self.truncated = False
        if (self.num_state == 0):
            self.reward = 500
            self.done = True
        else:
            # self.reward = -abs(self.the_num - guess_num)
            if self.total_guesses <= MAX_GUESSES:
                self.reward = 1 / abs(self.the_num - guess_num)
                self.done = False
            else:
                self.reward = -100
                self.done = True
                self.truncated = True

        self.reward = float(self.reward)

    def display_game(self):
        if (self.num_state == -1):
            print("The number is lower than the number you guess")
        elif (self.num_state == 1):
            print("The number is greater than the number you guess")
        elif (self.num_state == 0):
            print("Congrats you got the right number!")