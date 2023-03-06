import torch
import os
import stable_baselines3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import env_checker
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

import pyautogui
# import pydirectinput
import pytesseract
from mss import mss
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
import gym
from gym import Env
from gym.spaces import Box,Discrete

class webgame(Env):
    def __init__(self):
        super().__init__()
        # Setup spaces
        self.observation_space=Box(low=0,high=255,shape=(1,83,100),dtype=np.uint8)
        # self.observation_space = Box(low=0, high = 255,shape(1, 83, 100), dtype - np.uint8)
        self.action_space = Discrete(3)

        self.cap=mss()
        self.game_location={'top':300,'left':0,'width':600,'height':500}
        self.done_location={'top':405,'left':630,'width':660,'height':70}

        pass


    def step(self, action):

        action_map={
            0:'up',
            1:'down',
            2:'no_op'
        }
        if action !=2:
            pyautogui.press(action_map[action])

        done,done_cap=self.get_done()
        new_observation=self.get_observation()
        reward=10
        info={}
        return new_observation,reward,done,info



    def render(self):
        # plt.imshow('Game',np.array(self.cap.grab(self.game_location)))[:,:,:3]
        cv2.imshow('Game',np.array(self.cap.grab(self.game_location))[:,:,:3])
        if cv2.waitkey(1) & 0xFF ==ord('q'):
            self.close()

        pass
    def reset(self):
        time.sleep(.5)
        pyautogui.click(x=150,y=150)
        pyautogui.press('up')
        return self.get_observation()


    def close(self):
        cv2.destroyAllWindows()


    def get_observation(self):
        raw=np.array(self.cap.grab(self.game_location))[:,:,:3].astype(np.uint8)
        gray =cv2.cvtColor(raw,cv2.COLOR_BGR2GRAY)
        resized=cv2.resize(gray,(100,83))
        channel=np.reshape(resized,(1,83,100))
        return channel
    def get_done(self):
        done_cap = np.array(self.cap.grab(self.done_location))[:,:,:3]
        done_strings= ['GAME','GAHE']

        done=False
        res=pytesseract.image_to_string(done_cap)[:4]
        if res in done_strings:
            done=True

        return done,done_cap


env = webgame()
obs =env.get_observation()
plt.imshow(obs[0])
done, done_cap = env.get_done()

for episode in range(10):
    obs =env.reset()
    done = False
    total_reward=0
    while not done :
        obs, reward , done , info= env .step(env.action_space.sample())
        total_reward+=reward
    print("total reward for the episode is " + total_reward)

env_checker.check_env(env)

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


callback = TrainAndLoggingCallback(check_freq=1000, save_path=CHECKPOINT_DIR)


env = webgame()
model = DQN('CnnPolicy', env, tensorboard_log=LOG_DIR, verbose=1, buffer_size=1200000, learning_starts=1000)
model.learn(total_timesteps=100000, callback=callback)
model.load('train_first/best_mode l_50000')
model=DQN.load(os.path.join('train_first','best_model_88000'))


for episode in range(5):
    obs = env.reset()
    done = False
    total_reward = 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(int(action))
        # time.sleep(0.01)
        total_reward += reward
    print('Total Reward for episode {} is {}'.format(episode, total_reward))
    # time.sleep(2)

