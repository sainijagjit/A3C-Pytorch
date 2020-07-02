import cv2
import gym
import numpy as np
from gym.spaces.box import Box
import sys

def create_atari_env(env_id):
    env = gym.make(env_id)
    env = AtariRescale42x42(env)
    env = NormalizedEnv(env)
    return env

def _process_frame42(frame):
    #cropping the required area from the observation.
    frame = frame[34:34 + 160, :160]
    # Resize by half, then down to 42x42 (essentially mipmapping). If
    # we resize directly we lose pixels that, when mapped to 42x42,
    # aren't close enough to the pixel boundary.
    frame = cv2.resize(frame, (80, 80))
    frame = cv2.resize(frame, (42, 42))
    #We are converting rgb to greyscale by taking mean along the 2nd axis
    frame = frame.mean(2, keepdims=True)
    frame = frame.astype(np.float32)
    #Normalizing frame= frame*(1/255)
    #Normalizing the image vector
    frame *= (1.0 / 255.0)
    #Initial shape => (42,42,1)
    #Final shape => (1,42,42)
    # -1 (initial position of axis) 0 (final position of the axis)
    frame = np.moveaxis(frame, -1, 0)
    return frame

class AtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        #Box(low, high, shape, dtype)
        self.observation_space = Box(0.0, 1.0, [1, 42, 42])

    def observation(self, observation):
        return _process_frame42(observation)

class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        #Keeping note of no of steps
        self.num_steps += 1
        #We are calculating the running mean of the observations.
        #Observation.mean() => will give mean of the flattened array
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1 - self.alpha)

        #Applying the bias correction, to rectify error in initial timestep mean.
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)



