import os
import importlib
import numpy as np
import gymnasium
from gymnasium import spaces
from ple import PLE

from gym_pygame.envs.base import BaseEnv


class PixelcopterEnv(BaseEnv):
  def __init__(self, normalize=False, display=False, **kwargs):
    self.game_name = 'Pixelcopter'
    self.init(normalize, display, **kwargs)
    
  def get_ob_normalize(self, state):
    state_normal = self.get_ob(state)
    # TODO
    return state_normal

if __name__ == '__main__':
  env = PixelcopterEnv(normalize=True)
  env.seed(0)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(10):
    ob, _ = env.reset()
    while True:
      action = env.action_space.sample()
      ob, reward, done, _, _ = env.step(action)
      # env.render('rgb_array')
      env.render('human')
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      if done:
        break
  env.close()