import gymnasium as gym
from gymnasium import spaces

from minatar import Environment
from typing import Optional

class BaseEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array']}

  def __init__(self, game, display_time=50, use_minimal_action_set=False, **kwargs):
    self.game_name = game
    self.display_time = display_time
    self.game_kwargs = kwargs
    self.game = Environment(env_name=self.game_name, **kwargs)
    if use_minimal_action_set:
      self.action_set = self.game.minimal_action_set()
    else:
      self.action_set = list(range(self.game.num_actions()))
    self.action_space = spaces.Discrete(len(self.action_set))
    self.observation_space = spaces.Box(0.0, 1.0, shape=self.game.state_shape(), dtype=bool)

  def step(self, action):
    action = self.action_set[action]
    reward, done = self.game.act(action)
    return (self.game.state(), reward, done, False, {})
    
  def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
    super().reset(seed=seed)
    self.game.reset(self.np_random)
    return self.game.state(), {}

  def render(self, mode='human'):
    if mode == 'rgb_array':
      return self.game.state()
    elif mode == 'human':
      self.game.display_state(self.display_time)

  def close(self):
    if self.game.visualized:
      self.game.close_display()
    return 0