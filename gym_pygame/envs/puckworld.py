from .base import BaseEnv


class PuckWorldEnv(BaseEnv):
  # This is a continuous game.
  def __init__(self, normalize=False, display=False, render_mode='rgb_array', **kwargs):
    self.game_name = 'PuckWorld'
    self.render_mode = render_mode
    self.init(normalize, display, **kwargs)
    
  def get_ob_normalize(cls, state):
    state_normal = cls.get_ob(state)
    # TODO
    return state_normal

if __name__ == '__main__':
  env = PuckWorldEnv(normalize=True)
  print('Action space:', env.action_space)
  print('Action set:', env.action_set)
  print('Obsevation space:', env.observation_space)
  print('Obsevation space high:', env.observation_space.high)
  print('Obsevation space low:', env.observation_space.low)

  for i in range(1):
    ob, _ = env.reset(seed=0)
    for _ in range(500):
      action = env.action_space.sample()
      ob, reward, done, _, _ = env.step(action)
      # env.render('human')
      #env.render('rgb_array')
      print('Observation:', ob)
      print('Reward:', reward)
      print('Done:', done)
      # break
      if done:
        break
  env.close()
