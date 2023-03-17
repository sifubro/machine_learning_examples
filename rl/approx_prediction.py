# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy
from sklearn.kernel_approximation import Nystroem, RBFSampler

GAMMA = 0.9
ALPHA = 0.01
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')


def epsilon_greedy(greedy, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    return greedy[s]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)


def gather_samples(grid, n_episodes=10000):
  samples = []
  for _ in range(n_episodes):
    s = grid.reset()
    samples.append(s)
    while not grid.game_over():
      a = np.random.choice(ALL_POSSIBLE_ACTIONS)
      r = grid.move(a)
      s = grid.current_state()
      samples.append(s)
  return samples


class Model:
  def __init__(self, grid):
    # fit the featurizer to data
    samples = gather_samples(grid)
    # self.featurizer = Nystroem()
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)
    dims = self.featurizer.n_components

    # initialize linear model weights
    self.w = np.zeros(dims)

  def predict(self, s):
    x = self.featurizer.transform([s])[0]  #x = phi(s)
    return x @ self.w

  def grad(self, s):
    # return the gradient of V(s) w.r.t w which is x
    x = self.featurizer.transform([s])[0]
    return x


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  greedy_policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }

  model = Model(grid)
  mse_per_episode = []

  # repeat until convergence
  n_episodes = 10000
  for it in range(n_episodes):
    if (it + 1) % 100 == 0:
      print(it + 1)

    s = grid.reset()
    Vs = model.predict(s)  # w*x = w*phi(s)
    n_steps = 0  # how many steps we took in the episode
    episode_err = 0 # total square error in our estimates for the episode
    while not grid.game_over():
      a = epsilon_greedy(greedy_policy, s)
      r = grid.move(a)
      s2 = grid.current_state()

      # get the target
      if grid.is_terminal(s2):
        target = r
      else:
        Vs2 = model.predict(s2) # next value V(s') = w*phi(s')
        target = r + GAMMA * Vs2 # y

      # update the model
      g = model.grad(s) # gradient of V(s)
      err = target - Vs # = y - w*phi(s)  (TD0 error)
      model.w += ALPHA * err * g
      
      # accumulate error
      n_steps += 1
      episode_err += err*err

      # update state
      s = s2
      Vs = Vs2
    
    # mse over the episode
    mse = episode_err / n_steps
    mse_per_episode.append(mse)

  plt.plot(mse_per_episode)
  plt.title("MSE per episode")
  plt.show()

  # obtain predicted values
  V = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      V[s] = model.predict(s)
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(greedy_policy, grid)
