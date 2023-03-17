# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3 # threshold for convergence


def print_values(V, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      v = V.get((i,j), 0)
      if v >= 0:
        print(" %.2f|" % v, end="")
      else:
        print("%.2f|" % v, end="") # -ve sign takes up an extra space
    print("")


def print_policy(P, g):
  for i in range(g.rows):
    print("---------------------------")
    for j in range(g.cols):
      a = P.get((i,j), ' ')
      print("  %s  |" % a, end="")
    print("")



if __name__ == '__main__':

  ### define transition probabilities and grid ###
  # the key is (s, a, s'), the value is the probability
  # that is, transition_probs[(s, a, s')] = p(s' | s, a)
  # any key NOT present will considered to be impossible (i.e. probability 0)
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  rewards = {}

  grid = standard_grid()
  for i in range(grid.rows):
    for j in range(grid.cols):
      s = (i, j) # current state
      if not grid.is_terminal(s):  # check if s is a terminal state
        for a in ACTION_SPACE:  # if s it NOT a terminal state, loop through action space to build a probability distribution
          s2 = grid.get_next_state(s, a)  # s2 is the next state from s after doing action a  (we don't have to go to that state)
          transition_probs[(s, a, s2)] = 1 # everything not in the dictionary will be 0
          if s2 in grid.rewards:  # in general the reward may depend on the entire tuple (s, a, s2) rather than only the state s2
            rewards[(s, a, s2)] = grid.rewards[s2]

  ### fixed policy ###
  # states to actions
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L',
  }
  print_policy(policy, grid)

  # initialize V(s) = 0
  V = {}
  for s in grid.all_states():
    V[s] = 0

  gamma = 0.9 # discount factor

  # POLICY EVALUATION CODE
  # repeat until convergence
  it = 0
  while True:  # it stops only when Delta < threshold
    biggest_change = 0
    for s in grid.all_states():  # loop through all states
      if not grid.is_terminal(s): # if s is NOT a terminal state
        old_v = V[s]
        new_v = 0 # we will accumulate the answer
        for a in ACTION_SPACE:
          for s2 in grid.all_states():

            # action probability is deterministic
            # the action_probability is 1 only when a is the assigned action for state s
            action_prob = 1 if policy.get(s) == a else 0
            
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)  # get reward for the triplet (s, a, s')

            ## BELLMAN'S EQUATION
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + gamma * V[s2])

        # after done getting the new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    print("iter:", it, "biggest_change:", biggest_change)
    print_values(V, grid)
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  print("\n\n")
