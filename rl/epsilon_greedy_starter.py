# From the course: Bayesin Machine Learning in Python: A/B Testing
# https://deeplearningcourses.com/c/bayesian-machine-learning-in-python-ab-testing
# https://www.udemy.com/bayesian-machine-learning-in-python-ab-testing
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np


NUM_TRIALS = 10000
EPS = 0.1  # probability to select a random bandit and not the optimal (MLE estimate)
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]  #true win rate of each bandit


class Bandit:
  def __init__(self, p):
    # p: represents the *true* win rate for that bandit (pretend you don't know this value)
    self.p = p
    self.p_estimate = # TODO   (*estimate* of the true winrate for that bandit)
    self.N = # TODO  (number of samples)
  
  def pull(self):
    # pull the arm of the bandit
    # draw a 1 with probability p (1=win/0=lose)
    return np.random.random() < self.p

  def update(self, x):
    # update the distribution for the bandit whose arm we just pulled with .pull()
    # takes a sample value x and uses to update the p_estimate
    self.N = # TODO  
    self.p_estimate = # TODO 


def experiment():
  # runs epsilon-greedy algorithm

  # Initialize list of bandits objects with winrates equal to BANDIT_PROBABILITIES constant
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]

  rewards = np.zeros(NUM_TRIALS)
  num_times_explored = 0
  num_times_exploited = 0
  num_optimal = 0

  # index of the bandit with the maximum true mean
  optimal_j = np.argmax([b.p for b in bandits])
  print("optimal j:", optimal_j)

  for i in range(NUM_TRIALS):

    # use epsilon-greedy to select the next bandit j
    if np.random.random() < EPS:
      # choose bandit at random
      num_times_explored += 1
      j = # TODO
    else:
      # choose current best bandit based on p_estimate (MLE estimate)
      num_times_exploited += 1
      j = # TODO

    if j == optimal_j:
      num_optimal += 1

    # pull the arm for the bandit with the largest sample
    # x is the reward
    x = bandits[j].pull()

    # update rewards log
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    # i.e. update estimate of the mean reward
    bandits[j].update(x)

    

  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num_times_explored:", num_times_explored)
  print("num_times_exploited:", num_times_exploited)
  print("num times selected optimal bandit:", num_optimal)

  # plot the results
  cumulative_rewards = np.cumsum(rewards)  #cumulative sum of rewards
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)  #cumulative win rate per iteration
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

if __name__ == "__main__":
  experiment()
