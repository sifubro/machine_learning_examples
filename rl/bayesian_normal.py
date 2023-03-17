# https://deeplearningcourses.com/c/artificial-intelligence-reinforcement-learning-in-python
# https://www.udemy.com/artificial-intelligence-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


np.random.seed(1)
NUM_TRIALS = 2000
BANDIT_MEANS = [1, 2, 3]


class Bandit:
  def __init__(self, true_mean):
    self.true_mean = true_mean
    # parameters for mu - prior is N(0,1)
    self.m = 0  #self.predicted_mean. This is the mean of the mean of X
    self.lambda_ = 1  # lambda_0
    # Hence our prior is the Normal N(m=0,lambda=1)

    self.sum_x = 0

    self.tau = 1
    self.N = 0 #num of times we play this bandit

  def pull(self):
    # X ~ N(true_mean, tau^-1)
    # samples from Normal distribution with mean=true_mean and precision tau see 2:52 lecture
    return np.random.randn() / np.sqrt(self.tau) + self.true_mean

  def sample(self):
    # mu | X ~ N(m, lambda^-1)
    # posterior is also a Gaussian. This has mean=m=predicted_mean and precision=lambda
    return np.random.randn() / np.sqrt(self.lambda_) + self.m

  def update(self, x):
    # N = 1 
    self.m = (self.tau * x + self.lambda_ * self.m) / (self.tau + self.lambda_)
    self.lambda_ += self.tau
    self.N += 1

  def update_in_video(self, x):
    # N = 1 
    self.lambda_ += self.tau  #new lambda equal tau + old lambda (since N=1)
    self.sum_x += x

    # posterior mean m=predicted_mean
    self.predicted_mean = self.tau * self.sum_x / self.lambda_  #m0=0 (we assumed prior for the mean is zero)
    self.N += 1



def plot(bandits, trial):
  x = np.linspace(-3, 6, 200)
  for b in bandits:
    y = norm.pdf(x, b.m, np.sqrt(1. / b.lambda_))
    plt.plot(x, y, label=f"real mean: {b.true_mean:.4f}, num plays: {b.N}")
  plt.title(f"Bandit distributions after {trial} trials")
  plt.legend()
  plt.show()


def run_experiment():
  bandits = [Bandit(m) for m in BANDIT_MEANS]

  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]
  rewards = np.empty(NUM_TRIALS)
  for i in range(NUM_TRIALS):
    # Thompson sampling
    j = np.argmax([b.sample() for b in bandits])

    # plot the posteriors
    if i in sample_points:
      plot(bandits, i)

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

    # update rewards
    rewards[i] = x

  cumulative_average = np.cumsum(rewards) / (np.arange(NUM_TRIALS) + 1)

  # plot moving average ctr
  plt.plot(cumulative_average)
  for m in BANDIT_MEANS:
    plt.plot(np.ones(NUM_TRIALS)*m)
  plt.show()

  return cumulative_average

if __name__ == '__main__':
  run_experiment()


