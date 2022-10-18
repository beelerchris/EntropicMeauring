import numpy as np
from keras.datasets import mnist
import gym
from copy import deepcopy

class SubWorldEnv(gym.Env):
    def __init__(self, init_p_min=0, init_p_max=None):

        (train_x, train_y), (test_x, test_labels) = mnist.load_data()

        self.data = np.zeros((train_x.shape[0] + test_x.shape[0], train_x.shape[1], train_x.shape[2]), dtype=np.int32)

        self.data[:train_x.shape[0]] = np.array(train_x) // 128
        self.data[train_x.shape[0]:] = np.array(test_x) // 128
        self.labels = list(train_y) + list(test_labels)

        self.init_p_min = init_p_min
        if init_p_max is None or init_p_max > self.data.shape[1] * self.data.shape[2]:
            self.init_p_max = self.data.shape[1] * self.data.shape[2]
        else:
            self.init_p_max = init_p_max

        self.observation_space = gym.spaces.Box(
                low=np.zeros((2, self.data.shape[1], self.data.shape[2]), dtype=np.int32),
                high=np.ones((2, self.data.shape[1], self.data.shape[2]), dtype=np.int32)
            )

        self.action_space = gym.spaces.MultiDiscrete([28, 28, 2, max(self.labels)+1])

        self.inds = np.arange(0, self.data.shape[1]*self.data.shape[2], 1, dtype=np.int32)

        self.reset()

    def reset(self):
        
        self.done = False
        self.state = np.zeros((2, self.data.shape[1], self.data.shape[2]), dtype=np.int32)
        
        ind = np.random.randint(0, self.data.shape[0])
        self.true = deepcopy(self.data[ind])
        self.label = deepcopy(self.labels[ind])

        n_p = np.random.randint(self.init_p_min, self.init_p_max+1)
        inds = np.random.permutation(self.inds)[:n_p]
        for n in inds:
            i = n // self.data.shape[1]
            j = n % self.data.shape[2]

            self.state[0, i, j] += 1
            self.state[1, i, j] += self.true[i, j]

        return self.state

    def step(self, action):

        if action[2] == 1:
            done = True
            if action[3] == self.label:
                r = 1 - self.state[0].mean()
            else:
                r = -1 - self.state[0].mean()

        else:
            r = 0
            self.state[0, action[0], action[1]] = 1
            self.state[1, action[0], action[1]] = deepcopy(self.true[action[0], action[1]])

        return self.state, r, done, {}
