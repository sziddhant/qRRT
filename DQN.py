#!/usr/bin/env python

import gym
import gym_qrrt
import time
import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers import GaussianNoise
import collections
from statistics import median, mean
from keras import backend as K
import matplotlib.pyplot as plt
episodes = 10000
plotg=[[0 for x in range(720)] for y in range(720)]





class DQNAgent:
    def __init__(self, state_size=2, action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=50000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99999
        self.learning_rate = 0.01
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=2, activation='elu',kernel_initializer='random_uniform'))
        #model.add(GaussianNoise(0.5))
        #model.add(Dense(8, activation='elu',kernel_initializer='random_uniform'))
        model.add(Dense(4, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


if __name__ == "__main__":
    env = gym.make('qrrt-v0')
    agent = DQNAgent(env)
    #agent.model.save('obstacle.h5')
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, 2])
        cum_reward = 0
        for time_t in range(1500):
            action = agent.act(state)
            # action=env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 2])
            agent.remember(state, action, reward, next_state, done)
            #env.render()

            cum_reward += reward
            if done:
                if cum_reward>=0:
                    #env.render()
                    print("episode: {}/{}, score: {}".format(e, episodes, cum_reward))
                break
            state = (random.randrange(720),random.randrange(500))
            #state= (600,random.randrange(500))
            env.state=state
            state=np.reshape(state,[1,2])
            #print(state)
        print(e, cum_reward)
        #env.render()

        cum_reward = 0

        agent.replay(min(64,  len(agent.memory)))
    agent.model.save('obstacle.h5')
    for i in range(500):
        for j in range(720):
            state = (j, i)
            state = np.reshape(state, [1, 2])
            act_values = agent.model.predict(state)
            action = max(act_values[0])
            plotg[i][j] =action
        print (i)

    plt.imshow(plotg)
    plt.colorbar()
    plt.show()
    for e in range(10):
        state = env.reset()
        state = np.reshape(state, [1, 2])
        action = env.action_space.sample()
        #print(action)
        next_state, reward, done, _ = env.step(action)
        cum_reward = 0
        for time_t in range(1000):
            #env.render()
            act_values = agent.model.predict(state)
            action = np.argmax(act_values[0])
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 2])
            state = next_state
            cum_reward += reward
            env.render()
            if done:
                print("score: {}"
                      .format(time_t))
                print(cum_reward)
                break