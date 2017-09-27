# *****************************************************************************
#
# Example DQN Agent for Playing Unity Games
# Use Python 2.7. (NOT tested using Python 3!)
# Tested on CPU Macbook pro; using Keras with tensorflow backend
# Recommend using a virtual env
#
# By Michael Richardson (michael.j.richardson@mq.edu.au)
#
# Some of this code was adpated from: https://keon.io/deep-q-learning/
#
# June 2017
#
# *****************************************************************************

import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DDQN_Agent:
    def __init__(self, state_size, action_size, gamma, learning_rate, epsilon, epsilon_decay, epsilon_min, epsilon_delay, memory_length):
        self.state_size = state_size                # number of environment state inputs
        self.action_size = action_size              # number of possible actions
        self.gamma = gamma                          # discount rate (e.g., .99)
        self.learning_rate = learning_rate          # learning rate (e.g., .0025)
        self.epsilon = epsilon                      # begining exploration rate  (e.g., 1.0)
        self.epsilon_decay = epsilon_decay          # rate of exploration decay  (e.g., .9999)
        self.epsilon_min = epsilon_min              # minimum exploration rate  (e.g., .05)
        self.epsilon_delay = epsilon_delay          # delay (n-frames) before exploration decay starts  (e.g., 25000)
        self.memory_length = memory_length          # size of replay memory
        self.memory = deque(maxlen=memory_length)   # replay memory array (tracks last n [s,a,r,s'] updates)
        self.model = self._build_model()
        self.target_model = self._build_model()

    # Neural Network for DQ RL Model
    def _build_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='relu'))
        model.add(Dense(20, activation='relu', kernel_initializer='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Update target mode by copying weights from model to target_model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Update agent memeory array
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Act in a epsilone-greedy manner (DQ trainging action - model + random defined actions)
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    # Action Replay Process for DQN RL
    def replay(self, batch_size, delay_count):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                a = np.argmax(self.model.predict(next_state)[0])
                t = self.target_model.predict(next_state)[0]
                target[action] = reward + self.gamma * t[a]
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        if delay_count > self.epsilon_delay: 
        	if self.epsilon > self.epsilon_min:
        		self.epsilon *= self.epsilon_decay

    # Load previous saved weights for NN model
    def load(self, name):
        self.model.load_weights(name)

    # Save weights for NN model
    def save(self, name):
        self.model.save_weights(name)

    # Save current agent parameters
    def save_agent_parameters(self, name):
        dfile = open(name, "w")
        dfile.write("Gamma: " + str(self.gamma) + "\n")
        dfile.write("Epsilon: " + str(self.epsilon) + "\n")
        dfile.write("Epsilon decay rate: " + str(self.epsilon_decay) + "\n")
        dfile.write("Epsilon min: " + str(self.epsilon_min) + "\n")
        dfile.write("Epsilon delay: " + str(self.epsilon_delay) + "\n")
        dfile.write("Learning rate: " + str(self.learning_rate) + "\n")
        dfile.write("Memory length: " + str(self.memory_length) + "\n")
        dfile.close()

    # resets replay memeory (deque) to emplty.
    def erase_replay_memory(self):
        self.memory = deque() 