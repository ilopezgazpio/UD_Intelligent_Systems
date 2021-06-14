from LinearModel import LinearModel
import numpy as np


class DQNAgent(object):
    # Artificial intelligence responsible for observing past events, learn from them and take new actions

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95               # discount rate
        self.epsilon = 1.0              # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0]) # returns action


    def train(self, state, action, reward, next_state, done):
        '''
        prediction -> y = Q(s,a)
        target -> ŷ = reward + gamma * max_action Q(s_next, a_next)
        '''
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target
        # trick not to update neurons for other actions, we make error 0 for all actions except the one selected by policy

        # Run one training step
        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


