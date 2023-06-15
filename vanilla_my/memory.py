import numpy as np

class Trajectory():
    def __init__(self):
        self.states = []
        self.action = []
        self.reward = []
        self.next_states = []
        self.length = 0

    def remember(self, states, action, reward, next_states):
        self.length += 1
        self.states.append(states)
        self.action.append(action)
        self.reward.append(reward)
        self.next_states.append(next_states)

class CriticMemory:
    def __init__(self, batch_size):
        self.states = []
        self.state_bootstrapping = []
        self.returns = []
        self.value_bootstrapping = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), \
                np.array(self.state_bootstrapping), \
                np.array(self.returns), \
                np.array(self.value_bootstrapping), \
                batches

    def store_memory(self, state, state_bootstrapping, returns, value):
        self.states.append(state)
        self.state_bootstrapping.append(state_bootstrapping)
        self.returns.append(returns)
        self.value_bootstrapping.append(value)

    def clear_memory(self):
        self.states = []
        self.state_bootstrapping = []
        self.returns = []
        self.value_bootstrapping = []
    

