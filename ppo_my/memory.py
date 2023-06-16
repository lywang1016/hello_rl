import numpy as np

class Trajectory():
    def __init__(self):
        self.states = []
        self.action = []
        self.reward = []
        self.next_states = []
        self.probs_old = []
        self.length = 0

    def remember(self, states, action, reward, next_states, probs_old):
        self.length += 1
        self.states.append(states)
        self.action.append(action)
        self.reward.append(reward)
        self.next_states.append(next_states)
        self.probs_old.append(probs_old)

class Memory:
    def __init__(self, batch_size):
        self.states = []
        self.action = []
        self.returns = []
        self.probs_old = []
        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return np.array(self.states), \
                np.array(self.action), \
                np.array(self.returns), \
                np.array(self.probs_old), \
                batches

    def store_memory(self, state, action, returns, probs_old):
        self.states.append(state)
        self.action.append(action)
        self.returns.append(returns)
        self.probs_old.append(probs_old)

    def clear_memory(self):
        self.states = []
        self.action = []
        self.returns = []
        self.probs_old = []
    

