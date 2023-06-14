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
    

