import gym
from model import AC
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    s, info = env.reset()
    model = AC(env)
    reward = []
    MAX_EPISODE = 500
    for episode in range(MAX_EPISODE):
        s, info = env.reset()
        done = False
        ep_r = 0
        while not done:
            # env.render()
            a,log_prob = model.get_action(s)
            s_, rew, done, truncated, info  = env.step(a)
            ep_r += rew
            model.learn(log_prob,s,s_,rew)
            s = s_
        reward.append(ep_r)
        print(f"episode:{episode} ep_r:{ep_r}")
    plt.plot(reward)
    plt.show()