def reward_function_1(x, angle):
    a = abs(angle)
    if a < 0.04:
        reward = 1
    elif a < 0.1:
        reward = 0.5
    elif a < 0.18:
        reward = 0.1
    else:
        reward = -1
    x_ = abs(x)
    if x_ < 0.1:
        reward += 0.2
    elif x_ < 0.4:
        reward += 0
    elif x_ < 1.0:
        reward -= 0.3
    else:
        reward -= 0.8
    return reward