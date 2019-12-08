from qlearn_env import Env
import numpy as np
import time
import os

# create environment
env = Env()

# QTable : contains the Q-Values for every (state,action) pair
qtable = np.random.rand(env.stateCount, env.actionCount).tolist()

# hyperparameters
epochs = 50
gamma = 0.1
epsilon = 0.08
decay = 0.1

# training loop
for i in range(epochs):
    state, reward, done = env.reset()
    steps = 0

    while not done:
        os.system('clear')
        print("epoch #", i+1, "/", epochs)
        env.render()
        time.sleep(0.05)

        # count steps to finish game
        steps += 1

        
        action = qtable[state].index(max(qtable[state]))

        # take action
        next_state, reward, done = env.step(action)

        # update qtable value with Bellman equation
        qtable[state][action] = reward + gamma * max(qtable[next_state])

        # update state
        state = next_state
    # The more we learn, the less we take random actions
    

    print("\nDone in", steps, "steps".format(steps))
    time.sleep(0.8)