import numpy as np
from ads import UserAdvert
import matplotlib.pyplot as mpt

ACTION_SIZE = 3
STATE_SIZE = 4
TRAIN_STEPS = 10000  # Change this if needed
LOG_INTERVAL = 10

W = np.zeros(shape = (3,4))
grad = np.zeros(shape = (3,4))
average_reward = 0

def policy(state,w):
	z = state.dot(w)
	exp = np.exp(z)
	return exp/np.sum(exp)

def pi_grad(gibbs):
    s = np.reshape(gibbs,(-1,1))
    grad = np.diagflat(s) - np.dot(s, s.T)
    return grad

def learnBandit():
    env = UserAdvert()
    rew_vec = []
    global W
    global grad
    global average_reward
    for train_step in range(TRAIN_STEPS):
        learning_rate = np.exp(-0.0001*train_step)
        state = env.getState()
        stateVec = state["stateVec"]
        stateId = state["stateId"]
        # ---- UPDATE code below ---- #
        z = W.dot(stateVec)
        exp = np.exp(z)
        probs = exp/np.sum(exp)
        action = int(np.random.choice(range(3),p=probs.reshape(3,)))
        reward = env.getReward(stateId, action)
        average_reward += (reward - average_reward)/(train_step+1)
        # ----------------------------

        # ---- UPDATE code below ------
        for act in range(3):
            flag = (1 if act==action else 0)
            W[act] += learning_rate*(reward - average_reward)*(flag - probs[act])*stateVec
        # ----------------------------
        if train_step % LOG_INTERVAL == 0:
            print("Testing at: " + str(train_step))
            count = 0
            test = UserAdvert()
            for e in range(450):
                teststate = test.getState()
                testV = teststate["stateVec"]
                testI = teststate["stateId"]
                # ---- UPDATE code below ------ #
                z = W.dot(testV)
                exp = np.exp(z)
                probs = exp/np.sum(exp) 
                # ----------------------------
                act = int(np.random.choice(range(3), p=probs.reshape(3,)))
                reward = test.getReward(testI, act)
                count += (reward/450.0)
            rew_vec.append(count)

    # ---- UPDATE code below ------
    mpt.plot(rew_vec)
    mpt.title('alpha = exp(-0.0001*t)')
    mpt.ylabel('Average Reward')
    mpt.xlabel('Iterations')
    mpt.savefig()


if __name__ == '__main__':
    learnBandit()
