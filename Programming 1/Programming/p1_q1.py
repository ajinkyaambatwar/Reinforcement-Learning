import numpy as np
import matplotlib.pyplot as mpt
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

'''
@parameters
1.@q_star -- The actual q_star taken from a stndard gaussian distribution
2.@Q_average -- average Q to be computed incrementally in each step for each of the actions
3.@action_count -- Action count for each of the 10 actions
'''

def val_reset(parameters,k_arms):
        '''
        Resets values of average Q, q_star and action count after every run
        '''
        parameters['k_arms'] = k_arms
        parameters['Q_average'] = np.zeros(k_arms)
        parameters['action_count'] = np.zeros(k_arms)
        parameters['q_star'] = np.random.randn(k_arms)
        return parameters

def greedy_update(parameters):
        '''
        The greedy update
        Always chooses the action with maximum current
        value of Q_average. No exploration done.
        The actual reward(act_reward) is sampled
        from the the normal distribution with the mean value equal to the true
        mean value of the action for which the current Q_average is the maximum
        '''
        Q_average = parameters['Q_average']
        action_count = parameters['action_count']
        q_star = parameters['q_star']

        max_q = np.argmax(Q_average)
        act_reward = np.random.normal(q_star[max_q], 1)
        action_count[max_q] += 1
        Q_average[max_q] += (act_reward - Q_average[max_q])/(action_count[max_q])
        parameters['Q_average'] = Q_average
        parameters['action_count'] = action_count
        return max_q,act_reward

def eps_greedy(eps, parameters):
        '''
        epsilon greedy method
        A random action is chosen with probaility epsilon and the greedy
        action is choosen with probability (1-epsilon). This allows exploration.
        '''
        Q_average = parameters['Q_average']
        action_count = parameters['action_count']
        q_star = parameters['q_star']
        k_arms = parameters['k_arms']
        
        if np.random.random() < eps:    #np.random.random will sample a number from uniform[0,1]
                max_q = np.random.randint(k_arms)
                act_reward = np.random.normal(q_star[max_q], 1)
                action_count[max_q] += 1
                Q_average[max_q] += (act_reward - Q_average[max_q])/(action_count[max_q])
        else:
                max_q,act_reward = greedy_update(parameters)
        parameters['Q_average'] = Q_average
        parameters['action_count'] = action_count
        return max_q, act_reward


#---Setting up simulation variables----#
'''
@runs = 2000
@time = 1000
@eps = [0, 0.01, 0.1]
@rew -- An (len(eps), runs, time) shaped array to store the actual_rewards 
        of each iteration for each run for given values of eps
@best_action_count -- a (len(eps), runs, time) shaped array to counts how
        many times the optimal action was selected. 
@rewards -- A (len(eps), time) shaped array that takes the average of rewards 
        over the number of runs for each iteration and for each value of epsion
@best_action -- A (len(eps), time) shaped array which returns the percentage
        selection of optimal action
@average_reward -- averge reward after each step computed incrementally
'''

def simulate(eps, runs, time, parameters,k_arms, best_action_count, rew):
        '''
        Defines the setup for simulation.
        In every run, the Q_average, q_start and action_count parameters are reset to their initial values
        After all the iterations, the required average can be calculated by taking the mean of rew matrix across axis 1(runs axis)
        Same for best_action_count
        '''
        for i in tqdm(range(runs)):
                parameters = val_reset(parameters,k_arms)
                for j in range(time):
                        action_choice, actual_reward = eps_greedy(eps,parameters)
                        rew[i,j] = actual_reward
                        if action_choice == np.argmax(parameters['q_star']):
                                best_action_count[i,j] = 1
        rewards = rew.mean(axis = 0)
        best_action = best_action_count.mean(axis = 0)
        best_action = best_action*100
        rew = np.zeros(shape = (runs, time))
        best_action_count = np.zeros(rew.shape)
        return rewards,best_action

#-------Now simulation--------#
if __name__=='__main__':
        k_arms = 10
        parameters = {'k_arms': k_arms, 'Q_average': np.zeros(k_arms), 'action_count': np.zeros(k_arms),'q_star': np.random.randn(k_arms)}
        runs = 2000
        time = 1000
        eps = [0,0.01,0.1]
        rew = np.zeros(shape = (runs, time))
        best_action_count = np.zeros(rew.shape)
        pool = Pool(processes=3)
        partial_func = partial(simulate, runs = runs, time = time,k_arms = k_arms, parameters = parameters,
                           best_action_count = best_action_count, rew = rew)
        [[rewards1, best_action1], [rewards2, best_action2], [rewards3, best_action3]] = pool.map(partial_func, eps)
        print("Simulation done!!")

        #---Now plotting---#
        #The mean reward plot#
        mpt.figure(1)
        mpt.title('Average reward with $\epsilon$ greedy')
        mpt.xlabel("Iteration")
        mpt.ylabel("Average reward")
        rewards = [rewards1, rewards2, rewards3]
        for ep,reward in zip(eps,rewards):
                mpt.plot(reward,label ="$\epsilon = %.02f$" %(ep))
        mpt.legend()
        mpt.savefig('images/figure_1.png')
                
        #Optimal action percentage plot#
        mpt.figure(2)
        mpt.title('Optimal action with $\epsilon$ greedy')
        mpt.xlabel("Iteration")
        mpt.ylabel("Optimal action(%)")
        best_action = [best_action1, best_action2, best_action3]
        for ep,optimal_action in zip(eps,best_action):
                mpt.plot(optimal_action,label ="$\epsilon = %.02f$" %(ep))
        mpt.legend()
        mpt.savefig('images/figure_2.png')
        mpt.close()

