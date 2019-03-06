import numpy as np
import matplotlib.pyplot as mpt
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial

'''
@parameters(Dictionary)-
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

def update(c, step, parameters):
        '''
        The selected action is the one which has highest Q_estimation value
        '''
        Q_average = parameters['Q_average']
        action_count = parameters['action_count']
        q_star = parameters['q_star']

        Q_estimation = Q_average + c*np.sqrt(np.log(step+1)/(action_count+1e-5))
        max_q = np.argmax(Q_estimation)
        step = step+1
        act_reward = np.random.normal(q_star[max_q], 1)
        action_count[max_q] += 1
        Q_average[max_q] += (act_reward - Q_average[max_q])/(action_count[max_q])

        parameters['Q_average'] = Q_average
        parameters['action_count'] = action_count
        return step, max_q, act_reward

def simulate(c, runs, time, parameters, k_arms, best_action_count, rew,step):
        '''
        Defines the setup for simulation.
        In every run, the Q_average, q_start and action_count parameters are reset to their initial values
        After all the iterations, the required average can be calculated by taking the mean of rew matrix across axis 1(runs axis)
        Same for best_action_count
        '''
        for i in tqdm(range(runs)):
                parameters = val_reset(parameters, k_arms)
                for j in range(time):
                        step, max_q,actual_reward = update(c, step, parameters)
                        rew[i,j] = actual_reward
                        if max_q == np.argmax(parameters['q_star']):
                                best_action_count[i,j] = 1
        rewards = np.mean(rew, axis=0)
        best_action = np.mean(best_action_count,axis=0)
        best_action = best_action*100
        step = 0
        return rewards,best_action

#---Setting up simulation----#
'''
@runs = 2000
@time = 1000
@c -- UCB paramter that determines the extent of exploration 
@rew -- A (runs, time) shaped array to store the actual_rewards 
        of each iteration for each run for given value of Temp
@best_action_count -- A (runs, time) shaped array to counts how
        many times the optimal action was selected. 
@rewards -- A vector of length (time) that takes the average of rewards 
        over the number of runs for each iteration
@best_action -- A (time) lenghted vector which returns the percentage
        selection of optimal action
'''
if __name__ == '__main__':
        k_arms = 10
        parameters = {'k_arms': k_arms, 'Q_average': np.zeros(k_arms), 'action_count': np.zeros(k_arms),'q_star': np.random.randn(k_arms)}
        step = 0
        runs = 2000
        time = 1000
        c = [2,10]
        rew = np.zeros(shape = (runs, time))
        best_action_count = np.zeros(rew.shape)
        pool = Pool(processes=4)
        partial_func = partial(simulate, runs = runs, time = time, parameters = parameters, k_arms = k_arms,  
                                best_action_count = best_action_count, rew = rew,step=step)
        [[rewards1, best_action_count1],[rewards2, best_action_count2]] = pool.map(partial_func, c)
        print("Simulation done!!")
        
        #---Now Plotting---#

        #Average reward plot
        mpt.plot(1)
        mpt.title('Average reward with UCB')
        mpt.xlabel("Iteration")
        mpt.ylabel("Average reward")
        rewards = [rewards1,rewards2]
        for cval,reward in zip(c,rewards):
                mpt.plot(reward,label ="c = $%.02f$" %(cval))
        mpt.legend()
        mpt.savefig('images/figure_3.1.png')

        #Optimal action percentage plot#
        mpt.figure(2)
        mpt.title("OPtimal Action with UCB")
        mpt.xlabel("Iteration")
        mpt.ylabel("Optimal action(%)")
        best_action = [best_action_count1, best_action_count2]
        for cval,optimal_action in zip(c,best_action):
                mpt.plot(optimal_action,label ="c = $%.02f$" %(cval))
        mpt.legend()
        mpt.savefig('images/figure_3.2.png')
        mpt.close()
