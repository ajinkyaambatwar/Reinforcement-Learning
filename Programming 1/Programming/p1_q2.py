import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as mpt

'''
@parameters - 
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

def weighted_choice(weights):
        '''
        Now the goal is like this
        We choose an action with probability proportional
        to its action value which means a uniform random number
        generated if lies between gibbs(action) and gibbs(action with just lesser value)
        then that action is choosen
        SO for that now we shall use this method - 
        ->cummulative Sum up all the action values(technically it will become one-the last element)
        ->Now select a random number from uniform distibution (0,sum)
        ->Then the searchsorted will basically find that index(so here it will be the action)such that the order of cum sum 
        gibbs action probability is not altered.
        ->For ex. --> cumsum=[0.1,0.3,0.7,0.9,1](basically action 1 has 0.3 part of getting selected)
        Now random number if
        1) <0.1 - action 0
        2) >0.1 <0.3 action 1 etc.
        '''
        totals = np.cumsum(weights)
        norm = totals[-1]
        throw = np.random.rand()*norm
        return np.searchsorted(totals, throw)

def gibbs(q_average,T):
        '''
        Implementation of Gibbs-softmax distribution from which the action will be sampled with a probability
        '''
        den = np.sum(np.exp(q_average)/T)
        act_prob = np.exp(q_average/T)/den
        return act_prob

def act(T,parameters):
        '''
        Now action selection probability is proportional to exp(Q_a(t))
        '''
        Q_average = parameters['Q_average']
        action_count = parameters['action_count']
        q_star = parameters['q_star']

        act_prob = gibbs(Q_average,T)
        action_choice = weighted_choice(act_prob)
        act_reward = np.random.normal(q_star[action_choice], 1)
        action_count[action_choice] += 1
        Q_average[action_choice] += (act_reward - Q_average[action_choice])/(action_count[action_choice])

        parameters['Q_average'] = Q_average
        parameters['action_count'] = action_count
        return action_choice, act_reward

#---Setting up simulation----#
'''
@runs = 2000
@time = 1000
@Temp = [0.01, 1, 100]
@rew -- A (runs, time) shaped array to store the actual_rewards 
        of each iteration for each run for given value of Temp
@best_action_count -- A (runs, time) shaped array to counts how
        many times the optimal action was selected. 
@rewards -- A vector of length (time) that takes the average of rewards 
        over the number of runs for each iteration
@best_action -- A (time) lenghted vector which returns the percentage
        selection of optimal action
'''

def simulate(Temp, runs, time, parameters, k_arms, best_action_count, rew):
        '''
        Defines the setup for simulation.
        In every run, the Q_average, q_start and action_count parameters are reset to their initial values
        After all the iterations, the required average can be calculated by taking the mean of rew matrix across axis 1(runs axis)
        Same for best_action_count
        '''
        for i in tqdm(range(runs)):
                parameters = val_reset(parameters, k_arms)
                for j in range(time):
                        action_choice, actual_reward = act(Temp,parameters)
                        rew[i,j] = actual_reward
                        if action_choice == np.argmax(parameters['q_star']):
                                best_action_count[i,j] = 1
        rewards = rew.mean(axis = 0)
        best_action = best_action_count.mean(axis = 0)
        best_action = best_action*100
        rew = np.zeros(shape = (runs, time))
        best_action_count = np.zeros(rew.shape)
        return rewards,best_action


if __name__ == '__main__':
        k_arms = 10    
        parameters = {'k_arms': k_arms, 'Q_average': np.zeros(k_arms), 'action_count': np.zeros(k_arms),'q_star': np.random.randn(k_arms)}
        runs = 2000
        time = 1000
        Temp = [0.01, 1, 100]
        rew = np.zeros(shape = (runs, time))
        best_action_count = np.zeros(rew.shape)
        pool = Pool(processes=3)
        partial_func = partial(simulate, runs = runs, time = time, k_arms = k_arms,parameters = parameters,
                                best_action_count = best_action_count,rew = rew)
        [[rewards1, best_action1], [rewards2, best_action2], [rewards3, best_action3]] = pool.map(partial_func, Temp)
        print("Simulation done!!")

        #---Now plotting---#
        #The average reward plot#
        mpt.figure(1)
        mpt.title("Average reward with Softmax distribution")
        mpt.xlabel("Iteration")
        mpt.ylabel("Average reward")
        rewards = [rewards1,rewards2,rewards3]
        for T,reward in zip(Temp,rewards):
                mpt.plot(reward,label ="Temp = $%.02f$" %(T))
        mpt.legend()
        mpt.savefig('images/figure_2.1.png')

        #Optimal action percentage plot#
        mpt.figure(2)
        mpt.title("OPtimal Action with Softmax distribution")
        mpt.xlabel("Iteration")
        mpt.ylabel("Optimal action(%)")
        best_action = [best_action1, best_action2, best_action3]
        for T,optimal_action in zip(Temp,best_action):
                mpt.plot(optimal_action,label ="Temp = $%.02f$" %(T))
        mpt.legend()
        mpt.savefig('images/figure_2.2.png')
        mpt.close()
