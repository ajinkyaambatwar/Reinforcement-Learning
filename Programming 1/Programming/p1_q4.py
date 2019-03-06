import numpy as np
import matplotlib.pyplot as mpt
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import itertools

#User defined function imports#
from p1_q1 import simulate as s1
from p1_q2 import simulate as s2
from p1_q3 import simulate as s3
from p1_q1 import val_reset

if __name__ == '__main__':
    k_arms = [10,1000]
    Q_average_k0 = np.zeros(k_arms[0])
    action_count_k0 = np.zeros(k_arms[0])
    q_star_k0 = np.random.randn(k_arms[0])
    Q_average_k1 = np.zeros(k_arms[1])
    action_count_k1 = np.zeros(k_arms[1])
    q_star_k1 = np.random.randn(k_arms[1])

    parameter_list = [{'k_arms': k_arms[0], 'Q_average': Q_average_k0, 'action_count': action_count_k0,'q_star': q_star_k0},
                        {'k_arms': k_arms[1],'Q_average': Q_average_k1, 'action_count': action_count_k1,'q_star': q_star_k1}]
    
    runs = 2000
    time = 1000

    #For epsilon-greedy
    epsilon = 0.1
    rew_epsilon_greedy = np.zeros(shape = (runs, time))
    best_action_count_epsilon_greedy = np.zeros(rew_epsilon_greedy.shape)
    [rewards1_10_arms,best_action1_10_arms] = s1(epsilon,runs,time,parameter_list[0],k_arms[0],best_action_count_epsilon_greedy,rew_epsilon_greedy)
    [rewards1_1000_arms,best_action1_1000_arms] = s1(epsilon,runs,time,parameter_list[1],k_arms[1],best_action_count_epsilon_greedy,rew_epsilon_greedy)
    print("$\epsilon$ greedy simulated!!")

    #For softmax
    Temp = 0.01
    rew_softmax = np.zeros(shape = (runs, time))
    best_action_count_softmax = np.zeros(rew_softmax.shape)
    [rewards2_10_arms,best_action2_10_arms] = s2(Temp,runs,time,parameter_list[0],k_arms[0],best_action_count_softmax,rew_softmax)
    [rewards2_1000_arms,best_action2_1000_arms] = s2(Temp,runs,time,parameter_list[1],k_arms[1],best_action_count_softmax,rew_softmax)
    print('Softmax simulated')

    #For UCB
    c = 2
    step = 0
    rew_ucb = np.zeros(shape = (runs, time))
    best_action_count_ucb = np.zeros(rew_ucb.shape)
    [rewards3_10_arms,best_action3_10_arms] = s3(c,runs,time,parameter_list[0],k_arms[0],best_action_count_ucb,rew_ucb,step)
    [rewards3_1000_arms,best_action3_1000_arms] = s3(c,runs,time,parameter_list[1],k_arms[1],best_action_count_ucb,rew_ucb,step)
    print('UCB simulated')

    #----Plotting----#
    #Epsilon greedy
    #Average reward
    
    mpt.figure(11)
    mpt.title('Average reward comparison for $\epsilon$ greedy with $\epsilon = 0.1$')
    mpt.plot(rewards1_10_arms, label = '#arms = 10')
    mpt.plot(rewards1_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Average rewards')
    mpt.legend()
    mpt.savefig('images/rew_comp_eps.png')
    #Optimal Action
    mpt.figure(12)
    mpt.title('Optimal action comparison for $\epsilon$ greedy with $\epsilon = 0.1$')
    mpt.plot(best_action1_10_arms, label = '#arms = 10')
    mpt.plot(best_action1_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Optimal action(%)')
    mpt.legend()
    mpt.savefig('images/opt_act_comp_eps.png')
    print('epsilon-greedy plotted')
    
    #Softmax
    #Average reward
    mpt.figure(21)
    mpt.title('Average reward comparison for Softmax with Temp = 0.01')
    mpt.plot(rewards2_10_arms, label = '#arms = 10')
    mpt.plot(rewards2_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Average rewards')
    mpt.legend()
    mpt.savefig('images/rew_comp_temp.png')
    #Optimal Action
    mpt.figure(22)
    mpt.title('Optimal action comparison for Softmax with Temp = 0.01')
    mpt.plot(best_action2_10_arms, label = '#arms = 10')
    mpt.plot(best_action2_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Optimal action(%)')
    mpt.legend()
    mpt.savefig('images/opt_act_comp_temp.png')
    print("Softmax plotted")
    
    #UCB
    #Average reward
    mpt.figure(31)
    mpt.title('Average reward comparison for UCB with c=2')
    mpt.plot(rewards3_10_arms, label = '#arms = 10')
    mpt.plot(rewards3_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Average rewards')
    mpt.legend()
    mpt.savefig('images/rew_comp_ucb.png')
    #Optimal Action
    mpt.figure(32)
    mpt.title('Optimal action comparison for UCB with c=2')
    mpt.plot(best_action3_10_arms, label = '#arms = 10')
    mpt.plot(best_action3_1000_arms, label = '#arms = 1000')
    mpt.xlabel('Iterations')
    mpt.ylabel('Optimal action(%)')
    mpt.legend()
    mpt.savefig('images/opt_act_comp_ucb.png')
    print('UCB plotted')
    mpt.close()