There are 5 code files for 5 questions.
The first 4 question code files are names as p1_q(i).py
Each code file among first 3 consists of following main functions - 
    1. Val_reset -- Resets the value of Q_average, action_count and q_star after every episode
    2. Simulate -- Simulation function
Apart from this, p1_q1.py has two more functions for greedy and epsilon_greedy update
Similarly p1_q2.py has function to calculate the probab. distribution from Q_t('gibbs' function) and weighted_choice function that returns the action as per the required criteria for softmax

To run the code 1 file in terminal type- 
    python3 p1_q1.py 
etc.

You need to have following libraries in your python environment to run this code files without error - 
numpy, matplotlib, multiprocessing, tqdm, functools

Also an 'images' folder is added where the images will be saved after plotting.