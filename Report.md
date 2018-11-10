# Introduction

The goal of this project is to train an agent with deep reinforcement learning algorithms so that it can collect as many yellow bananas and avoid as many blue bananas as possible in a large, square world.

# Implemention
Two algorithms, DQN and Double DQN ,with single and dueling network structure have been implemented.

## DQN algorithm

    Initialize replay memory D to capacity N
    
    Initialize Q_local and Q_target network for approximating action-value function with random weights θ_local and θ_target.
  
    for episode from 1 to M do
  
            reset the environment and get the state information s(t)
    
            while not done, do :
  
                    with probability epsilon, select a random action a(t).
                    oterwise select a(t) = ArgMax(Q_local(s(t),a)) 
          
                    execute a(t) to observe reward r(t), next state information s(t+1) and done or not
          
                    store ( s(t), a(t),r(t),s(t+1), done) into D 
          
                    if number of sample in D is greater than batch size, in every UPDATE_EVERY time steps do : 
                        
                            randomly generate mini batch of tuple ( s(t), a(t),r(t),s(t+1), done) from D
                                
                            if s(t+1) is a termination state, 
                                    
                                    set y = r(t)
                                        
                            else
                                
                                    a(t+1)=ArgMax(Q_target(s(t+1),a))
                                        
                                    set y = r(t)+Gamma*Q_target(s(t+1),a(t+1))
                                
                            y^ = Q_local(s(t),a(t))
                                
                            perform a gradient desecent step with learning rate LR on ( y-y^)^2 with respect to Q_local network parameters θ_local.
                                
                            perform a soft update to Q_target network parameters θ_target by formula:
                                
                                    θ_target = τ*θ_local + (1 - τ)*θ_target
                                
                        
                    if done, exit while loop
            
            set epsilon=Max(epsilon*epsilon_decay,epsilon_min)
            
    end for                    
                        
## Double DQN algorithm

Double DQN algoritm is almost the same as DQN algorithm. The only difference is when estimating a(t+1), it is using **ArgMax(Q_local(s(t+1),a))** instead of using Q_target in ArgMax.

## Hyperparamters

After exploring several combinations, values below for hyperparameters allows the agent to solve the problem in the least amount of time required.

Hyperparameter | Value
--- | ---    
Batch size | 32
Gamma | 1
τ | 1e-3
LR | 5e-4
UPDATE_EVERY | 4 


## Single network    
   
Single network structure consists of one input layer, 3 hidden layers and one output layer. Detail is shown in diagram below:
   
   
   ![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/network1.jpg)
    
## Dueling network

Dueling network consists of one input layer which is then shared by 2 networks. Each network consist of 3 hidden layers and one output layers. One network is outputting a single value V which represents the state value. The other one is outputing values which number equals to number of actions (4 for this problem). This represnts the advantage function value adv(i) = Q(i)-V for each action. Then the two output layers are linked up to form the final output layers by mathematical operation Q = V+(adv(i)- mean of adv)[https://arxiv.org/abs/1511.06581]. Detail is shown in diagram below:   


![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/deuling_net.jpg)

# Results

Below are the number of episodes needed to solve the environment and the evolution of rewards per episode during training for each algorithm with different network structures.

### DQN with sinlge network

Number of episodes needed to solve the environment : **1145**

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/q.png)

### Double DQN with sinlge network

Number of episodes needed to solve the environment : **864**

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/dq.png)

### DQN with dueling network

Number of episodes needed to solve the environment : **1064**

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/dueling_q.png)

### Double DQN with dueling network

Number of episodes needed to solve the environment : **1147**

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/dueling_dq.png)

Performance of each algorithms are compared in terms of 100 episodes average score in a 3000 episodes horizon.

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/compare_avg.png)

*Episode# in chart above is per 100 episodes*

It appears that DQN with dueling structure and Double DQN with single and dueling strcutres are generally performing better than DQN with single structure.

Below is an example showing how an agent trained by Double DQN behaves in the environment.

![alt text](https://github.com/kelvin84hk/deep-reinforcement-learning/blob/master/P1_Navigation/pics/dq.gif)


# Ideas for future work

The agent can be further improved by the following:

1. Implementing prioritized experience replay so that the agent can focus more on experience which has larger error.

2. Implementing Expected SARSA. For
    ```
    a(t+1)=ArgMax(Q_target(s(t+1),a))
    y= r(t)+Gamma*Q_target(s(t+1),a(t+1))
    ```
   instead of only taking account of the maximum Q value in the next state, we can take epsilon into account so that
     ```   
     a1(t+1)=ArgMax(Q_target(s(t+1),a))
     a2(t+1)=random(4)
     y=r(t)+Gamma*((1-epsilon)*Q_target(s(t+1),a1(t+1)) + epsilon*Q_target(s(t+1),a2(t+1)))
     ```
     This will change the algorithm from off-policy to on-policy which can help to stablize the results.

3. Implementing Rainbow Algorithm [https://arxiv.org/pdf/1710.02298.pdf] which combines good features from different algorithms to form an itegrated agent.
