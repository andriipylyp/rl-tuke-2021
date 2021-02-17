import numpy as np
import gym 
import time

class Agent:
    '''
    Agent class made for the Reinforcement Learning course TUKE 2021
    to test qlearning algorithms on the Taxi-v3 environment.
    
    q_table_shape - shape of the q_table
    exploration_rate - exploration rate on the begining of the training procedure
    decay_rate - rate of exploration_rate reduction for training episode
    learning_rate - learning rate for bellman equation
    discount_rate - discount for the target mesurenment in bellman equation
    env - evironment

    '''
    def __init__(self, exploration_rate = 1, decay_rate = 0.01, learning_rate = 0.6, discount_rate = 0.6):
        self.env = gym.make('Taxi-v3')
        self.table_shape = (self.env.observation_space.n, self.env.action_space.n)
        self.q_tables = {
            'dq': np.zeros((2,self.table_shape[0],self.table_shape[1])), 
            'q': np.zeros(self.table_shape), 
            'sarsa': np.zeros(self.table_shape)
            }
        self.exploration_rate = exploration_rate
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.discount_rate = discount_rate
        
    
    def getScore(self, table, steps):
        score = 0
        env_s = gym.make('Taxi-v3')
        state = env_s.reset()
        for step in range(steps):
            action = np.argmax(table[state,:])

            new_state, reward, isDone, _ = env_s.step(action)

            state = new_state
            score += reward
            if isDone == True:
                break
        return score


    def qTraining(self, episodes_num, episode_size):
        self.exploration_rate = 1
        q_table = np.zeros(self.table_shape)
        results = []
        start_time = time.time()
        for episode in range(episodes_num):
            state = self.env.reset()
            for step in range(episode_size):
                if np.random.random() < self.exploration_rate:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(q_table[state,:])
                    
                new_state, reward, isDone, _ = self.env.step(action)
                
                #bellman equation
                q_table[state, action] = q_table[state, action] + self.learning_rate * (reward + self.discount_rate * 
                                                                np.max(q_table[new_state,:]) - q_table[state,action])
                
                state = new_state

                if isDone == True:
                    break

            results.append(self.getScore(q_table, episode_size))
            
            self.exploration_rate = self.exploration_rate - 0.001*np.exp(-self.decay_rate*episode)
        self.q_tables['q'] = q_table
        return (time.time() - start_time, results)

    def sarsaTraining(self, episodes_num, episode_size):
        self.exploration_rate = 1
        q_table = np.zeros(self.table_shape)
        results = []
        start_time = time.time()
        for episode in range(episodes_num):
            state = self.env.reset()
            for step in range(episode_size):
                #a
                if np.random.random() < self.exploration_rate:
                    action = self.env.action_space.sample()
                    
                new_state, reward, isDone, _ = self.env.step(action)
                #a'
                new_action = np.argmax(q_table[new_state,:])
                
                #bellman equation but target policy is always same as Behavior policy
                q_table[state, action] = q_table[state, action] + self.learning_rate * (reward + self.discount_rate * q_table[new_state,new_action] - q_table[state,action])
                
                state = new_state
                action = new_action
                if isDone == True:
                    break
            results.append(self.getScore(q_table, episode_size))
            
            self.exploration_rate = self.exploration_rate - 0.001*np.exp(-self.decay_rate*episode)
        self.q_tables['sarsa'] = q_table
        return (time.time() - start_time, results)

    def doubleQTraining(self, episodes_num, episode_size):
        self.exploration_rate = 1
        dq_table1 = np.zeros(self.table_shape)
        dq_table2 = np.zeros(self.table_shape)
        results = []
        start_time = time.time()
        for episode in range(episodes_num):
            state = self.env.reset()
            for step in range(episode_size):
                table = 1 if np.random.random() > 0.5 else 2
                if table == 1:
                    #a
                    if np.random.random() < self.exploration_rate:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(dq_table1[state,:])
                    new_state, reward, isDone, _ = self.env.step(action)
                    #a'
                    new_action = np.argmax(dq_table1[new_state,:])
                    
                    dq_table1[state, action] = dq_table1[state, action] + self.learning_rate * (reward + self.discount_rate * 
                                                                dq_table2[new_state,new_action] - dq_table1[state,action])
                elif table == 2:
                    #a
                    if np.random.random() < self.exploration_rate:
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(dq_table2[state,:])
                    new_state, reward, isDone, _ = self.env.step(action)
                    #a'
                    new_action = np.argmax(dq_table2[new_state,:])
                    
                    dq_table2[state, action] = dq_table2[state, action] + self.learning_rate * (reward + self.discount_rate * 
                                                                dq_table1[new_state,new_action] - dq_table2[state,action])

                state = new_state
                if isDone == True:
                    break
            results.append(self.getScore(dq_table2, episode_size))
            
            self.exploration_rate = self.exploration_rate - 0.001*np.exp(-self.decay_rate*episode)
        self.q_tables['dq'][0] = dq_table1
        self.q_tables['dq'][1] = dq_table2
        return (time.time() - start_time, results)