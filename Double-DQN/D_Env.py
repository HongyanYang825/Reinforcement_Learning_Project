import math
import tqdm
import numpy as np


class StockTrade(object):
    def __init__(self, data_set, start_date, start_balance, window, horizon):
        self.data_set = data_set
        self.start_date = start_date
        self.start_balance = start_balance
        self.window = window
        self.horizon = horizon
        self.data_list = self.get_data()
        
        # We define the action space
        self.action_space = {'buy': 1, 'hold': 0, 'sell': -1}
        self.action_names = ['buy', 'hold', 'sell']

        # We define other useful variables
        self.balance = None  # track the agent's account balance in one episode
        self.action = None  # track the agent's action
        self.t = 0  # track the current time step in one episode

    def get_data(self):
        data_list = []
        with open('data/' + self.data_set + '.csv', 'r') as file:
            lines = file.read().splitlines()
        for line in lines[1:]:
            data_list.append(float(line.split(",")[1]))
        return data_list

    def get_state(self, t):
        # t is the time stamp in the data_list
        if t - self.window >= -1:
            prices = self.data_list[(t-self.window+1): (t+1)]
        else: 
            prices = (-(t-self.window+1)*[self.data_list[0]]) + self.data_list[0: (t+1)]
        scaled_state = []
        for i in range(0, self.window - 1):
            scaled_state.append(1 / (1 + math.exp(prices[i]-prices[i+1])))
        return np.array([scaled_state])

    def reset(self):
        # We reset the agent's start date to a random valid trading day
        start_date = np.random.choice(range(self.window-1, len(self.data_list)-self.horizon), 1)[0]
        self.start_date = start_date
        # We reset the agent's balance to the start balance
        self.balance = self.start_balance
        # We reset the timeout tracker to be 0
        self.t = 0
        # We reset the agent's state
        state = self.get_state(self.start_date + self.t)
        # We set the information
        info = {}
        return state, info

    def step(self, action):
        '''
        Args:
            action (string): all feasible values are ['buy', 'hold', 'sell']
        '''
        # Get agent's current balance
        current_balance = self.balance
        # Convert the action name to operation
        act_arr = self.action_space[action]
        # Get today's and next day's stock open price
        time_stamp = self.start_date + self.t
        price_today = self.data_list[time_stamp]
        price_next_day = self.data_list[time_stamp+1]
        # Compute the reward
        reward = (price_next_day-price_today) * act_arr
        self.balance += reward
        # Check the termination
        if self.t == self.horizon - 1 or self.balance < 0:
            terminated = True
        else:
            terminated = False
        # Update the agent's next state, action and time step trackers
        next_state = self.get_state(time_stamp+1)
        self.action = action
        self.t += 1
        return next_state, reward, terminated, False, {}

def test():
    my_env = StockTrade("train_set", 200, 10, 50, 200)
    state, _ = my_env.reset()
    for i in range(1000):
        action = np.random.choice(list(my_env.action_space.keys()), 1)[0]
        next_state, reward, done, _, _ = my_env.step(action)
        #if i % 100 == 0:
            #print(next_state)
        #print(reward)
        #print(my_env.balance)
        if done:
            state, _ = my_env.reset()
            print("done!")
        else:
            state = next_state
