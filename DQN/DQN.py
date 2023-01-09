import torch
import torch.nn as nn

import numpy as np


def customized_weights_init(m):
    '''
    Customized uniform weight initialization
    '''
    # compute the gain
    gain = nn.init.calculate_gain('relu')
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)

class DeepQNet(nn.Module):
    def __init__(self, input_dim, num_hidden_layer, dim_hidden_layer, output_dim):
        super(DeepQNet, self).__init__()
        
        # define the input dimension
        self.input_dim = input_dim
        # define the number of the hidden layers
        self.hidden_num = num_hidden_layer
        # define the hidden dimension
        self.hidden_dim = dim_hidden_layer
        # define the output dimension
        self.output_dim = output_dim

        # define the input linear layer here
        self.input = nn.Linear(self.input_dim, self.hidden_dim)
        # define the activation function after the input layer
        self.fc0 = nn.ReLU()
        # define the first hidden layer here
        self.hl0 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # define the activation function after the first hidden layer
        self.fc1 = nn.ReLU()
        # define the second hidden layer here
        self.hl1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # define the activation function after the second hidden layer
        self.fc2 = nn.ReLU()
        # define the output layer here
        self.output = nn.Linear(self.hidden_dim, self.output_dim)        

    def forward(self, x):
        '''
        Implement the forward propagation
        '''
        # forward x through the input layer
        x = self.input(x)
        # apply activation
        x = self.fc0(x)
        # forward x throught the first hidden layer
        x = self.hl0(x)
        # apply activation
        x = self.fc1(x)
        # forward x throught the second hidden layer
        x = self.hl1(x)
        # apply activation
        x = self.fc2(x)
        # forward x throught the output layer
        y = self.output(x)
        return y

class ReplayBuffer(object):
    '''
    Implement the Replay Buffer as a class, which contains:
        - self._data_buffer (list): a list variable to store all transition tuples
        - add: a function to add new transition tuple into the buffer
        - sample_batch: a function to sample a batch training data from the buffer
    '''
    def __init__(self, buffer_size):
        '''
        Args:
            buffer_size (int): size of the replay buffer
        '''
        # total size of the replay buffer
        self.total_size = buffer_size
        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, obs, act, reward, next_obs, done):
        # create a tuple
        trans = (obs, act, reward, next_obs, done)
        # add the tuple to update the replay buffer
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trans)
        else:
            self._data_buffer[self._next_idx] = trans
        # calculate the next index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        '''
        Function to fetch the state, action, reward, next state, and done arrays
        Args:
            indices (list): list contains the index of all sampled transition tuples
        '''
        # lists for transitions
        obs_list, actions_list, rewards_list, next_obs_list, dones_list = [], [], [], [], []
        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            obs_list.append(np.array(obs, copy=False))
            actions_list.append(np.array(act, copy=False))
            rewards_list.append(np.array(reward, copy=False))
            next_obs_list.append(np.array(next_obs, copy=False))
            dones_list.append(np.array(d, copy=False))
        # return the sampled batch data as numpy arrays
        return (np.array(obs_list), np.array(actions_list), np.array(rewards_list),
                np.array(next_obs_list), np.array(dones_list))

    def sample_batch(self, batch_size):
        '''
        Args:
            batch_size (int): size of the sampled batch data.
        '''
        # sample indices with replaced
        indices = [np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)]
        return self._encode_sample(indices)

class LinearSchedule(object):
    '''
    Customized decay epsilon values
    '''
    def __init__(self, start_value, end_value, duration):
        # start value
        self._start_value = start_value
        # end value
        self._end_value = end_value
        # time steps that value changes from the start value to the end value
        self._duration = duration
        # difference between the start value and the end value
        self._schedule_amount = end_value - start_value

    def get_value(self, time):
        '''
        Get decay epsilon values
        '''
        # if time > duration, use the end value, else use the scheduled value
        if time > self._duration:
            return self._end_value
        else:
            return ((time/self._duration)*self._schedule_amount) + self._start_value
