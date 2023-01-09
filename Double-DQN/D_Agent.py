from D_DQN import *

import torch
import torch.nn as nn

import numpy as np


class Double_DQNAgent(object):
    def __init__(self, params):
        # save the parameters
        self.params = params
        # environment parameters
        self.action_dim = params['action_dim']
        self.obs_dim = params['observation_dim']
        # executable actions
        self.action_space = params['action_space']

        # create value network
        self.behavior_policy_net_1 = DeepQNet(input_dim=params['observation_dim'],
                                              num_hidden_layer=params['hidden_layer_num'],
                                              dim_hidden_layer=params['hidden_layer_dim'],
                                              output_dim=params['action_dim'])
        self.behavior_policy_net_2 = DeepQNet(input_dim=params['observation_dim'],
                                              num_hidden_layer=params['hidden_layer_num'],
                                              dim_hidden_layer=params['hidden_layer_dim'],
                                              output_dim=params['action_dim'])
        # create target network
        self.target_policy_net = DeepQNet(input_dim=params['observation_dim'],
                                          num_hidden_layer=params['hidden_layer_num'],
                                          dim_hidden_layer=params['hidden_layer_dim'],
                                          output_dim=params['action_dim'])

        # initialize target network with behavior network
        self.behavior_policy_net_1.apply(customized_weights_init)
        self.behavior_policy_net_2.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net_1.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device("cpu")
        self.behavior_policy_net_1.to(self.device)
        self.behavior_policy_net_2.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer_1 = torch.optim.Adam(self.behavior_policy_net_1.parameters(),
                                            lr=params['learning_rate'])
        self.optimizer_2 = torch.optim.Adam(self.behavior_policy_net_2.parameters(),
                                            lr=params['learning_rate'])
        
    def get_action(self, obs, eps):
        if np.random.random() < eps:  
            action = np.random.choice(self.action_space, 1)[0]
            return action
        else:  
            obs = self._arr_to_tensor(obs).view(1, -1)
            with torch.no_grad():
                q_values_1 = self.behavior_policy_net_1(obs)
                q_values_2 = self.behavior_policy_net_2(obs)
                q_values = q_values_1 + q_values_2
                action = q_values.max(dim=1)[1].item()
            return self.action_space[int(action)]

    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        obs_tensor = batch_data_tensor['obs']
        actions_tensor = batch_data_tensor['action']
        next_obs_tensor = batch_data_tensor['next_obs']
        rewards_tensor = batch_data_tensor['reward']
        dones_tensor = batch_data_tensor['done']
        actions_tensor = actions_tensor.reshape(actions_tensor.shape[0],1,1)

        if np.random.random() < 0.5:
            # compute the q value estimation using the behavior network
            predicted_targets = self.behavior_policy_net_1(obs_tensor).gather(2,actions_tensor)
            # compute the TD target using the target network
            with torch.no_grad():
                # Select argmax action
                _, actions = self.behavior_policy_net_1(obs_tensor).max(dim=1, keepdim=True)
                labels_next = self.behavior_policy_net_2(next_obs_tensor).gather(dim=1, index=actions)
            labels = rewards_tensor + (self.params['gamma']* labels_next*(1-dones_tensor))
            # compute the loss
            td_loss = nn.MSELoss()(predicted_targets, labels).to(self.device)
            # minimize the loss
            self.optimizer_1.zero_grad()
            td_loss.backward()
            self.optimizer_1.step()
        else:
            # compute the q value estimation using the behavior network
            predicted_targets = self.behavior_policy_net_2(obs_tensor).gather(2,actions_tensor)
            # compute the TD target using the target network
            with torch.no_grad():
                # Select argmax action
                _, actions = self.behavior_policy_net_2(obs_tensor).max(dim=1, keepdim=True)
                labels_next = self.behavior_policy_net_1(next_obs_tensor).gather(dim=1, index=actions)
            labels = rewards_tensor + (self.params['gamma']* labels_next*(1-dones_tensor))
            # compute the loss
            td_loss = nn.MSELoss()(predicted_targets, labels).to(self.device)
            # minimize the loss
            self.optimizer_2.zero_grad()
            td_loss.backward()
            self.optimizer_2.step()
        return td_loss.item()

    def update_target_policy(self):
        '''
        Copy the behavior policy network to the target network
        '''
        # hard update
        self.target_policy_net.load_state_dict(self.behavior_policy_net_1.state_dict())

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {'obs': [], 'action': [], 'reward': [], 'next_obs': [], 'done': []}
        # get the numpy arrays
        obs_arr, action_arr, reward_arr, next_obs_arr, done_arr = batch_data
        # convert to tensors
        batch_data_tensor['obs'] = torch.tensor(obs_arr, dtype=torch.float32).to(self.device)
        #batch_data_tensor['action'] = torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        batch_data_tensor['action'] = torch.tensor(action_arr, dtype=torch.int64).view(-1, 1).to(self.device)
        batch_data_tensor['reward'] = torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        batch_data_tensor['next_obs'] = torch.tensor(next_obs_arr, dtype=torch.float32).to(self.device)
        batch_data_tensor['done'] = torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        return batch_data_tensor
