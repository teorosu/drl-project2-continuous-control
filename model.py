
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes, seed):
        super(PolicyNetwork, self).__init__()
        torch.manual_seed(seed)

        self.linear = nn.ModuleList()
        sizes = [num_inputs] + hidden_sizes + [num_actions]
        for i,o in zip(sizes[:-1], sizes[1:]):
            self.linear.append(nn.Linear(i, o))

    def forward(self, state):
        x = state.float()

        # (-> stage -> relu ->)
        for stage in self.linear[:-1]:
            x = stage(x)
            x = F.relu(x)
        
        # (-> stage -> tanh ->)
        x = self.linear[-1](x)
        x = torch.tanh(x)
        return x

    def action(self, state):
        action = self.forward(state)
        return action.detach().cpu().numpy()

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_sizes, seed):
        super(ValueNetwork, self).__init__()
        torch.manual_seed(seed)

        self.linear = nn.ModuleList()
        sizes = [num_inputs + num_actions] + hidden_sizes + [1]
        for i,o in zip(sizes[:-1], sizes[1:]):
            self.linear.append(nn.Linear(i, o))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)

        # (-> stage -> relu ->)
        for stage in self.linear[:-1]:
            x = stage(x)
            x = F.relu(x)
        
        # (-> stage -> ) 
        x = self.linear[-1](x)
        return x