from experience import ReplayBuffer
from model import PolicyNetwork, ValueNetwork
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

LR = 1e-3               # learning rate
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DDPGAgent:
    def __init__(self, state_size, action_size):
        # Configs
        self.state_size = state_size
        self.action_size = action_size

        # Experience memory
        self.memory = ReplayBuffer(capacity=BUFFER_SIZE)

        # Init the policy network(s)
        self.policy_network = PolicyNetwork(state_size, action_size, hidden_sizes=[256, 128], seed=0).to(DEVICE)
        self.target_policy_network = PolicyNetwork(state_size, action_size, hidden_sizes=[256, 128], seed=0).to(DEVICE)
        self.soft_update(self.policy_network, self.target_policy_network, 1)

        # Init the value network(s)
        self.value_network = ValueNetwork(state_size, action_size, hidden_sizes=[256, 128], seed=0).to(DEVICE)
        self.target_value_network = ValueNetwork(state_size, action_size, hidden_sizes=[256, 128], seed=0).to(DEVICE)
        self.soft_update(self.policy_network, self.target_policy_network, 1)

        # Network optimizers
        self.value_optimizer  = optim.Adam(self.value_network.parameters(),  lr=LR)
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=LR)

    def act(self, state):
        state = torch.from_numpy(state).float().to(DEVICE)

        self.policy_network.eval()
        with torch.no_grad():
            action = self.policy_network.action(state).squeeze()
        self.policy_network.train()
        
        return action

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample(BATCH_SIZE)
            self.learn(experiences, GAMMA)

    def learn(self, experiences, gamma):
        # Sample experiences
        states, actions, rewards, next_states = experiences

        # Actor. Policy loss function
        policy_actions = self.policy_network(states)
        policy_loss = -torch.mean(self.value_network(states.detach(), policy_actions))

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Critic. Value loss function
        policy_next_actions = self.target_policy_network(next_states).squeeze()
        q_states = self.value_network(states, actions)
        expected_q_states = rewards + gamma * self.target_value_network(next_states.detach(), policy_next_actions.detach())
        value_loss = F.mse_loss(expected_q_states.squeeze(), q_states.squeeze())

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.policy_network, self.target_policy_network, TAU) 
        self.soft_update(self.value_network, self.target_value_network, TAU) 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
