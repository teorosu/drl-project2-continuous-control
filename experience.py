from collections import deque
import numpy as np
import torch
import random

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self, states, actions, rewards, next_states):
        # py list to np.array
        rewards = np.array(rewards)[:, np.newaxis]

        # Fill details for new entry
        entry = {}
        entry['state'] = states
        entry['action'] = actions
        entry['reward'] = rewards
        entry['next_state'] = next_states

        self.memory.append(entry)
    
    def sample(self, batch_size):
        # Sample experiences
        experiences = random.sample(self.memory, k=batch_size)

        # Generate tensor
        states = torch.from_numpy(np.vstack([e['state'] for e in experiences])).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack([e['action'] for e in experiences])).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack([e['reward'] for e in experiences])).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack([e['next_state'] for e in experiences])).float().to(DEVICE)

        # Return results
        return (states, actions, rewards, next_states)
    
    def __len__(self):
        return len(self.memory)