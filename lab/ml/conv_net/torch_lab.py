import math
import random

import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms as T

from itertools import count
from collections import namedtuple

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
DoubleTensor = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
Tensor = FloatTensor
print(f'use_cuda: {use_cuda}')

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.head = nn.Linear(32*8*8, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


BATCH_SIZE = 4
GAMMA = 0.99
EPS_START = 0.9
EPS_END=0.1
EPS_DECAY = 200

qnet = DQN(6)
qnet.cuda()

target_qnet = DQN(6)
target_qnet.load_state_dict(qnet.state_dict())
target_qnet.eval()
target_qnet.cuda()


optimizer = optim.RMSprop(qnet.parameters())
memory = ReplayMemory(1000000)

steps_done = 0
def select_action(state):
    global steps_done
    sample = random.random()
    eps_thresh = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_thresh:
        return qnet(
                V(state, volatile=True).type(FloatTensor)
                ).data.max(1)[1].view(1,1)
    else:
        return LongTensor([[random.randrange(6)]])


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    batch = Transition(*zip(*transitions))
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    non_final_next_states = V(torch.cat([s for s in batch.next_state if s is not None]), volatile=True)

    state_batch = V(torch.cat(batch.state))
    action_batch = V(torch.cat(batch.action))
    reward_batch = V(torch.cat(batch.reward))

    state_action_values = qnet(state_batch).gather(1, action_batch)

    next_state_values = V(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = target_qnet(non_final_next_states).max(1)[0]
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    expected_state_action_values = V(expected_state_action_values.data)

    loss = F.mse_loss(state_action_values, expected_state_action_values)
    optimizer.zero_grad()
    loss.backward()
    for param in qnet.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def main():
    num_ep = 2
    for i_ep in range(num_ep):
        data = torch.from_numpy(np.zeros((1, 4, 64, 64))).type(FloatTensor)

    for t in count():
        action = select_action(data)

        reward, done = (torch.from_numpy(np.array([random.random()])).type(FloatTensor), (random.random() > 0.5))
        memory.push(data, action, data, reward)
        print(reward, done)
        if t > 12:
            break
        optimize_model()
    pass

if __name__ == '__main__':
    main()