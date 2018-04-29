import logging
import math
import random

import numpy as np
import shutil
import torch
from pathlib import Path
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable as V
from torchvision import transforms as T

from itertools import count
from collections import namedtuple

from typing import Dict

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

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        self.fc1 = nn.Linear(32 * 2 * 2, 32 * 2 * 2)

        self.head = nn.Linear(32 * 2 * 2, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))

        output = self.head(x)

        return output


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, next_state, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(state, action, next_state, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class ModelHandler:
    def __init__(self, model_path: Path, num_actions=6):
        logging.getLogger().setLevel(logging.DEBUG)
        self.model_path = model_path
        self.num_actions = num_actions
        self.policy_net = DQN(num_actions)
        self.target_net = DQN(num_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        #  target_net is never being trained, it is slowly updated by copying the policy_net.
        #  .eval() sets mode for target_net from train to predict
        self.target_net.eval()
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            logging.info('Enabling CUDA')
            self.policy_net.cuda()
            self.target_net.cuda()

        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=0.00025, momentum=0.95)
        self.memory = ReplayMemory(1000000)

        self.BATCH_SIZE = 32
        self.GAMMA = 0.99
        self.EPS_START = 1.0
        self.EPS_END = 0.1
        self.EPS_DECAY = 1000000

        self.epoch = 0
        self.best_precision = 0

        self.eps = lambda steps: self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                 math.exp(-1 * steps / self.EPS_DECAY)
        self.steps_done = 0

    def set_mode(self, mode):
        mode_to_eps = {
            'train': lambda steps: self.EPS_END + (self.EPS_START - self.EPS_END) * \
                                   math.exp(-1 * steps / self.EPS_DECAY),
            'eval': lambda steps: -1
            }

        self.eps = mode_to_eps[mode]

    def select_action(self, state):
        if type(state) == np.ndarray:
            state = Tensor(state)
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
            # torch.unsqueeze_(state, 0)
        sample = random.random()
        eps_thresh = self.eps(self.steps_done)
        self.steps_done += 1

        action_type = 'Greedy' if sample > eps_thresh else 'Random'
        log_str = f'samp: {sample:6.2f} > {eps_thresh:6.2f} ? {action_type:7}'
        logging.debug(log_str)

        if sample > eps_thresh:
            actions = self.policy_net(
                    V(state, volatile=True).type(FloatTensor)
                    ).data.max(1)[1].view(1, 1)
            return actions
        else:
            return LongTensor([[random.randrange(self.num_actions)]])

    def update_policy_net(self):
        if len(self.memory) < self.BATCH_SIZE:
            logging.debug('Not enough samples in memory.')
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))
        non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)))

        non_final_next_states = V(torch.stack([s for s in batch.next_state if s is not None]), volatile=True)

        state_batch = V(torch.stack(batch.state))
        reward_batch = V(torch.stack(batch.reward))
        action_batch = V(torch.stack(batch.action))

        if self.use_cuda:
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            non_final_next_states = non_final_next_states.cuda()

        state_action_values = self.policy_net(state_batch.float()).gather(1, action_batch.long())

        next_state_values = V(torch.zeros(self.BATCH_SIZE).type(Tensor))
        next_state_values[non_final_mask] = self.target_net(non_final_next_states.float()).max(1)[0]
        expected_state_action_values = (next_state_values * self.GAMMA).view(-1, 1) + reward_batch
        expected_state_action_values = V(expected_state_action_values.data)

        loss = F.mse_loss(state_action_values, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def push_memory(self, state, action, next_state, reward):
        state_t = state
        action_t = Tensor([action])
        next_state_t = next_state if next_state is not None else None
        reward_t = Tensor([reward])
        self.memory.push(state_t, action_t, next_state_t, reward_t)

    def load_memory(self):
        file_name = 'replay_memory.mem'
        mem_path = str(self.model_path / file_name)
        try:
            memory = torch.load(str(mem_path))
        except FileNotFoundError:
            logging.info('No memory file to load from')
            return
        num_none = 0
        for i, trans in enumerate(memory['memory']):
            if i % 2000 == 0:
                print(f"Loading memory, {i} / {len(memory['memory'])}")
            if trans is None:
                num_none += 1
                print(f'None: {i}')
                continue
            self.memory.push(*trans)
            self.steps_done += 1

    def save_memory(self):
        file_name = 'replay_memory.mem'
        mem_path = str(self.model_path / file_name)
        memory = {
            "memory":       self.memory.memory,
            "num_elements": len(self.memory)
            }
        torch.save(memory, str(mem_path))
        logging.info(f'Saved replay memory to {str(mem_path)}, num_el: {len(self.memory)}')

    def save_model(self, ep):
        model_state = {
            'epoch':      ep,
            'state_dict': self.target_net.state_dict(),
            'optimizer':  self.optimizer.state_dict()
            }
        is_best = True
        self._save_checkpoint(model_state, is_best, filename=f'checkpoint-{ep}.mld')

        pass

    def _save_checkpoint(self, model_state: Dict, is_best: bool, filename='checkpoint.mdl'):
        (self.model_path / 'saved').mkdir(parents=True, exist_ok=True)
        model_path = str(self.model_path / 'saved' / filename)
        torch.save(model_state, model_path)
        if is_best:
            shutil.copyfile(model_path, str(self.model_path / 'model_best.mdl'))

    def load_best(self):
        file_path = self.model_path / 'model_best.mdl'
        if file_path.exists():
            print(f"=> loading checkpoint '{str(file_path)}'")
            checkpoint = torch.load(str(file_path))
            self.epoch = checkpoint['epoch']
            self.steps_done = checkpoint['epoch']

            self.target_net.load_state_dict(checkpoint['state_dict'])
            self.target_net.eval()
            self.policy_net.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.target_net.cuda()
            self.policy_net.cuda()

            print(f"=> loaded checkpoint {str(file_path)} (epoch {checkpoint['epoch']})")

        else:
            print(f"=> no checkpoint found at {str(file_path)}")


if __name__ == '__main__':
    mh = ModelHandler(None, num_actions='32')
