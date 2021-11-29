import numpy as np
import torch 
import torch.nn as nn
from torch.nn.modules.loss import MSELoss
from torch.optim import RMSprop, Adam
from tensorboardX import SummaryWriter
import os

class DeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 lr=0.01,
                 reward_decay=0.95,
                 epsilon=0.96,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 train_epochs=10,
                 epsilon_increment=None,
                 use_cuda=True,
                 gsize = 8,
                 last_learn_step = 0,
                 logdir= None,
                 modeldir = 'data',
                 ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = lr
        self.gamma = reward_decay
        self.epsilon_max = epsilon
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = epsilon_increment
        self.epsilon = 0 if epsilon_increment is not None else self.epsilon_max
        self.train_epochs = train_epochs
        self.use_cuda=use_cuda
        self.learn_step_counter = last_learn_step
        self.gsize = gsize
        self._build_net_cnn()
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        if logdir != None:
            self.writer = SummaryWriter(logdir)
        else:
            self.writer = None
        self.modeldir = modeldir
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
    #def preprocess_state(self, state):  # 预处理
    #    return np.log(state + 1) / 16

    def choose_action(self, state, det=False):
        istate = state.reshape([2, self.gsize, self.gsize])
        istate = istate[np.newaxis, :, :]
        #state = self.preprocess_state(state)
        #print(istate.shape)
        if det or np.random.uniform() < self.epsilon:
            action_value = self.q_eval_model(torch.Tensor(istate).to('cuda' if self.use_cuda else 'cpu'))
            action_value = np.squeeze(action_value.detach().cpu().numpy())
            smin = np.min(action_value)
            #print(len(action_value))
            action_value = [smin - 10 if state[i] != 0 or state[i + self.gsize ** 2] != 0 else action_value[i] for i in range(self.gsize ** 2)]
            #print(np.sum([action_value == smin - 10]))
            action_index = np.argmax(action_value)
            if state[action_index] != 0 :
                print('Excuse me?')
                print(action_value[action_index] == smin - 10)
                print(np.sum([action_value == smin - 10]))
        else:
            #print('Random')
            action_map = [-1 if state[i] != 0 or state[i + self.gsize ** 2] != 0 else np.random.random() for i in range(self.gsize ** 2)]
            action_index = np.argmax(action_map)
        return action_index

    def _build_net_cnn(self):
        self.q_eval_model = nn.Sequential(
            nn.Conv2d(2, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),

            nn.Flatten(),
            nn.Linear(256 * (self.gsize - 3) ** 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.gsize ** 2)).to('cuda' if self.use_cuda else 'cpu')

        self.q_target_model = nn.Sequential(
            nn.Conv2d(2, 64, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2, 2), 1),

            nn.Flatten(),
            nn.Linear(256 * (self.gsize - 3) ** 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.gsize ** 2)).to('cuda' if self.use_cuda else 'cpu')
        
        self.opt = Adam(self.q_eval_model.parameters(), lr=self.lr)
        self.loss = MSELoss()
    
    def _fit(self, model, opt, loss, input, output, epochs):
        output = output.detach()
        for _ in range(epochs):
            pred = model(input)
            ploss = loss(pred, output)
            #if torch.sum(output == 1) != 0:
                #importantloss = torch.sum(((pred - output) ** 2 * output ** 2)) / torch.sum(output != 0)
                # vloss = ploss + importantloss * 10
            # else:
            #     vloss = ploss
            opt.zero_grad()
            ploss.backward()
            opt.step()
        #importantloss = torch.sum(((pred - output) ** 2 * output ** 2)) / torch.sum(output == 1)
        if self.writer:
            self.writer.add_scalar('loss', ploss.item(), self.learn_step_counter)
            if torch.sum(output == 1) + torch.sum(output == -1) != 0:
                cond = output ** 2 == 1.0
                importantloss = torch.sum(torch.where(cond, (pred - output) ** 2, torch.zeros_like(output))) / (torch.sum(output == 1) + torch.sum(output == -1))
                self.writer.add_scalar('important', importantloss.item(), self.learn_step_counter)
        #print('Learnt')
        #print('loss: %.6f' %(ploss))

    def target_replace_op(self):
        self.q_target_model.load_state_dict(self.q_eval_model.state_dict())

    def store_memory(self, s, s_, a, r):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        memory = np.hstack((s, s_, [a, r]))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = memory
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_replace_op()
            #print('target_params_replaced!')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]
        n_features = self.n_features
        s = torch.Tensor(batch_memory[:, 0:n_features].reshape([-1, 2, self.gsize, self.gsize])).to('cuda' if self.use_cuda else 'cpu')
        s_ = torch.Tensor(batch_memory[:, n_features:n_features*2].reshape([-1, 2, self.gsize, self.gsize])).to('cuda' if self.use_cuda else 'cpu')
        a = torch.Tensor(batch_memory[:, n_features*2]).long().to('cuda' if self.use_cuda else 'cpu')
        r = torch.Tensor(batch_memory[:, n_features*2+1]).to('cuda' if self.use_cuda else 'cpu')

        #print(torch.sum(r == 1), torch.sum(r == 0))
        #q_next = self.q_target_model(s_)
        q_eval = self.q_eval_model(s)

        q_target = q_eval.clone()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        #print(q_next.shape)
        #print((q_next == None).any())
        #v = torch.max(q_next, dim=1)
        #print(v)
        #print(a)
        q_target[batch_index, a] = torch.where(r != 0, r, self.gamma * torch.max(self.q_target_model(s_), dim=1)[0])
        
        #self.q_eval_model.fit(s, q_target, epochs=self.train_epochs, verbose=0)
        self._fit(self.q_eval_model, self.opt, self.loss, s, q_target, self.train_epochs)
        self.learn_step_counter += 1

    def save_model(self, episode):
        #self.q_eval_model.save('dqn2048_cnn-{}.h5'.format(episode))
        torch.save(self.q_eval_model.state_dict(), '{}/gomoku-{}.h5'.format(self.modeldir, episode))
    
    def load_model(self, episode):
        self.q_eval_model.load_state_dict(torch.load('{}/gomoku-{}.h5'.format(self.modeldir, episode), map_location=lambda a, b: a if self.use_cuda==False else None))