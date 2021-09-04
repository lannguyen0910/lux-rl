import random
import torch.optim as optim

from loss import compute_loss
from memory import ReplayBuffer
from collections import defaultdict
from model import *

class ActorAgent():
    def __init__(self, cfg, in_size):
        self.in_size = in_size
        self.cfg = cfg
        self.out_worker = cfg.out_worker
        self.out_ctiles = cfg.out_ctiles
        self.seed = random.seed(cfg.random_seed)
        self.device = cfg.device

        self.worker_model = Actor(
            in_size=in_size, out_size=cfg.out_worker, seed=cfg.random_seed).to(self.device)
        self.worker_model_optimizer = optim.Adam(
            self.worker_model.parameters(), lr=cfg.LR_ACTOR)
        self.worker_model_scheduler = optim.lr_scheduler.ExponentialLR(
            self.worker_model_optimizer, gamma=cfg.LR_DECAY)

        self.target_worker_model = Actor(
            in_size=in_size, out_size=cfg.out_worker, seed=cfg.random_seed).to(self.device)
        self.target_worker_model_optimizer = optim.Adam(
            self.worker_model.parameters(), lr=cfg.LR_ACTOR)
        self.target_worker_model_scheduler = optim.lr_scheduler.ExponentialLR(
            self.worker_model_optimizer, gamma=cfg.LR_DECAY)

        self.target_worker_model.load_state_dict(
            self.worker_model.state_dict())

        self.ctiles_model = Actor(
            in_size=in_size, out_size=cfg.out_ctiles, seed=cfg.random_seed).to(self.device)
        self.ctiles_model_optimizer = optim.Adam(
            self.ctiles_model.parameters(), lr=cfg.LR_ACTOR)
        self.ctiles_model_scheduler = optim.lr_scheduler.ExponentialLR(
            self.ctiles_model_optimizer, gamma=cfg.LR_DECAY)

        self.target_ctiles_model = Actor(
            in_size=in_size, out_size=cfg.out_ctiles, seed=cfg.random_seed).to(self.device)
        self.target_ctiles_model_optimizer = optim.Adam(
            self.ctiles_model.parameters(), lr=cfg.LR_ACTOR)
        self.target_ctiles_model_scheduler = optim.lr_scheduler.ExponentialLR(
            self.ctiles_model_optimizer, gamma=cfg.LR_DECAY)

        self.target_ctiles_model.load_state_dict(
            self.ctiles_model.state_dict())

        self.worker_memory = ReplayBuffer(
            int(1e6), cfg.BATCH_SIZE, cfg.random_seed)
        self.ctiles_memory = ReplayBuffer(
            int(1e4), cfg.BATCH_SIZE, cfg.random_seed)

        self.target_update_counter = 0

    def learnworker(self, states, target):
        states = states.to(self.device)
        target = target.to(self.device)
        pred = self.worker_model(states)

        # print(pred.shape)
        # print(target.shape)
        
        metrics = defaultdict(float)

        loss = compute_loss(pred, target, metrics)

        self.worker_model_optimizer.zero_grad()
        loss.backward()
        self.worker_model_optimizer.step()

        return metrics

    def learnctiles(self, states, target):
        states = states.to(self.device)
        target = target.to(self.device)

        pred = self.ctiles_model(states)

        metrics = defaultdict(float)
        loss = compute_loss(pred, target, metrics)

        self.ctiles_model_optimizer.zero_grad()
        loss.backward()
        self.ctiles_model_optimizer.step()
        return metrics

    def act(self, state, is_worker=True):
        state = torch.from_numpy(state).float().to(self.device)
        if is_worker:
            self.worker_model.eval()
            with torch.no_grad():
                out = self.worker_model(state)
                out = out.cpu().data.numpy()
            self.worker_model.train()
        else:
            self.ctiles_model.eval()
            with torch.no_grad():
                out = self.ctiles_model(state)
                out = out.cpu().data.numpy()
            self.ctiles_model.train()
        return out

    def add(self, state, action, new_state, reward, done, is_worker=True):
        if is_worker:
            self.worker_memory.add(state, action, new_state, reward, done)
        else:
            self.ctiles_memory.add(state, action, new_state, reward, done)

    def lr_step(self):
        self.worker_model_scheduler.step()
        self.ctiles_model_scheduler.step()

    def step(self):
        worker_metric_mean = defaultdict(float)
        ctiles_metric_mean = defaultdict(float)
        if len(self.worker_memory) > 20_000:
            for _ in range(100):
                experiences = self.worker_memory.sample()
                worker_metrics = self.learnworker(experiences)
                for key, val in worker_metrics.items():
                    worker_metric_mean[key] += val
            for key, val in worker_metric_mean.items():
                worker_metric_mean[key] = val / 1000
        if len(self.ctiles_memory) > 20_000:
            for _ in range(100):
                experiences = self.ctiles_memory.sample()
                ctiles_metrics = self.learnctiles(experiences)
                for key, val in ctiles_metrics.items():
                    ctiles_metric_mean[key] += val

            for key, val in ctiles_metric_mean.items():
                ctiles_metric_mean[key] = val / 100

        return worker_metric_mean, ctiles_metric_mean

    def train(self, terminal_state):
        if len(self.worker_memory) < 20_000 or len(self.ctiles_memory) < 10_000:
            return
        worker_minibatch = self.worker_memory.sample()
        ctiles_minibatch = self.ctiles_memory.sample()

        # print(ctiles_minibatch)
        #  transition: (states1, actions, states2, rewards, dones)
        current_worker_states = worker_minibatch[0].to(self.device)
        current_worker_qs_list = self.worker_model(
            current_worker_states).to(self.device)
        new_current_worker_state = worker_minibatch[2].to(self.device)
        future_qs_worker_list = self.target_worker_model(
            new_current_worker_state).to(self.device)

        current_ctiles_states = ctiles_minibatch[0].to(self.device)
        current_ctiles_qs_list = self.ctiles_model(
            current_ctiles_states).to(self.device)
        new_current_ctiles_state = ctiles_minibatch[2].to(self.device)
        future_qs_ctiles_list = self.target_ctiles_model(
            new_current_ctiles_state).to(self.device)

        X_worker = torch.tensor([]).to(self.device)
        y_worker = torch.tensor([]).to(self.device)

        for index in range(len(worker_minibatch[0])):
            state = worker_minibatch[0][index]
            action = int(worker_minibatch[1][index])
            new_state = worker_minibatch[2][index]
            reward = worker_minibatch[3][index]
            done = worker_minibatch[4][index]

            if not done:
                max_future_worker_q = torch.max(future_qs_worker_list[index])
                new_q_worker = reward + self.cfg.DISCOUNT * max_future_worker_q
            else:
                new_q_worker = reward
            current_qs = current_worker_qs_list[index]
            current_qs[action] = new_q_worker

            X_worker = torch.cat((X_worker, torch.unsqueeze(state, 0)))
            y_worker = torch.cat((y_worker, torch.unsqueeze(current_qs, 0)))

        X_ctiles = torch.tensor([]).to(self.device)
        y_ctiles = torch.tensor([]).to(self.device)

        for index in range(len(ctiles_minibatch)):
            state = ctiles_minibatch[0][index]
            action = int(ctiles_minibatch[1][index])
            new_state = ctiles_minibatch[2][index]
            reward = ctiles_minibatch[3][index]
            done = ctiles_minibatch[4][index]

            if not done:
                max_future_ctiles_q = torch.max(future_qs_ctiles_list[index])
                new_q_ctiles = reward + self.cfg.DISCOUNT * max_future_ctiles_q
            else:
                new_q_ctiles = reward
            current_qs = current_ctiles_qs_list[index]
            if action > 2:
                action = 2
            current_qs[action] = new_q_ctiles

            X_ctiles = torch.cat((X_ctiles, torch.unsqueeze(state, 0)))
            y_ctiles = torch.cat((y_ctiles, torch.unsqueeze(current_qs, 0)))

        self.learnworker(X_worker, y_worker)
        self.learnctiles(X_ctiles, y_ctiles)
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > self.cfg.UPDATE_TARGET_EVERY:
            self.target_worker_model.load_state_dict(
                self.worker_model.state_dict())
            self.target_ctiles_model.load_state_dict(
                self.ctiles_model.state_dict())
            self.target_update_counter = 0
