import sys
sys.path.append('recovery/DeepFTSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .DeepFTSrc.src.constants import *
from .DeepFTSrc.src.utils import *
from .DeepFTSrc.src.train import *

class DeepFTRecovery(Recovery):
    def __init__(self, hosts, env):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'framework'
        self.model_name = f'DeepFT_{self.env_name}_{hosts}'
        self.model_loaded = False

    def load_model(self):
        # Load training time series and thresholds
        self.train_time_data = normalize_time_data(np.load(data_folder + self.env_name + '/' + data_filename))
        self.thresholds = np.percentile(self.train_time_data, PERCENTILES, axis=0) 
        if self.env_name == 'simulator': self.thresholds *= percentile_multiplier
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model); self.model_loaded = True; exit()

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name); norm_series = self.train_time_data
        train_time_data, train_schedule_data, anomaly_data, class_data, thresholds = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            aloss, tloss, factor = backprop(self.epoch, self.model, self.optimizer, train_time_data, train_schedule_data, self.env.stats, norm_series, thresholds)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((aloss, tloss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_folder, f'{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def optimize_decision(self, state, original_decision):
        init = torch.tensor(deepcopy(original_decision), dtype=torch.double, requires_grad=True)
        optimizer = torch.optim.AdamW([init] , lr=0.8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        iteration = 0; equal = 0; z_old = 100; zs = []
        while iteration < 200:
            cpu_old = deepcopy(init.data[:,0:-self.hosts]); alloc_old = deepcopy(init.data[:,-self.hosts:])
            pred_state, prototypes = self.model(state, init)
            z, end = optimization_loss(pred_state.view(-1), prototypes, self.thresholds)
            optimizer.zero_grad(); z.backward(create_graph=True); optimizer.step(); scheduler.step()
            init.data = convertToOneHot(init.data, cpu_old, self.hosts)
            equal = equal + 1 if torch.all(alloc_old.eq(init.data[:,-self.hosts:])) else 0
            if equal > 30 or end: break
            iteration += 1; z_old = z.item()
        init.requires_grad = False 
        return init.data

    def get_data(self):
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return time_data, schedule_data

    def run_model(self, time_series, original_decision):
        if not self.model_loaded: self.load_model()
        state, schedule = self.get_data()
        result = self.optimize_decision(state, schedule)
        prev_alloc = {}
        for c in self.env.containerlist:
            oneHot = [0] * len(self.env.hostlist)
            if c: prev_alloc[c.id] = c.getHostID()
        decision = []
        for cid in prev_alloc:
            one_hot = result[cid, -self.hosts:].tolist()
            new_host = one_hot.index(max(one_hot))
            if prev_alloc[cid] != new_host: decision.append((cid, new_host))
        return decision

