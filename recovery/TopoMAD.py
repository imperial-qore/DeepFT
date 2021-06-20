import sys
sys.path.append('recovery/TopoMADSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .TopoMADSrc.src.constants import *
from .TopoMADSrc.src.utils import *
from .TopoMADSrc.src.train import *

class TopoMADRecovery(Recovery):
    def __init__(self, hosts, env):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.model_name = f'TopoMAD_{self.env_name}_{hosts}'
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
        freeze(self.model); self.model_loaded = True

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name); norm_series = self.train_time_data
        train_time_data, train_schedule_data, anomaly_data, class_data, thresholds = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            aloss = backprop(self.epoch, self.model, self.optimizer, train_time_data, train_schedule_data, self.env.stats, norm_series, thresholds)
            anomaly_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tAScore = {anomaly_score}')
            self.accuracy_list.append((aloss, anomaly_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_folder, f'{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def recover_decision(self, state, original_decision):
        pred_state = self.model(state)
        anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), self.thresholds)
        host_selection = []
        for hid, anomaly in enumerate(anomaly_any_dim[0]):
            if anomaly: host_selection.append(hid)
        if host_selection == []:
            return original_decision
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.env.scheduler.FirstFitPlacement(container_selection)
        container_alloc = [-1] * len(self.env.hostlist)
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision)
        for cid, hid in target_selection:
            if container_alloc[cid] != hid:
                decision_dict[cid] = hid
        return list(decision_dict.items())

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
        return self.recover_decision(state, original_decision)

