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
        self.model_name = f'DeepFT_{hosts}'
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.load_model()

    def load_model(self):
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model)
        self.train_time_data = load_npyfile(os.path.join(data_folder, self.env_name), data_filename)

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name)
        train_time_data, train_schedule_data, anomaly_data, class_data = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            loss, factor = backprop(self.epoch, self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.optimizer)
            anomaly_score, class_score = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tFactor = {factor},\tAScore = {anomaly_score},\tCScore = {class_score}')
            self.accuracy_list.append((loss, factor, anomaly_score, class_score))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_folder, f'{self.env_name}_{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def recover_decision(self, embedding, schedule_data, original_decision):
        new_schedule_data = self.gen(embedding, schedule_data)
        probs = self.disc(schedule_data, new_schedule_data)
        self.gan_plotter.new_better(probs[1] >= probs[0])
        if probs[0] > probs[1]: # original better
            return original_decision
        # Form new decision
        host_alloc = []; container_alloc = [-1] * len(self.env.hostlist)
        for i in range(len(self.env.hostlist)): host_alloc.append([])
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                host_alloc[c.getHostID()].append(c.id) 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision); hosts_from = [0] * self.hosts
        for cid in np.concatenate(host_alloc):
            cid = int(cid)
            one_hot = schedule_data[cid].tolist()
            new_host = one_hot.index(max(one_hot))
            if container_alloc[cid] != new_host: 
                decision_dict[cid] = new_host
                hosts_from[container_alloc[cid]] = 1
        self.gan_plotter.plot_test(hosts_from)
        return list(decision_dict.items())

    def run_encoder(self, schedule_data):
        # Get latest data from Stat
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return self.model(time_data, schedule_data)

    def run_model(self, time_series, original_decision):
        # Run encoder
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        anomaly, prototype = self.run_encoder(schedule_data)
        # If no anomaly predicted, return original decision 
        for a in anomaly:
            prediction = torch.argmax(a).item() 
            if prediction == 1: 
                self.gan_plotter.update_anomaly_detected(1)
                break
        else:
            self.gan_plotter.update_anomaly_detected(0)
            return original_decision
        # Form prototype vectors for diagnosed hosts
        embedding = [torch.zeros_like(p) if torch.argmax(anomaly[i]).item() == 0 else p for i, p in enumerate(prototype)]
        self.gan_plotter.update_class_detected(get_classes(embedding, self.model))
        embedding = torch.stack(embedding)
        # Pass through GAN
        if self.training:
            self.train_gan(embedding, schedule_data)
            return original_decision
        return self.recover_decision(embedding, schedule_data, original_decision)

