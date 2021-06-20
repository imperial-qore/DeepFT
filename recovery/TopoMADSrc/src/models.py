import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

class TopoMAD_simulator_16(nn.Module):
	def __init__(self):
		super(TopoMAD_simulator_16, self).__init__()
		self.name = 'TopoMAD_simulator_16'
		self.lr = 0.002
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 1 # w_size = 3
		self.n_latent = self.n_feats # 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.state_decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 3), nn.Sigmoid(),
		)

	def encode(self, t):
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def state_decode(self, t):
		states = []
		for elem in t:
			states.append(self.state_decoder(elem))	
		return torch.stack(states)

	def forward(self, t):
		t = self.encode(t)
		states = self.state_decode(t)
		return states

class TopoMAD_framework_16(nn.Module):
	def __init__(self):
		super(TopoMAD_framework_16, self).__init__()
		self.name = 'TopoMAD_framework_16'
		self.lr = 0.002
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 1 # w_size = 3
		self.n_latent = self.n_feats # 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.state_decoder = nn.Sequential(
			nn.Linear(self.n_hosts * self.n_latent, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_feats), nn.Sigmoid(),
		)

	def encode(self, t):
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def state_decode(self, t):
		return self.state_decoder(t.view(-1)).view(self.n_hosts, 3)	

	def forward(self, t):
		t = self.encode(t)
		states = self.state_decode(t)
		return states