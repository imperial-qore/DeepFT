import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

## Simple Multi-Head Self-Attention Model
class DeepFT_16(nn.Module):
	def __init__(self):
		super(DeepFT_16, self).__init__()
		self.name = 'DeepFT_16'
		self.lr = 0.001
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
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(K+1)]
		# 0th : no anomaly, rest: kth anomaly

	def encode(self, t, s):
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def state_decode(self, t):
		states = []
		for elem in t:
			states.append(self.state_decoder(elem))	
		return torch.stack(states)

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		states = self.state_decode(t)
		prototypes = self.prototype_decode(t)
		return states, prototypes
