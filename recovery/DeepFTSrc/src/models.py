import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

## Simple Multi-Head Self-Attention Model
class DeepFT_16(nn.Module):
	def __init__(self):
		super(DeepFT_16, self).__init__()
		self.name = 'DeepFT_16'
		self.lr = 0.0008
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 3 # w_size = 5
		self.n_latent = 10
		self.n_hidden = 16
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.encoder = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats, self.n_hosts * self.n_latent), nn.LeakyReLU(True),
		)
		self.anomaly_decoder = nn.Sequential(
			nn.Linear(self.n_latent, 2), nn.Softmax(dim=0),
		)
		self.prototype_decoder = nn.Sequential(
			nn.Linear(self.n_latent, PROTO_DIM), nn.Sigmoid(),
		)
		self.prototype = [torch.rand(PROTO_DIM, requires_grad=False, dtype=torch.double) for _ in range(3)]

	def encode(self, t, s):
		t = self.encoder(t.view(-1)).view(self.n_hosts, self.n_latent)	
		return t

	def anomaly_decode(self, t):
		anomaly_scores = []
		for elem in t:
			anomaly_scores.append(self.anomaly_decoder(elem).view(1, -1))	
		return anomaly_scores

	def prototype_decode(self, t):
		prototypes = []
		for elem in t:
			prototypes.append(self.prototype_decoder(elem))	
		return prototypes

	def forward(self, t, s):
		t = self.encode(t, s)
		anomaly_scores = self.anomaly_decode(t)
		prototypes = self.prototype_decode(t)
		return anomaly_scores, prototypes
