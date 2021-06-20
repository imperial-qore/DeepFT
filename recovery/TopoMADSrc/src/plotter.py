import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import statistics
import os, glob
import random
import numpy as np
from sklearn.manifold import TSNE
import seaborn as sns
from .constants import *

plt.style.use(['science', 'ieee'])
plt.rcParams["text.usetex"] = False
plt.rcParams['figure.figsize'] = 6, 2

os.makedirs(plot_folder, exist_ok=True)

def smoother(y, box_pts=1):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class Model_Plotter():
	def __init__(self, env, modelname):
		self.env = env
		self.model_name = modelname
		self.n_hosts = int(modelname.split('_')[-1])
		self.folder = os.path.join(plot_folder, env, 'model')
		self.prefix = self.folder + '/' + self.model_name
		self.epoch = 0
		os.makedirs(self.folder, exist_ok=True)
		for f in glob.glob(self.folder + '/*'): os.remove(f)
		self.tsne = TSNE(n_components=2, perplexity=50, n_iter=1000)
		self.colors = ['r', 'g', 'b', 'gray', 'magenta', 'violet']
		plt.rcParams["font.family"] = "Maven Pro"
		self.init_params()

	def init_params(self):
		self.source_anomaly_scores = []
		self.target_anomaly_scores = []
		self.correct_series = []
		self.pred_line = []
		self.true_line = []

	def update_anomaly(self, source_anomaly, target_anomaly, correct):
		self.source_anomaly_scores.append(source_anomaly)
		self.target_anomaly_scores.append(target_anomaly.tolist())
		self.correct_series.append(correct)

	def update_lines(self, pred, true):
		self.pred_line.append(pred.tolist())
		self.true_line.append(true.tolist())

	def plot(self, accuracy_list, epoch):
		self.epoch = epoch; self.prefix2 = self.prefix + '_' + str(self.epoch) + '_'
		self.pred_line, self.true_line = np.array(self.pred_line).transpose(), np.array(self.true_line).transpose()
		self.aloss_list = [i[0] for i in accuracy_list]
		self.loss_list = [i[0] for i in accuracy_list]
		self.anomaly_score_list = [i[1] for i in accuracy_list]
		self.plot1('Loss', self.loss_list)
		self.plot1('Anomaly Loss', self.aloss_list)
		self.plot1('Anomaly Prediction Score', self.anomaly_score_list)
		self.plot1('Correct Anomaly', self.correct_series, xlabel='Timestamp')
		self.plotLine('Time Series', 'Pred Line', 'True Line', self.pred_line[0], self.true_line[0])
		self.source_anomaly_scores = np.array(self.source_anomaly_scores)
		self.target_anomaly_scores = np.array(self.target_anomaly_scores)
		self.plot_heatmap('Anomaly Scores', 'Prediction', 'Ground Truth', self.source_anomaly_scores, self.target_anomaly_scores)
		self.init_params()

	def plot1(self, name1, data1, smooth = True, xlabel='Epoch'):
		if smooth: data1 = smoother(data1)
		fig, ax = plt.subplots(1, 1)
		ax.set_ylabel(name1)
		ax.plot(data1, linewidth=0.2)
		ax.set_xlabel(xlabel)
		fig.savefig(self.prefix2 + f'{name1}.pdf')
		plt.close()

	def plot2(self, name1, name2, data1, data2, smooth = True, xlabel='Epoch'):
		if smooth: data1, data2 = smoother(data1), smoother(data2)
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		ax.set_ylabel(name1); ax.set_xlabel(xlabel)
		l1 = ax.plot(data1, linewidth=0.6, label=name1, c = 'red')
		ax2 = ax.twinx()
		l2 = ax2.plot(data2, '--', linewidth=0.6, alpha=0.8, label=name2)
		ax2.set_xlabel(xlabel)
		ax2.set_ylabel(name2)
		plt.legend(handles=l1+l2, loc=9, bbox_to_anchor=(0.5, 1.2), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf', pad_inches=0)
		plt.close()

	def plotLine(self, title, name1, name2, data1, data2, smooth = True, xlabel='Timestamp'):
		fig, ax = plt.subplots(1, 1, figsize=(3,1.9))
		ax.set_ylabel(title); ax.set_xlabel(xlabel)
		l1 = ax.plot(data1, linewidth=0.6, label=name1, c = 'red')
		l2 = ax.plot(data2.reshape(-1), '--', linewidth=0.6, alpha=0.8, label=name2)
		plt.legend(handles=l1+l2, loc=9, bbox_to_anchor=(0.5, 1.2), ncol=2, prop={'size': 7})
		fig.savefig(self.prefix2 + f'{name1}_{name2}.pdf', pad_inches=0)
		plt.close()

	def plot_heatmap(self, title, name1, name2, data1, data2):
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 1.8))
		ax1.set_title(title)
		yticks = np.linspace(0, self.n_hosts, 4, dtype=np.int)
		h1 = sns.heatmap(data1.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax1)
		h2 = sns.heatmap(data2.transpose(),cmap="YlGnBu", yticklabels=yticks, linewidth=0.01, ax = ax2)
		ax1.set_yticks(yticks); ax2.set_yticks(yticks); 
		xticks = np.linspace(0, data1.shape[0]-2, 5, dtype=np.int)
		ax1.set_xticks(xticks); ax2.set_xticks(xticks); ax2.set_xticklabels(xticks, rotation=0)
		ax1.set_xticklabels(xticks, rotation=0)
		ax2.set_xlabel('Timestamp')
		ax2.set_ylabel(name2); ax1.set_ylabel(name1)
		fig.savefig(self.prefix2 + f'{title}_{name1}_{name2}.pdf', bbox_inches = 'tight')
		plt.close()
