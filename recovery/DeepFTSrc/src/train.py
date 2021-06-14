from .constants import *
from .utils import *
import torch.nn as nn
from tqdm import tqdm
from .plotter import *

anomaly_loss = nn.MSELoss(reduction = 'none')
mse_loss = nn.MSELoss(reduction = 'mean')

# Model Training
def triplet_loss(anchor, positive_class, model):
	global PROTO_UPDATE_FACTOR
	positive_loss = mse_loss(anchor, model.prototype[positive_class].detach().clone())
	negative_class_list = list(range(K+1))
	negative_class_list.remove(positive_class)
	negative_loss = []
	for nc in negative_class_list:
		negative_loss.append(mse_loss(anchor, model.prototype[nc]))
	loss = positive_loss - torch.sum(torch.tensor(negative_loss))
	if positive_loss <= torch.min(torch.stack(negative_loss)):
		factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
		model.prototype[positive_class] = factor * anchor + (1 - factor) * model.prototype[positive_class]
	return loss

def custom_loss(model, pred_state, prototypes, true_state, thresholds):
	global PROTO_UPDATE_FACTOR
	aloss = mse_loss(pred_state.view(-1), torch.tensor(true_state, dtype=torch.double))
	tloss = torch.tensor(0, dtype=torch.double)
	anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), thresholds)
	for i, p in enumerate(prototypes):
		if anomaly_any_dim[0][i]: 
			devs = [mse_loss(p, model.prototype[k+1]).item() for k in range(K)]
			true_class = np.argmin(devs) + 1 # closest of k true prototypes
		else:
			true_class = 0
		tloss += triplet_loss(p, true_class, model)
	PROTO_UPDATE_FACTOR *= PROTO_FACTOR_DECAY
	return aloss, tloss

def backprop(epoch, model, optimizer, train_time_data, train_schedule_data, stats, norm_series, thresholds, training = True):
	global PROTO_UPDATE_FACTOR
	aloss_list, tloss_list = [], []
	for i in tqdm(range(train_time_data.shape[0]), leave=False, position=1):
		state, schedule = train_time_data[i], train_schedule_data[i]
		pred_state, prototypes = model(state, schedule)
		true_state = run_simulation(stats, schedule)
		true_state = normalize_test_time_data(true_state, norm_series)
		aloss, tloss = custom_loss(model, pred_state, prototypes, state, thresholds) # true_state
		aloss_list.append(aloss.item()); tloss_list.append(tloss.item())
		loss = aloss + 0.01 * tloss
		if training:
			optimizer.zero_grad(); loss.backward(); optimizer.step()
	tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(aloss_list)+np.mean(tloss_list)},\tALoss = {np.mean(aloss_list)},\tTLoss = {np.mean(tloss_list)}')
	factor = PROTO_UPDATE_FACTOR + PROTO_UPDATE_MIN
	return np.mean(aloss_list), np.mean(tloss_list), factor

# Accuracy 
def anomaly_accuracy(pred_state, target_anomaly, thresholds, model_plotter):
	correct = 0; res_list = []; tp, fp, tn, fn = 0, 0, 0, 0
	anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), thresholds)
	anomaly_any_dim = anomaly_any_dim[0] + 0
	for i, res in enumerate(anomaly_any_dim):
		res_list.append(res)
		if res == target_anomaly[i]:
			correct += 1
			if target_anomaly[i] == 1: tp += 1
			else: tn += 1
		else:
			if target_anomaly[i] == 1: fn += 1
			else: fp += 1
	model_plotter.update_anomaly(res_list, target_anomaly, correct/pred_state.shape[0])
	return correct/pred_state.shape[0], tp, tn, fp, fn

def class_accuracy(pred_state, prototypes, thresholds, model, model_plotter):
	correct, total = 0, 1e-4; proto_class = []
	anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), thresholds)
	anomaly_any_dim = anomaly_any_dim[0]
	for i, p in enumerate(prototypes):
		if anomaly_any_dim[i]: 
			devs = [mse_loss(p, model.prototype[k+1]).item() for k in range(K)]
			true_class = np.argmin(devs) + 1
		else:
			true_class = 0
		total += 1
		positive_loss = mse_loss(p, model.prototype[true_class])
		negative_class_list = list(range(K+1))
		negative_class_list.remove(true_class)
		negative_loss = []
		for nc in negative_class_list:
			negative_loss.append(mse_loss(p, model.prototype[nc]))
		if positive_loss <= torch.min(torch.stack(negative_loss)):
			correct += 1
		proto_class.append((p, true_class))
	model_plotter.update_class(proto_class, correct/total)
	return correct / total

def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, model_plotter):
	anomaly_correct, class_correct, class_total = 0, 0, 0; tpl, tnl, fpl, fnl = [], [], [], []
	for i, d in enumerate(train_time_data):
		pred_state, prototypes = model(train_time_data[i], train_schedule_data[i])
		model_plotter.update_lines(pred_state.view(-1), train_time_data[i][-1])
		res, tp, tn, fp, fn = anomaly_accuracy(pred_state, anomaly_data[i], thresholds, model_plotter)
		anomaly_correct += res
		tpl.append(tp); tnl.append(tn); fpl.append(fp); fnl.append(fn)
		tp += res; fp += res; tn += (1 - res); fn += (1 - res)
		if np.sum(anomaly_data[i]) > 0:
			class_total += 1
			class_correct += class_accuracy(pred_state, prototypes, thresholds, model, model_plotter)
	tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fn)
	p, r = tp/(tp+fp), tp/(tp+fn)
	tqdm.write(f'P = {p}, R = {r}, F1 = {2 * p * r / (p + r)}')
	return anomaly_correct / len(train_time_data), class_correct / class_total

# Optimization loss
def optimization_loss(pred_state, prototypes, thresholds):
	anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), thresholds)
	anomaly_any_dim = anomaly_any_dim[0] + 0
	end = np.sum(anomaly_any_dim) == 0
	z = pred_state - torch.tensor(thresholds, dtype=torch.double)
	z = torch.mean(torch.nn.ReLU(True)(z))
	return z, end