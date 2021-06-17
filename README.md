[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/PreGAN/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDeepFT&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/DeepFT/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

# DeepFT

DeepFT: Self-Supervised Deep Learning based Surrogate Models for Fault-Tolerant Edge Computing.

Can be extended to semi-supervised = dynamically fine-tune using teacher forcing (need to solve the exposure bias problem).


## Model
Dual-headed neural model for parameter sharing and generalization.
1. State Encoder
2. Decision Encoder
3. State Decoder (SD)
4. Prototype Embedding Network (PEN)

## Motivation

PreGAN freezes its models as we dont have labeled data at the time of testing. 
But, we can use self-supervised learning to fine-tune the model and transformer model for unsupervised model training. The classification model does not need any true classes, only abstract classes/embeddings. Thus, we can make a model that needs no training/labelled data. It can be trained using unsupervised (for transformers) and self-supervised (for prototype embeddings) so that we can optimize it dynamically. Use this to train a loss -> update scheduling decision using backprop to input. 

## Pipeline

Joint training of anomaly detection and classification engines. 
At training time, we have a dataset of W and W' (next window).
1. SD - trained using unsupervised learning using reconstruction loss. 
	- reconstruction loss from W' (reconstructed output = W") = MSE(W', W")
	- reconstruct each dimension (i.e. each host feature vector independently)
	- joint training
2. PEN - trained using 1 class for no-fault (NAP), k classes for faults (kP).
	- autoregressive training (in that epoch, use POT to generate autoregressive fault labels)
	- trained using triplet loss
	- factored prediction (predict for each host - makes it agnostic to num hosts)

Autoregressive labelling: ! (Fault Score > POT and W' > W") -> means suddent upward spike.
Downward spiked are not considered anomalous. If the above is true then NAP is ground-truth else the kP
class to which the prediction is closest.

Testing. At test time we dont have W' so we use co-simulated self-supervision to generate W' and then the 
fault score. Also, we dont want to consider those hosts which have downward spikes. So we calculate
fault-score for each host i. And take dot-product with ReLU (W' > W").
- loss = \Sum\_i (MSE(W'\_i, W"\_i) + Delta(P - NAP)\_i) . ReLU(W'\_i > W"\_i)
- S <- S - gamma * Nabla_S (loss)

Run till convergence or Fault Score < threshold.

## Implementation Details

- Each prototype is (mu, sigma) as NAP is much more dense than other classes. Thus, we use Bregmann distance as
our Delta function in the triplet loss and the optimization loss.

## Figures and Comparisons

- Decrease and convergence of the optimization loss

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
