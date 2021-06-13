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
(1) State Encoder
(2) Decision Encoder
(3) Fault Score Predictor (FSP)
(4) Prototype Embedding Network (PEN)

## Motivation

PreGAN freezes its models as we dont have labeled data at the time of testing. 
But, we can use self-supervised learning to fine-tune the model and transformer model for unsupervised model training. The classification model does not need any true classes, only abstract classes/embeddings. Thus, we can make a model that needs no training/labelled data. It can be trained using unsupervised (for transformers) and self-supervised (for prototype embeddings) so that we can optimize it dynamically. Use this to train a loss -> update scheduling decision using backprop to input. 

## Pipeline

Joint training of anomaly detection and classification engines. 
(1) FSP - trained using unsupervised learning using reconstruction loss. 
	- co-simulated self-supervision.
	- joint training
(2) PEN - trained using 1 class for no-fault (NAP), k classes for faults (kP).
	- autoregressive training
	- factored prediction

Testing.
- loss = Fault score + Delta(P - NAP)
- S <- S - gamma * Nabla_S (loss)

Run till convergence or Fault Score < threshold.

## Implementation Details


## Figures and Comparisons


## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
