[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/PreGAN/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDeepFT&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/DeepFT/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/PreGAN.git
cd PreGAN/
```
Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
export PATH=$PATH:~/.local/bin
```
Change line 118 in `main.py` to use one of the implemented fault-tolerance techniques: `DeepFTRecovery`, `PCFTRecovery`, `DFTMRecovery`, `ECLBRecovery`,  `AWGGRecovery` or `TopoMADRecovery` and run the code using the following command.
```console
python3 main.py
````

## External Links
| Items | Contents | 
| --- | --- |
| **Pre-print** | (coming soon) |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship |

## Cite this work
Our work is accepted in IEEE Conference on Computer Communications (INFOCOM) 2023. Cite our work using the bibtex entry below.
```bibtex
@inproceedings{tuli2022deepft,
  title={{DeepFT: Fault-Tolerant Edge Computing using a Self-Supervised Deep Surrogate Model}},
  author={Tuli, Shreshth and Casale, Giuliano and Cherkasova, Ludmila and Jennings, Nicholas R},
  booktitle={IEEE Conference on Computer Communications (INFOCOM)},
  year={2023},
  organization={IEEE}
}

```

## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.
