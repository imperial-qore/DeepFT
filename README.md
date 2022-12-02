<h1 align="center">DeepFT</h1>
<div align="center">
  <a href="https://github.com/imperial-qore/DeepFT/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDeepFT&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/DeepFT/actions">
    <img src="https://github.com/imperial-qore/COSCO/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
</div>

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/DeepFT.git
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
